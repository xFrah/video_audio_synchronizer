import sys
import os
import json
import av
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from natsort import natsorted

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox,
    QGroupBox, QFormLayout, QTextEdit, QSpinBox, QRadioButton, QButtonGroup,
    QSlider, QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QCheckBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QRect
from PyQt6.QtGui import QCursor, QImage, QPixmap, QFont, QKeySequence, QShortcut

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def format_time(seconds):
    if seconds is None or seconds < 0:
        return "00:00:00.000"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "0s"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{seconds:.1f}s"


class FramePreviewPopup(QWidget):
    """Floating tooltip window showing hover preview thumbnail and time info."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 2px solid #00adb5;
                border-radius: 8px;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        self.lbl_title = QLabel("Time: 00:00:00.000")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #00adb5;")
        layout.addWidget(self.lbl_title)
        
        self.lbl_image = QLabel()
        self.lbl_image.setFixedSize(240, 135)
        self.lbl_image.setStyleSheet("background-color: #000000; border: 1px solid #444;")
        self.lbl_image.setScaledContents(True)
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_image)

    def update_preview(self, title_text, pixmap):
        self.lbl_title.setText(title_text)
        if pixmap and not pixmap.isNull():
            self.lbl_image.setPixmap(pixmap)
        else:
            self.lbl_image.setText("No Frame Available")


class AudioWaveformWorker(QThread):
    """Decodes audio track into a downsampled waveform amplitude array."""
    waveform_ready = pyqtSignal(object, object)  # times, amplitudes
    progress_signal = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, target_samples=2000):
        super().__init__()
        self.video_path = video_path
        self.target_samples = target_samples

    def run(self):
        try:
            self.progress_signal.emit(10, "Opening media for audio waveform extraction...")
            container = av.open(self.video_path)
            if not container.streams.audio:
                self.error_signal.emit("No audio stream found in media file.")
                container.close()
                return

            audio_stream = container.streams.audio[0]
            
            # Determine total duration
            duration = 0
            if container.duration:
                duration = container.duration / 1000000.0
            elif audio_stream.duration and audio_stream.time_base:
                duration = float(audio_stream.duration * audio_stream.time_base)

            self.progress_signal.emit(30, "Decoding audio frames...")
            raw_samples = []
            
            # Decode audio packets
            sample_count = 0
            max_decode_samples = 5000000  # Cap safety limit
            
            for frame in container.decode(audio_stream):
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=0)  # Mix down to mono
                
                # Downsample chunk locally if very large
                if len(arr) > 1000:
                    step = len(arr) // 200
                    arr = arr[::step]
                    
                raw_samples.append(np.abs(arr))
                sample_count += len(arr)
                if sample_count > max_decode_samples:
                    break

            container.close()

            if not raw_samples:
                self.error_signal.emit("Could not decode audio samples.")
                return

            full_audio = np.concatenate(raw_samples)
            if len(full_audio) == 0:
                self.error_signal.emit("Empty audio track.")
                return

            # Bin into target_samples bins
            self.progress_signal.emit(80, "Generating waveform envelope...")
            num_bins = min(self.target_samples, len(full_audio))
            bin_size = len(full_audio) // num_bins
            
            reshaped = full_audio[:num_bins * bin_size].reshape(num_bins, bin_size)
            rms = np.sqrt(np.mean(reshaped ** 2, axis=1))
            
            # Normalize waveform 0 to 1
            max_val = np.max(rms)
            if max_val > 0:
                rms = rms / max_val

            if duration <= 0:
                duration = 1.0

            times = np.linspace(0, duration, num_bins)
            self.progress_signal.emit(100, "Audio waveform loaded.")
            self.waveform_ready.emit(times, rms)

        except Exception as e:
            self.error_signal.emit(f"Failed to generate waveform: {str(e)}")


class SplitWorker(QThread):
    """Executes FFmpeg stream copy (-c copy) to split episodes without re-encoding."""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, input_file, output_folder, splits, naming_pattern, use_keyframe_snap=True):
        super().__init__()
        self.input_file = input_file
        self.output_folder = output_folder
        self.splits = sorted(splits)  # list of (start_time, end_time, episode_idx)
        self.naming_pattern = naming_pattern
        self.use_keyframe_snap = use_keyframe_snap

    def run(self):
        try:
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            ext = os.path.splitext(self.input_file)[1]
            if not ext:
                ext = ".mkv"

            total = len(self.splits)
            for idx, (start_t, end_t, part_num) in enumerate(self.splits, 1):
                # Build output filename using pattern template
                out_name = self.naming_pattern.replace("{basename}", base_name)
                out_name = out_name.replace("{index:02d}", f"{part_num:02d}")
                out_name = out_name.replace("{index}", str(part_num))
                out_name = out_name.replace("{part:02d}", f"{part_num:02d}")
                out_name = out_name.replace("{part}", str(part_num))
                out_name = out_name.replace("{ep}", f"Part{part_num:02d}")
                
                if not out_name.lower().endswith(ext.lower()):
                    out_name += ext

                out_path = os.path.join(self.output_folder, out_name)
                
                self.log_signal.emit(f"Splitting Part {part_num}/{total}: {format_time(start_t)} -> {format_time(end_t)}")
                self.log_signal.emit(f"Target Output: {out_path}")
                self.progress_signal.emit(idx - 1, total, f"Exporting Part {part_num} of {total}...")

                # Construct FFmpeg command
                cmd = ["ffmpeg", "-y"]
                
                if self.use_keyframe_snap:
                    # -ss before -i seeks fast to keyframe
                    cmd.extend(["-ss", f"{start_t:.3f}", "-to", f"{end_t:.3f}", "-i", self.input_file])
                else:
                    # -ss after -i cuts precisely
                    cmd.extend(["-i", self.input_file, "-ss", f"{start_t:.3f}", "-to", f"{end_t:.3f}"])

                cmd.extend(["-c", "copy", "-map", "0", out_path])

                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    startupinfo=startupinfo
                )
                
                _, stderr = process.communicate()
                
                if process.returncode != 0:
                    self.log_signal.emit(f"FFmpeg Warning/Error on Part {part_num}:\n{stderr}")
                else:
                    self.log_signal.emit(f"Successfully saved: {out_name}")

            self.progress_signal.emit(total, total, "Splitting complete!")
            self.finished_signal.emit(True, f"All {total} parts successfully split!")

        except Exception as e:
            self.finished_signal.emit(False, f"Splitting failed: {str(e)}")


class AudioTimelineCanvas(FigureCanvas):
    """Matplotlib Figure Canvas for the Sony-Vegas style Audio Timeline."""
    clicked_signal = pyqtSignal(float)
    hover_signal = pyqtSignal(float, int, int)
    leave_signal = pyqtSignal()

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 2.2), dpi=100)
        self.fig.patch.set_facecolor('#202124')
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2d2e31')

        self.times = None
        self.amplitudes = None
        self.duration = 1.0
        self.playhead_time = 0.0
        self.split_points = []

        self.style_axes()

        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_hover)
        self.mpl_connect('axes_leave_event', self.on_leave)

    def style_axes(self):
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')
        self.ax.get_yaxis().set_visible(False)

    def set_waveform(self, times, amplitudes, duration):
        self.times = times
        self.amplitudes = amplitudes
        self.duration = duration
        self.redraw()

    def set_playhead(self, t):
        self.playhead_time = t
        self.redraw()

    def set_split_points(self, splits):
        self.split_points = sorted(splits)
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#2d2e31')
        self.style_axes()

        if self.times is not None and len(self.times) > 0:
            self.ax.fill_between(self.times, 0, self.amplitudes, color='#00adb5', alpha=0.75, label='Audio Waveform')
            self.ax.plot(self.times, self.amplitudes, color='#00fff5', linewidth=1.0)
            
            self.ax.set_xlim(0, self.duration)
            self.ax.set_ylim(0, 1.1)

            self.ax.set_xlabel("Timeline (Seconds)", color="white", fontsize=9)

            for idx, pt in enumerate(self.split_points, 1):
                self.ax.axvline(pt, color='#ffde59', linestyle='--', linewidth=2.0, zorder=8)
                self.ax.text(pt, 0.95, f" Part Split {idx}", color='#ffde59', fontweight='bold', fontsize=9, zorder=9)

            self.ax.axvline(self.playhead_time, color='#ff3366', linewidth=2.5, zorder=10)

        self.fig.tight_layout()
        self.draw()

    def on_click(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            t_clicked = float(event.xdata)
            t_clicked = np.clip(t_clicked, 0, self.duration)
            self.clicked_signal.emit(t_clicked)

    def on_hover(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            t_hover = float(event.xdata)
            t_hover = np.clip(t_hover, 0, self.duration)
            
            pos = self.mapToGlobal(QPoint(int(event.x), self.height() - int(event.y)))
            self.hover_signal.emit(t_hover, pos.x(), pos.y())
        else:
            self.leave_signal.emit()

    def on_leave(self, event):
        self.leave_signal.emit()


class EpisodeSplitterApp(QMainWindow):
    """Main PyQt6 Application Window for Episode Splitter."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Episode Splitter - Lossless Video Trimmer")
        self.resize(1150, 920)

        self.settings_file = "splitter_settings.json"
        
        self.app_mode = "folder"
        self.input_folder_path = None
        self.output_folder_path = None
        self.video_files = []
        self.current_index = 0
        
        self.current_video_path = None
        self.container = None
        self.video_stream = None
        self.duration = 0.0
        self.fps = 25.0
        self.total_frames = 0
        self.current_time = 0.0
        self.current_frame_idx = 0
        
        self.split_points = []
        self.audio_waveform_worker = None
        self.split_worker = None
        self.audio_times = None
        self.audio_amplitudes = None

        self.play_timer = QTimer()
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self.advance_playback)
        self.is_playing = False

        self.preview_popup = FramePreviewPopup(self)

        self.init_ui()
        self.load_settings()
        self.setup_keyboard_shortcuts()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #202124;
            }
            QWidget {
                color: #e8eaed;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #3c4043;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                background-color: #292a2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                color: #00adb5;
            }
            QPushButton {
                background-color: #3c4043;
                color: #ffffff;
                border: 1px solid #5f6368;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4e51;
                border-color: #00adb5;
            }
            QPushButton:disabled {
                background-color: #2a2b2e;
                color: #70757a;
                border-color: #3c4043;
            }
            QPushButton#btn_action {
                background-color: #00adb5;
                color: #ffffff;
                border: none;
                font-size: 14px;
                padding: 8px 18px;
            }
            QPushButton#btn_action:hover {
                background-color: #00fff5;
                color: #121212;
            }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #5f6368;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #3c4043;
                border: 1px solid #3c4043;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #292a2d;
                color: #00adb5;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #3c4043;
            }
            QProgressBar {
                border: 1px solid #5f6368;
                border-radius: 5px;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #00adb5;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3c4043;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                color: #d1d5db;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # 1. Mode Selection & Input / Output Folder Controls
        files_group = QGroupBox("File & Folder Configuration")
        files_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        self.btn_group_mode = QButtonGroup(self)
        self.radio_folder = QRadioButton("Folder Mode (Batch Process Multiple Videos)")
        self.radio_file = QRadioButton("Single File Mode")
        self.radio_folder.setChecked(True)
        
        self.btn_group_mode.addButton(self.radio_folder, 0)
        self.btn_group_mode.addButton(self.radio_file, 1)
        mode_layout.addWidget(self.radio_folder)
        mode_layout.addWidget(self.radio_file)
        mode_layout.addStretch()
        self.radio_folder.toggled.connect(self.on_mode_toggled)
        files_layout.addLayout(mode_layout)

        h1 = QHBoxLayout()
        self.btn_input = QPushButton("Select Input Folder")
        self.btn_input.clicked.connect(self.select_input)
        self.lbl_input = QLabel("No input selected")
        self.lbl_input.setStyleSheet("color: #bdc1c6;")
        h1.addWidget(self.btn_input)
        h1.addWidget(self.lbl_input, stretch=1)

        self.btn_output = QPushButton("Select Output Folder")
        self.btn_output.clicked.connect(self.select_output)
        self.lbl_output = QLabel("No output folder selected")
        self.lbl_output.setStyleSheet("color: #bdc1c6;")
        h1.addWidget(self.btn_output)
        h1.addWidget(self.lbl_output, stretch=1)
        files_layout.addLayout(h1)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Previous Video")
        self.btn_prev.clicked.connect(self.prev_file)
        self.btn_prev.setEnabled(False)

        self.lbl_file_nav = QLabel("No Video Loaded")
        self.lbl_file_nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_file_nav.setStyleSheet("font-weight: bold; font-size: 14px; color: #00adb5;")

        self.btn_next = QPushButton("Next Video ▶")
        self.btn_next.clicked.connect(self.next_file)
        self.btn_next.setEnabled(False)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_file_nav, stretch=1)
        nav_layout.addWidget(self.btn_next)
        files_layout.addLayout(nav_layout)

        self.lbl_meta = QLabel("Metadata: Ready")
        self.lbl_meta.setStyleSheet("color: #9aa0a6; font-style: italic;")
        files_layout.addWidget(self.lbl_meta)

        files_group.setLayout(files_layout)
        main_layout.addWidget(files_group)

        # 2. GIANT FRAME VIEWER
        frame_group = QGroupBox("Giant Frame Viewer")
        frame_layout = QVBoxLayout()
        
        self.lbl_frame_viewer = QLabel("No Video Loaded")
        self.lbl_frame_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_frame_viewer.setMinimumHeight(340)
        self.lbl_frame_viewer.setStyleSheet("""
            QLabel {
                background-color: #0d0d0e;
                border: 2px solid #3c4043;
                border-radius: 6px;
                font-size: 16px;
                color: #70757a;
            }
        """)
        frame_layout.addWidget(self.lbl_frame_viewer)

        info_line = QHBoxLayout()
        self.lbl_timecode = QLabel("⏱ Time: 00:00:00.000")
        self.lbl_timecode.setStyleSheet("font-size: 14px; font-weight: bold; color: #00fff5;")
        self.lbl_frame_num = QLabel("Frame: #0 / 0")
        self.lbl_frame_num.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        
        info_line.addWidget(self.lbl_timecode)
        info_line.addStretch()
        info_line.addWidget(self.lbl_frame_num)
        frame_layout.addLayout(info_line)

        ctrl_layout = QHBoxLayout()
        
        self.btn_sub10s = QPushButton("≪ -10s")
        self.btn_sub10s.clicked.connect(lambda: self.seek_relative(-10.0))
        
        self.btn_sub1s = QPushButton("＜ -1s")
        self.btn_sub1s.clicked.connect(lambda: self.seek_relative(-1.0))

        self.btn_prev_frame = QPushButton("‹ -1 Frame")
        self.btn_prev_frame.clicked.connect(lambda: self.step_frame(-1))

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setStyleSheet("background-color: #00adb5; color: #ffffff;")
        self.btn_play.clicked.connect(self.toggle_play)

        self.btn_next_frame = QPushButton("+1 Frame ›")
        self.btn_next_frame.clicked.connect(lambda: self.step_frame(1))

        self.btn_add1s = QPushButton("+1s ＞")
        self.btn_add1s.clicked.connect(lambda: self.seek_relative(1.0))

        self.btn_add10s = QPushButton("+10s ≫")
        self.btn_add10s.clicked.connect(lambda: self.seek_relative(10.0))

        ctrl_layout.addWidget(self.btn_sub10s)
        ctrl_layout.addWidget(self.btn_sub1s)
        ctrl_layout.addWidget(self.btn_prev_frame)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_next_frame)
        ctrl_layout.addWidget(self.btn_add1s)
        ctrl_layout.addWidget(self.btn_add10s)

        ctrl_layout.addSpacing(20)
        self.btn_add_split = QPushButton("✂ Add Split Point Here")
        self.btn_add_split.setStyleSheet("background-color: #ffde59; color: #121212; font-weight: bold;")
        self.btn_add_split.clicked.connect(self.add_split_point)
        
        self.btn_clear_splits = QPushButton("❌ Clear Splits")
        self.btn_clear_splits.clicked.connect(self.clear_split_points)
        
        ctrl_layout.addWidget(self.btn_add_split)
        ctrl_layout.addWidget(self.btn_clear_splits)

        frame_layout.addLayout(ctrl_layout)
        frame_group.setLayout(frame_layout)
        main_layout.addWidget(frame_group, stretch=2)

        # 3. INTERACTIVE AUDIO TIMELINE CANVAS
        timeline_group = QGroupBox("Audio Waveform Timeline (Click to Seek / Hover to Preview)")
        timeline_layout = QVBoxLayout()

        self.timeline_canvas = AudioTimelineCanvas(self)
        self.timeline_canvas.clicked_signal.connect(self.seek_to_time)
        self.timeline_canvas.hover_signal.connect(self.on_timeline_hover)
        self.timeline_canvas.leave_signal.connect(self.on_timeline_leave)
        
        timeline_layout.addWidget(self.timeline_canvas)
        timeline_group.setLayout(timeline_layout)
        main_layout.addWidget(timeline_group, stretch=1)

        # 4. EPISODE SEGMENTS TABLE & EXPORT OPTIONS
        bottom_layout = QHBoxLayout()

        segments_group = QGroupBox("Part Cut Segments")
        segments_layout = QVBoxLayout()

        self.table_segments = QTableWidget()
        self.table_segments.setColumnCount(4)
        self.table_segments.setHorizontalHeaderLabels(["Part", "Start Time", "End Time", "Duration"])
        self.table_segments.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_segments.setMaximumHeight(140)
        segments_layout.addWidget(self.table_segments)
        segments_group.setLayout(segments_layout)
        bottom_layout.addWidget(segments_group, stretch=2)

        export_group = QGroupBox("Lossless Export Settings")
        export_layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.txt_naming = QLineEdit("{basename}_Part{index:02d}.mkv")
        self.chk_keyframe_snap = QCheckBox("Fast Keyframe Snap (-c copy)")
        self.chk_keyframe_snap.setChecked(True)
        self.chk_keyframe_snap.setToolTip("Stream-copy splits instantly on nearest keyframe without re-encoding.")
        
        form_layout.addRow("Naming Template:", self.txt_naming)
        form_layout.addRow("", self.chk_keyframe_snap)
        export_layout.addLayout(form_layout)

        self.btn_split_export = QPushButton("✂ Split & Save Parts (No Re-encoding)")
        self.btn_split_export.setObjectName("btn_action")
        self.btn_split_export.clicked.connect(self.start_splitting)
        export_layout.addWidget(self.btn_split_export)

        export_group.setLayout(export_layout)
        bottom_layout.addWidget(export_group, stretch=1)

        main_layout.addLayout(bottom_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(90)
        main_layout.addWidget(self.log_text)

    def setup_keyboard_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step_frame(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step_frame(1))
        QShortcut(QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_Left), self, lambda: self.seek_relative(-1.0))
        QShortcut(QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_Right), self, lambda: self.seek_relative(1.0))

    def log(self, msg):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_mode_toggled(self):
        new_mode = "folder" if self.radio_folder.isChecked() else "file"
        if new_mode == self.app_mode:
            return
        self.app_mode = new_mode
        self.input_folder_path = None
        self.video_files = []
        self.current_index = 0
        
        self.btn_input.setText("Select Input File" if self.app_mode == "file" else "Select Input Folder")
        self.lbl_input.setText("No input selected")
        self.update_file_navigation()
        self.save_settings()

    def select_input(self):
        if self.app_mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
            if path:
                self.input_folder_path = path
                self.video_files = [path]
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
            if path:
                self.input_folder_path = path
                self.video_files = self.get_media_files(path)

        if self.video_files:
            self.lbl_input.setText(self.input_folder_path)
            self.current_index = 0
            self.load_video_at_index(0)
        else:
            self.lbl_input.setText("No valid video files found!")
            
        self.save_settings()

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if path:
            self.output_folder_path = path
            self.lbl_output.setText(path)
            self.save_settings()

    def get_media_files(self, folder):
        exts = ('.mp4', '.mkv', '.avi', '.mov')
        files = []
        for f in os.listdir(folder):
            if f.lower().endswith(exts):
                files.append(os.path.join(folder, f))
        return natsorted(files)

    def update_file_navigation(self):
        total = len(self.video_files)
        if total == 0:
            self.lbl_file_nav.setText("No Video Loaded")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
        else:
            curr_name = os.path.basename(self.video_files[self.current_index])
            self.lbl_file_nav.setText(f"File {self.current_index + 1} of {total}: {curr_name}")
            self.btn_prev.setEnabled(self.current_index > 0)
            self.btn_next.setEnabled(self.current_index < total - 1)

    def next_file(self):
        if self.current_index < len(self.video_files) - 1:
            self.load_video_at_index(self.current_index + 1)

    def prev_file(self):
        if self.current_index > 0:
            self.load_video_at_index(self.current_index - 1)

    def load_video_at_index(self, index):
        if not self.video_files or index < 0 or index >= len(self.video_files):
            return
        
        self.current_index = index
        self.update_file_navigation()
        video_path = self.video_files[index]
        self.open_video(video_path)

    def open_video(self, video_path):
        self.stop_playback()
        self.current_video_path = video_path
        self.split_points = []
        self.timeline_canvas.set_split_points([])
        self.update_segments_table()

        try:
            if self.container:
                self.container.close()
                
            self.container = av.open(video_path)
            self.video_stream = self.container.streams.video[0]
            
            w = self.video_stream.width
            h = self.video_stream.height
            self.fps = float(self.video_stream.average_rate or 25.0)
            
            if self.video_stream.duration and self.video_stream.time_base:
                self.duration = float(self.video_stream.duration * self.video_stream.time_base)
            elif self.container.duration:
                self.duration = self.container.duration / 1000000.0
            else:
                self.duration = 1.0

            self.total_frames = self.video_stream.frames
            if not self.total_frames or self.total_frames == 0:
                self.total_frames = int(self.duration * self.fps)

            meta_str = f"Res: {w}x{h} | FPS: {self.fps:.2f} | Duration: {format_duration(self.duration)} | Est. Frames: {self.total_frames}"
            self.lbl_meta.setText(meta_str)
            self.log(f"Loaded Video: {os.path.basename(video_path)} ({meta_str})")

            self.current_time = 0.0
            self.current_frame_idx = 0
            self.render_frame_at_time(0.0)

            self.log("Extracting audio waveform timeline...")
            self.audio_waveform_worker = AudioWaveformWorker(video_path)
            self.audio_waveform_worker.waveform_ready.connect(self.on_waveform_ready)
            self.audio_waveform_worker.error_signal.connect(lambda e: self.log(f"Waveform Error: {e}"))
            self.audio_waveform_worker.start()

            self.update_segments_table()

        except Exception as e:
            self.log(f"Error opening video {video_path}: {str(e)}")

    def on_waveform_ready(self, times, amplitudes):
        self.audio_times = times
        self.audio_amplitudes = amplitudes
        self.timeline_canvas.set_waveform(times, amplitudes, self.duration)
        self.timeline_canvas.set_playhead(self.current_time)
        self.log("Audio waveform loaded on timeline canvas.")

    def render_frame_at_time(self, timestamp):
        if not self.container or not self.video_stream:
            return

        try:
            timestamp = np.clip(timestamp, 0, self.duration)
            time_base = self.video_stream.time_base
            target_pts = int(timestamp / float(time_base))

            self.container.seek(target_pts, stream=self.video_stream, backward=True)

            target_frame = None
            for frame in self.container.decode(self.video_stream):
                target_frame = frame
                if frame.pts and frame.pts >= target_pts:
                    break

            if target_frame is None:
                return

            img = target_frame.to_image()
            qimg = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.lbl_frame_viewer.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_frame_viewer.setPixmap(scaled_pixmap)

            self.current_time = timestamp
            self.current_frame_idx = int(timestamp * self.fps)
            
            self.lbl_timecode.setText(f"⏱ Time: {format_time(self.current_time)}")
            self.lbl_frame_num.setText(f"Frame: #{self.current_frame_idx} / {self.total_frames}")
            self.timeline_canvas.set_playhead(self.current_time)

        except Exception:
            pass

    def seek_to_time(self, t):
        self.stop_playback()
        self.render_frame_at_time(t)

    def seek_relative(self, delta_seconds):
        self.stop_playback()
        new_time = np.clip(self.current_time + delta_seconds, 0, self.duration)
        self.render_frame_at_time(new_time)

    def step_frame(self, frame_delta):
        self.stop_playback()
        frame_time_step = 1.0 / (self.fps if self.fps > 0 else 25.0)
        new_time = np.clip(self.current_time + (frame_delta * frame_time_step), 0, self.duration)
        self.render_frame_at_time(new_time)

    def toggle_play(self):
        if self.is_playing:
            self.stop_playback()
        else:
            if not self.current_video_path:
                return
            self.is_playing = True
            self.btn_play.setText("⏸ Pause")
            self.play_timer.start()

    def stop_playback(self):
        self.is_playing = False
        self.play_timer.stop()
        self.btn_play.setText("▶ Play")

    def advance_playback(self):
        if not self.is_playing:
            return
        frame_time_step = 1.0 / (self.fps if self.fps > 0 else 25.0)
        new_time = self.current_time + frame_time_step
        if new_time >= self.duration:
            self.stop_playback()
            new_time = self.duration
        self.render_frame_at_time(new_time)

    def on_timeline_hover(self, t_hover, screen_x, screen_y):
        if not self.container or not self.video_stream:
            self.preview_popup.hide()
            return

        pix = self.get_quick_thumbnail(t_hover)
        title = f"Hover: {format_time(t_hover)}"
        self.preview_popup.update_preview(title, pix)
        self.preview_popup.move(screen_x + 15, screen_y + 15)
        self.preview_popup.show()
        self.preview_popup.raise_()

    def on_timeline_leave(self):
        self.preview_popup.hide()

    def get_quick_thumbnail(self, timestamp):
        try:
            time_base = self.video_stream.time_base
            target_pts = int(timestamp / float(time_base))
            self.container.seek(target_pts, stream=self.video_stream, backward=True)

            for frame in self.container.decode(self.video_stream):
                small_frame = frame.reformat(width=240, height=135, format="rgb24")
                img_bytes = bytes(small_frame.to_ndarray())
                qimg = QImage(img_bytes, 240, 135, 240 * 3, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(qimg)
        except Exception:
            return None
        return None

    def add_split_point(self):
        if self.duration <= 0:
            return
        t = round(self.current_time, 3)
        if t <= 0 or t >= self.duration:
            return
        if t not in self.split_points:
            self.split_points.append(t)
            self.split_points.sort()
            self.timeline_canvas.set_split_points(self.split_points)
            self.update_segments_table()
            self.log(f"Added Split Point at {format_time(t)}")

    def clear_split_points(self):
        self.split_points = []
        self.timeline_canvas.set_split_points([])
        self.update_segments_table()
        self.log("Cleared all split points.")

    def calculate_segments(self):
        points = [0.0] + self.split_points + [self.duration]
        segments = []
        for idx in range(len(points) - 1):
            start = points[idx]
            end = points[idx + 1]
            if end - start > 0.1:
                segments.append((start, end, idx + 1))
        return segments

    def update_segments_table(self):
        segments = self.calculate_segments()
        self.table_segments.setRowCount(len(segments))

        for row, (start, end, part_num) in enumerate(segments):
            dur = end - start
            self.table_segments.setItem(row, 0, QTableWidgetItem(f"Part {part_num}"))
            self.table_segments.setItem(row, 1, QTableWidgetItem(format_time(start)))
            self.table_segments.setItem(row, 2, QTableWidgetItem(format_time(end)))
            self.table_segments.setItem(row, 3, QTableWidgetItem(format_duration(dur)))

    def start_splitting(self):
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            self.log("ERROR: No valid video loaded.")
            return

        if not self.output_folder_path:
            self.log("ERROR: Please select an Output Folder first.")
            return

        segments = self.calculate_segments()
        if not segments:
            self.log("ERROR: No cut segments to split.")
            return

        naming_template = self.txt_naming.text().strip() or "{basename}_Part{index:02d}.mkv"
        use_keyframe = self.chk_keyframe_snap.isChecked()

        self.btn_split_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log(f"Starting Lossless Split ({len(segments)} Parts)...")

        self.split_worker = SplitWorker(
            self.current_video_path,
            self.output_folder_path,
            segments,
            naming_template,
            use_keyframe_snap=use_keyframe
        )
        self.split_worker.log_signal.connect(self.log)
        self.split_worker.progress_signal.connect(self.update_progress)
        self.split_worker.finished_signal.connect(self.on_split_finished)
        self.split_worker.start()

    def update_progress(self, current, total, status):
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)

    def on_split_finished(self, success, message):
        self.btn_split_export.setEnabled(True)
        self.progress_bar.setValue(100)
        if success:
            self.log(f"SUCCESS: {message}")
        else:
            self.log(f"ERROR: {message}")

    def load_settings(self):
        if not os.path.exists(self.settings_file):
            return
        try:
            with open(self.settings_file, "r") as f:
                data = json.load(f)

            mode = data.get("app_mode", "folder")
            if mode == "file":
                self.radio_file.setChecked(True)
            else:
                self.radio_folder.setChecked(True)

            input_path = data.get("input_path")
            if input_path and os.path.exists(input_path):
                self.input_folder_path = input_path
                self.lbl_input.setText(input_path)
                if os.path.isdir(input_path):
                    self.video_files = self.get_media_files(input_path)
                else:
                    self.video_files = [input_path]
                
                if self.video_files:
                    self.load_video_at_index(0)

            out_path = data.get("output_path")
            if out_path and os.path.exists(out_path):
                self.output_folder_path = out_path
                self.lbl_output.setText(out_path)

            naming = data.get("naming_template")
            if naming:
                self.txt_naming.setText(naming)

        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        try:
            settings = {
                "app_mode": self.app_mode,
                "input_path": self.input_folder_path,
                "output_path": self.output_folder_path,
                "naming_template": self.txt_naming.text()
            }
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_time > 0 and self.container:
            self.render_frame_at_time(self.current_time)


def main():
    app = QApplication(sys.argv)
    window = EpisodeSplitterApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
