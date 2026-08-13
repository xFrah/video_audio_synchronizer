import sys
import os
import glob
import json
import av
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy import signal, interpolate
from natsort import natsorted

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, 
    QGroupBox, QFormLayout, QTextEdit, QSpinBox, QRadioButton, QButtonGroup,
    QSlider, QToolTip
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QCursor, QImage, QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def format_time(seconds):
    if seconds is None or seconds < 0:
        return "00:00:00.000"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"


def get_thumbnail(thumbnails, target_time):
    if not thumbnails:
        return None
    times = [item[0] for item in thumbnails]
    idx = np.searchsorted(times, target_time)
    idx = np.clip(idx, 0, len(thumbnails) - 1)
    
    _, img_bytes, w, h = thumbnails[idx]
    qimg = QImage(img_bytes, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class FramePreviewPopup(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("""
            QWidget {
                background-color: #202124;
                border: 1px solid #5f6368;
                border-radius: 6px;
                color: white;
                font-family: sans-serif;
            }
            QLabel {
                border: none;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        self.lbl_header = QLabel("")
        self.lbl_header.setStyleSheet("font-weight: bold; color: #e8eaed; font-size: 11px;")
        self.lbl_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_header)
        
        imgs_layout = QHBoxLayout()
        imgs_layout.setSpacing(8)
        
        # Audio Source Container
        box_audio = QVBoxLayout()
        lbl_a_title = QLabel("🎵 Audio Source Frame")
        lbl_a_title.setStyleSheet("color: #8ab4f8; font-size: 10px; font-weight: bold;")
        lbl_a_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_audio = QLabel()
        self.lbl_img_audio.setFixedSize(200, 112)
        self.lbl_img_audio.setStyleSheet("background-color: #000; border: 1px solid #3c4043;")
        self.lbl_img_audio.setScaledContents(True)
        box_audio.addWidget(lbl_a_title)
        box_audio.addWidget(self.lbl_img_audio)
        
        # Video Source Container
        box_video = QVBoxLayout()
        lbl_v_title = QLabel("🎬 Video Source Frame")
        lbl_v_title.setStyleSheet("color: #81c995; font-size: 10px; font-weight: bold;")
        lbl_v_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_video = QLabel()
        self.lbl_img_video.setFixedSize(200, 112)
        self.lbl_img_video.setStyleSheet("background-color: #000; border: 1px solid #3c4043;")
        self.lbl_img_video.setScaledContents(True)
        box_video.addWidget(lbl_v_title)
        box_video.addWidget(self.lbl_img_video)
        
        imgs_layout.addLayout(box_audio)
        imgs_layout.addLayout(box_video)
        layout.addLayout(imgs_layout)
        
        self.lbl_footer = QLabel("")
        self.lbl_footer.setStyleSheet("color: #9aa0a6; font-size: 10px;")
        self.lbl_footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_footer)

    def update_preview(self, header_text, pix_audio, pix_video, footer_text):
        self.lbl_header.setText(header_text)
        self.lbl_footer.setText(footer_text)
        if pix_audio and not pix_audio.isNull():
            self.lbl_img_audio.setPixmap(pix_audio)
        else:
            self.lbl_img_audio.clear()
            
        if pix_video and not pix_video.isNull():
            self.lbl_img_video.setPixmap(pix_video)
        else:
            self.lbl_img_video.clear()
        self.adjustSize()


def get_color_percentages(image):
    total_pixels = image.shape[0] * image.shape[1]
    max_intensity = total_pixels * 255.0
    b_sum = np.sum(image[:, :, 0])
    g_sum = np.sum(image[:, :, 1])
    r_sum = np.sum(image[:, :, 2])
    r = r_sum / max_intensity
    g = g_sum / max_intensity
    b = b_sum / max_intensity
    
    # Smooth Chromatic Balance (0.0 = Cool/Blue, 0.5 = Neutral, 1.0 = Warm/Red)
    total = r + g + b + 1e-6
    color_dir = ((r / total) - (b / total) + 1.0) / 2.0
    return r, g, b, color_dir


def find_time_shift(t1, y1, t2, y2, scale, log_callback=None):
    t1 = np.array(t1)
    y1 = np.array(y1)
    t2 = np.array(t2)
    y2 = np.array(y2)
    
    t2_scaled = t2 * scale
    fs = 1000.0
    
    t1_uniform = np.arange(t1[0], t1[-1], 1/fs)
    interp1 = interpolate.interp1d(t1, y1, kind='linear', fill_value="extrapolate")
    y1_uniform = interp1(t1_uniform)
    y1_centered = y1_uniform - np.mean(y1_uniform)
    
    t2_uniform = np.arange(t2_scaled[0], t2_scaled[-1], 1/fs)
    interp2 = interpolate.interp1d(t2_scaled, y2, kind='linear', fill_value="extrapolate")
    y2_uniform = interp2(t2_uniform)
    y2_centered = y2_uniform - np.mean(y2_uniform)
    
    corr = signal.correlate(y1_centered, y2_centered, mode='full')
    lags = signal.correlation_lags(len(y1_centered), len(y2_centered), mode='full')
    
    lag_idx = np.argmax(corr)
    lag_samples = lags[lag_idx]
    coarse_shift = lag_samples / fs
    
    msg = f"Coarse shift detected: {coarse_shift:.4f}s"
    if log_callback:
        log_callback(msg)
    else:
        print(msg)

    fine_shifts = np.linspace(coarse_shift - 0.1, coarse_shift + 0.1, 201)
    best_mse = float('inf')
    best_shift = coarse_shift
    f1 = interpolate.interp1d(t1, y1, kind='linear', fill_value="extrapolate")
    
    for s in fine_shifts:
        t2_shifted = t2_scaled + s
        start_t = max(t1[0], t2_shifted[0])
        end_t = min(t1[-1], t2_shifted[-1])
        
        if end_t <= start_t:
            continue
            
        t_eval = np.arange(start_t, end_t, 0.01)
        if len(t_eval) < 10: continue
        
        v1 = f1(t_eval)
        f2 = interpolate.interp1d(t2_scaled, y2, kind='linear', fill_value="extrapolate")
        v2 = f2(t_eval - s)
        
        mse = np.mean((v1 - v2)**2)
        if mse < best_mse:
            best_mse = mse
            best_shift = s
            
    msg = f"Fine-tuned shift: {best_shift:.4f}s"
    if log_callback:
        log_callback(msg)
    else:
        print(msg)
    return best_shift

    return best_shift


class FFmpegWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            for line in self.process.stdout:
                self.log_signal.emit(line.strip())
            
            self.process.wait()
            if self.process.returncode == 0:
                self.finished_signal.emit(True, "Muxing complete.")
            else:
                self.finished_signal.emit(False, f"ffmpeg exited with code {self.process.returncode}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def cancel(self):
        if self.process:
            try:
                self.process.terminate()
            except:
                pass


class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object, float, float, int)
    error_signal = pyqtSignal(str)

    def __init__(self, file1, file2, scale_factor, mode, frame_step=1, max_frames=float('inf')):
        super().__init__()
        self.file1 = file1
        self.file2 = file2
        self.scale_factor = scale_factor
        self.mode = mode
        self.frame_step = max(1, frame_step)
        self.max_frames = max_frames
        self._is_cancelled = False

    def analyze_video(self, video_path, label):
        self.log_signal.emit(f"Analyzing {label}: {os.path.basename(video_path)}")
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            timestamps = []
            reds = []
            greens = []
            blues = []
            hues = []
            thumbnails = []

            total_frames = stream.frames
            if not total_frames or total_frames == 0:
                try:
                    dur = 0
                    if stream.duration and stream.time_base:
                        dur = float(stream.duration * stream.time_base)
                    elif container.duration:
                        dur = container.duration / 1000000.0
                    
                    rate = stream.average_rate or stream.r_frame_rate
                    if dur > 0 and rate:
                        total_frames = int(dur * float(rate))
                except Exception:
                    total_frames = 0

            if not total_frames or total_frames == 0:
                total_to_process = 0
            else:
                total_to_process = min(total_frames, self.max_frames)
            
            frame_number = 0
            processed_count = 0
            for frame in container.decode(stream):
                if self._is_cancelled:
                    break
                if frame_number >= self.max_frames:
                    break
                frame_number += 1
                
                if (frame_number - 1) % self.frame_step != 0:
                    continue
                
                small_frame = frame.reformat(width=200, height=112, format="rgb24")
                image = small_frame.to_ndarray()
                r, g, b, h = get_color_percentages(image)
                
                timestamps.append(frame.time)
                reds.append(r)
                greens.append(g)
                blues.append(b)
                hues.append(h)
                thumbnails.append((frame.time, bytes(image), 200, 112))
                processed_count += 1
                
                if processed_count % 5 == 0 or frame_number == 1:
                    self.progress_signal.emit(frame_number, total_to_process, f"Analyzing {label}...")
                    
            container.close()
            return timestamps, reds, greens, blues, hues, thumbnails
        except Exception as e:
            self.error_signal.emit(f"Error analyzing {video_path}: {str(e)}")
            return None

    def run(self):
        data1 = self.analyze_video(self.file1, "Audio Source")
        if self._is_cancelled:
            self.error_signal.emit("Analysis stopped by user.")
            return
        if not data1: return
        
        data2 = self.analyze_video(self.file2, "Video Source")
        if self._is_cancelled:
            self.error_signal.emit("Analysis stopped by user.")
            return
        if not data2: return
        
        self.log_signal.emit("Calculating time shift...")
        self.progress_signal.emit(0, 0, "Calculating shift...")
        
        t1, r1, g1, b1, h1 = data1[:5]
        t2, r2, g2, b2, h2 = data2[:5]
        
        grad1 = np.array(h1)
        grad2 = np.array(h2)
        
        if self.mode == 0:
            shift = find_time_shift(t1, grad1, t2, grad2, self.scale_factor, log_callback=lambda m: self.log_signal.emit(m))
        else:
            shift = find_time_shift(t2, grad2, t1, grad1, self.scale_factor, log_callback=lambda m: self.log_signal.emit(m))
        
        if not self._is_cancelled:
            self.finished_signal.emit(data1, data2, self.scale_factor, shift, self.mode)
        else:
            self.error_signal.emit("Analysis stopped by user.")

    def cancel(self):
        self._is_cancelled = True


class VideoAudioSyncApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Audio Synchronizer")
        self.resize(1000, 800)
        
        self.audio_folder_path = None
        self.video_folder_path = None
        self.output_folder_path = None
        
        self.audio_files = []
        self.video_files = []
        
        self.current_index = 0
        self.audio_fps = None
        self.video_fps = None
        self.worker = None
        self.ffmpeg_worker = None
        self.last_shift = None
        self.last_scale = None
        self.last_mode = None
        self.last_t_common = None
        self.last_diff = None
        self.baseline_diff = 0.0
        self.std_diff = 0.0
        self.diff_collection = None
        self.removal_collection = None
        self.audio_thumbnails = []
        self.video_thumbnails = []
        
        self.app_mode = "folder"
        self.settings_file = "settings.json"
        
        self.ratio_timer = QTimer()
        self.ratio_timer.setSingleShot(True)
        self.ratio_timer.timeout.connect(self.apply_ratio_change)
        
        self.init_ui()
        self.preview_popup = FramePreviewPopup(self)
        self.load_settings()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Folders Selection
        files_group = QGroupBox("Source Selection")
        files_layout = QVBoxLayout()
        
        # Mode Toggle
        mode_layout = QHBoxLayout()
        self.btn_group_mode = QButtonGroup(self)
        
        self.radio_folder = QRadioButton("Folder Mode")
        self.radio_folder.setChecked(True)
        self.radio_file = QRadioButton("Single File Mode")
        
        self.btn_group_mode.addButton(self.radio_folder, 0)
        self.btn_group_mode.addButton(self.radio_file, 1)
        
        mode_layout.addWidget(self.radio_folder)
        mode_layout.addWidget(self.radio_file)
        mode_layout.addStretch()
        
        self.radio_folder.toggled.connect(self.on_mode_toggled)
        files_layout.addLayout(mode_layout)
        
        # Audio Folder and Output Folder
        h1 = QHBoxLayout()
        self.btn_audio = QPushButton("Select Audio Folder")
        self.btn_audio.clicked.connect(lambda: self.select_folder(1))
        self.lbl_audio = QLabel("No folder selected")
        h1.addWidget(self.btn_audio)
        h1.addWidget(self.lbl_audio, stretch=1)
        
        self.btn_output = QPushButton("Select Output Folder")
        self.btn_output.clicked.connect(lambda: self.select_folder(3))
        self.lbl_output = QLabel("No folder selected")
        h1.addWidget(self.btn_output)
        h1.addWidget(self.lbl_output, stretch=1)
        
        self.info_audio = QLabel("")
        self.info_audio.setStyleSheet("color: gray;")
        
        # Video Folder
        h2 = QHBoxLayout()
        self.btn_video = QPushButton("Select Video Folder")
        self.btn_video.clicked.connect(lambda: self.select_folder(2))
        self.lbl_video = QLabel("No folder selected")
        h2.addWidget(self.btn_video)
        h2.addWidget(self.lbl_video, stretch=1)
        
        self.info_video = QLabel("")
        self.info_video.setStyleSheet("color: gray;")
        
        files_layout.addLayout(h1)
        files_layout.addWidget(self.info_audio)
        files_layout.addLayout(h2)
        files_layout.addWidget(self.info_video)
        files_group.setLayout(files_layout)
        main_layout.addWidget(files_group)
        
        # Mapping Settings
        self.mapping_group = QGroupBox("Mapping Settings")
        mapping_layout = QVBoxLayout()
        
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Ratio (Audio:Video):"))
        self.spin_ratio_a = QSpinBox()
        self.spin_ratio_a.setRange(1, 100)
        self.spin_ratio_a.setValue(1)
        self.spin_ratio_a.valueChanged.connect(self.on_ratio_changed)
        
        self.spin_ratio_v = QSpinBox()
        self.spin_ratio_v.setRange(1, 100)
        self.spin_ratio_v.setValue(1)
        self.spin_ratio_v.valueChanged.connect(self.on_ratio_changed)
        
        ratio_layout.addWidget(self.spin_ratio_a)
        ratio_layout.addWidget(QLabel(":"))
        ratio_layout.addWidget(self.spin_ratio_v)
        ratio_layout.addStretch()
        
        mapping_layout.addLayout(ratio_layout)
        
        lbl_mapping_desc = QLabel("Ratio of Audio files to Video files. For example, 1:2 means 1 Audio file corresponds to 2 Video files.")
        lbl_mapping_desc.setStyleSheet("font-style: italic; color: gray;")
        lbl_mapping_desc.setWordWrap(True)
        mapping_layout.addWidget(lbl_mapping_desc)
        
        self.mapping_group.setLayout(mapping_layout)
        main_layout.addWidget(self.mapping_group)
        
        # Current Pair Details
        pair_group = QGroupBox("Current Pair Details")
        pair_layout = QVBoxLayout()
        
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Previous")
        self.btn_prev.clicked.connect(self.prev_pair)
        self.btn_prev.setEnabled(False)
        
        self.lbl_nav = QLabel("Pair 0 of 0")
        self.lbl_nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_pair)
        self.btn_next.setEnabled(False)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_nav, stretch=1)
        nav_layout.addWidget(self.btn_next)
        
        pair_layout.addLayout(nav_layout)
        
        # Audio File Details
        self.lbl_pair_audio = QLabel("No audio file")
        self.info_pair_audio = QLabel("")
        self.info_pair_audio.setStyleSheet("color: gray;")
        
        # Video File Details
        self.lbl_pair_video = QLabel("No video file")
        self.info_pair_video = QLabel("")
        self.info_pair_video.setStyleSheet("color: gray;")
        
        pair_layout.addWidget(self.lbl_pair_audio)
        pair_layout.addWidget(self.info_pair_audio)
        pair_layout.addWidget(self.lbl_pair_video)
        pair_layout.addWidget(self.info_pair_video)
        
        pair_group.setLayout(pair_layout)
        main_layout.addWidget(pair_group)
        
        # Settings
        settings_group = QGroupBox("Analysis & Scaling Settings")
        settings_layout = QFormLayout()
        
        self.spin_frame_step = QSpinBox()
        self.spin_frame_step.setRange(1, 100)
        self.spin_frame_step.setValue(1)
        self.spin_frame_step.valueChanged.connect(lambda: self.save_settings())
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Scale Video to match Audio", "Scale Audio to match Video"])
        self.combo_scale.setEnabled(False)
        self.combo_scale.currentIndexChanged.connect(lambda: self.save_settings())
        
        settings_layout.addRow("Frame Sampling (1 in N frames):", self.spin_frame_step)
        settings_layout.addRow("Scaling Mode:", self.combo_scale)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Controls and Progress
        controls_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.btn_sync_save = QPushButton("Sync and Save")
        self.btn_sync_save.clicked.connect(self.start_sync_save)
        self.btn_sync_save.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton { background-color: #d32f2f; color: white; }
            QPushButton:disabled { background-color: #555555; color: #aaaaaa; }
        """)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.lbl_status = QLabel("Ready")
        
        controls_layout.addWidget(self.btn_analyze)
        controls_layout.addWidget(self.btn_sync_save)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.progress_bar, stretch=1)
        controls_layout.addWidget(self.lbl_status)
        main_layout.addLayout(controls_layout)
        
        # Diff Threshold
        diff_layout = QHBoxLayout()
        diff_layout.addWidget(QLabel("Color Tint Difference Threshold:"))
        self.slider_diff = QSlider(Qt.Orientation.Horizontal)
        self.slider_diff.setMinimum(0)
        self.slider_diff.setMaximum(100)
        self.slider_diff.setValue(20)
        self.slider_diff.setEnabled(False)
        self.slider_diff.valueChanged.connect(self.update_diff_plot)
        self.lbl_diff_val = QLabel("1.0σ")
        diff_layout.addWidget(self.slider_diff, stretch=1)
        diff_layout.addWidget(self.lbl_diff_val)
        main_layout.addLayout(diff_layout)
        
        # Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)
        
        # Plot Canvas
        self.figure = Figure(figsize=(8, 6))
        self.figure.patch.set_facecolor('#202124')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_hover)
        main_layout.addWidget(self.canvas, stretch=1)
        
    def log(self, msg):
        self.log_text.append(msg)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.log("Stopping analysis...")
            self.btn_stop.setEnabled(False)
            
        if self.ffmpeg_worker and self.ffmpeg_worker.isRunning():
            self.ffmpeg_worker.cancel()
            self.log("Stopping ffmpeg...")
            self.btn_stop.setEnabled(False)

    def on_mode_toggled(self):
        new_mode = "folder" if self.radio_folder.isChecked() else "file"
        if new_mode == self.app_mode:
            return
            
        self.app_mode = new_mode
        
        self.audio_folder_path = None
        self.video_folder_path = None
        self.audio_files = []
        self.video_files = []
        self.current_index = 0
        self.audio_fps = None
        self.video_fps = None
        self.last_shift = None
        self.last_scale = None
        self.last_mode = None
        self.btn_sync_save.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.figure.clear()
        self.canvas.draw()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Ready")
        
        if self.app_mode == "folder":
            self.btn_audio.setText("Select Audio Folder")
            self.btn_video.setText("Select Video Folder")
            self.mapping_group.setEnabled(True)
            self.btn_prev.setVisible(True)
            self.btn_next.setVisible(True)
            self.lbl_nav.setVisible(True)
        else:
            self.btn_audio.setText("Select Audio File")
            self.btn_video.setText("Select Video File")
            self.mapping_group.setEnabled(False)
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.lbl_nav.setVisible(False)
            
        self.set_folder_info(True)
        self.set_folder_info(False)
        self.load_current_pair()
        self.save_settings()

    def on_ratio_changed(self):
        self.ratio_timer.start(2000)
        
    def apply_ratio_change(self):
        self.save_settings()
        self.current_index = 0
        self.load_current_pair()

    def set_folder_info(self, is_audio):
        files = self.audio_files if is_audio else self.video_files
        count = len(files)
        res = "Unknown"
        if count > 0:
            res = self.extract_resolution(files[0])
            
        file_word = "file" if self.app_mode == "file" else "files"
        info_text = f"{count} valid {file_word} found | Resolution: {res} (from first file)"
        
        no_text = "No file selected" if self.app_mode == "file" else "No folder selected"
        
        if is_audio:
            self.lbl_audio.setText(self.audio_folder_path if self.audio_folder_path else no_text)
            self.info_audio.setText(info_text)
        else:
            self.lbl_video.setText(self.video_folder_path if self.video_folder_path else no_text)
            self.info_video.setText(info_text)
        
    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    
                self.app_mode = settings.get("app_mode", "folder")
                
                self.radio_folder.blockSignals(True)
                self.radio_file.blockSignals(True)
                if self.app_mode == "file":
                    self.radio_file.setChecked(True)
                else:
                    self.radio_folder.setChecked(True)
                self.radio_folder.blockSignals(False)
                self.radio_file.blockSignals(False)
                    
                if self.app_mode == "folder":
                    audio_folder = settings.get("audio_folder")
                    if audio_folder and os.path.isdir(audio_folder):
                        self.audio_folder_path = audio_folder
                        self.audio_files = self.get_media_files(audio_folder)
                        self.set_folder_info(True)
                        
                    video_folder = settings.get("video_folder")
                    if video_folder and os.path.isdir(video_folder):
                        self.video_folder_path = video_folder
                        self.video_files = self.get_media_files(video_folder)
                        self.set_folder_info(False)
                else:
                    audio_file = settings.get("audio_file")
                    if audio_file and os.path.isfile(audio_file):
                        self.audio_folder_path = audio_file
                        self.audio_files = [audio_file]
                        self.set_folder_info(True)
                        
                    video_file = settings.get("video_file")
                    if video_file and os.path.isfile(video_file):
                        self.video_folder_path = video_file
                        self.video_files = [video_file]
                        self.set_folder_info(False)
                        
                output_folder = settings.get("output_folder")
                if output_folder and os.path.isdir(output_folder):
                    self.output_folder_path = output_folder
                    self.lbl_output.setText(output_folder)
                    
                scale_mode = settings.get("scale_mode", 0)
                self.combo_scale.setCurrentIndex(scale_mode)
                
                frame_step = settings.get("frame_step", 1)
                self.spin_frame_step.blockSignals(True)
                self.spin_frame_step.setValue(frame_step)
                self.spin_frame_step.blockSignals(False)
                
                ratio_a = settings.get("ratio_a", 1)
                ratio_v = settings.get("ratio_v", 1)
                self.spin_ratio_a.blockSignals(True)
                self.spin_ratio_v.blockSignals(True)
                self.spin_ratio_a.setValue(ratio_a)
                self.spin_ratio_v.setValue(ratio_v)
                self.spin_ratio_a.blockSignals(False)
                self.spin_ratio_v.blockSignals(False)
                
                self.current_index = 0
                self.load_current_pair()
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def save_settings(self):
        settings = {
            "app_mode": self.app_mode,
            "output_folder": self.output_folder_path,
            "scale_mode": self.combo_scale.currentIndex(),
            "frame_step": self.spin_frame_step.value(),
            "ratio_a": self.spin_ratio_a.value(),
            "ratio_v": self.spin_ratio_v.value()
        }
        if self.app_mode == "folder":
            settings["audio_folder"] = self.audio_folder_path
            settings["video_folder"] = self.video_folder_path
        else:
            settings["audio_file"] = self.audio_folder_path
            settings["video_file"] = self.video_folder_path
            
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def extract_resolution(self, file_path):
        try:
            container = av.open(file_path)
            stream = container.streams.video[0]
            w = stream.width
            h = stream.height
            container.close()
            return f"{w}x{h}" if w and h else "Unknown"
        except Exception:
            return "Unknown"

    def extract_info(self, file_path):
        try:
            container = av.open(file_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            duration = stream.duration * float(stream.time_base) if stream.duration else 0
            frames = stream.frames
            container.close()
            return f"FPS: {fps:.2f} | Duration: {duration:.2f}s | Frames: {frames}", fps
        except Exception as e:
            return f"Error reading file info: {e}", None
            
    def get_media_files(self, folder_path):
        valid_exts = ('.mp4', '.avi', '.mkv', '.mov')
        files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(valid_exts):
                files.append(os.path.join(folder_path, f))
        return natsorted(files)

    def select_folder(self, folder_num):
        if folder_num == 1:
            title = "Select Audio File" if self.app_mode == "file" else "Select Audio Folder"
        elif folder_num == 2:
            title = "Select Video File" if self.app_mode == "file" else "Select Video Folder"
        else:
            title = "Select Output Folder"
            
        if self.app_mode == "file" and folder_num in (1, 2):
            folder_path, _ = QFileDialog.getOpenFileName(self, title, "", "Video/Audio Files (*.mp4 *.avi *.mkv *.mov)")
        else:
            folder_path = QFileDialog.getExistingDirectory(self, title, "")
            
        if folder_path:
            if folder_num == 1:
                self.audio_folder_path = folder_path
                self.audio_files = [folder_path] if self.app_mode == "file" else self.get_media_files(folder_path)
                self.set_folder_info(True)
            elif folder_num == 2:
                self.video_folder_path = folder_path
                self.video_files = [folder_path] if self.app_mode == "file" else self.get_media_files(folder_path)
                self.set_folder_info(False)
            elif folder_num == 3:
                self.output_folder_path = folder_path
                self.lbl_output.setText(folder_path)
                
            if folder_num in (1, 2):
                self.current_index = 0
                self.load_current_pair()
            self.save_settings()

    def get_current_indices(self):
        ratio_a = self.spin_ratio_a.value()
        ratio_v = self.spin_ratio_v.value()
        if ratio_a > 1 and ratio_v == 1:
            return self.current_index, self.current_index // ratio_a
        elif ratio_v > 1 and ratio_a == 1:
            return self.current_index // ratio_v, self.current_index
        else:
            return self.current_index, self.current_index

    def load_current_pair(self):
        self.figure.clear()
        self.canvas.draw()
        
        ratio_a = self.spin_ratio_a.value()
        ratio_v = self.spin_ratio_v.value()
        
        num_audio = len(self.audio_files)
        num_video = len(self.video_files)
        
        if ratio_a > 1 and ratio_v == 1:
            total_pairs = min(num_audio, num_video * ratio_a)
        elif ratio_v > 1 and ratio_a == 1:
            total_pairs = min(num_audio * ratio_v, num_video)
        else:
            total_pairs = min(num_audio, num_video)
        
        if total_pairs == 0:
            self.lbl_nav.setText("Pair 0 of 0")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.lbl_pair_audio.setText("No matching files")
            self.info_pair_audio.setText("")
            self.lbl_pair_video.setText("No matching files")
            self.info_pair_video.setText("")
            return
            
        self.lbl_nav.setText(f"Pair {self.current_index + 1} of {total_pairs}")
        self.btn_prev.setEnabled(self.current_index > 0)
        self.btn_next.setEnabled(self.current_index < total_pairs - 1)
        
        audio_idx, video_idx = self.get_current_indices()
        audio_file = self.audio_files[audio_idx]
        video_file = self.video_files[video_idx]
        
        self.lbl_pair_audio.setText(f"<b>Audio Source:</b> {os.path.basename(audio_file)}")
        info_audio, fps_audio = self.extract_info(audio_file)
        self.info_pair_audio.setText(info_audio)
        self.audio_fps = fps_audio
        
        self.lbl_pair_video.setText(f"<b>Video Source:</b> {os.path.basename(video_file)}")
        info_video, fps_video = self.extract_info(video_file)
        self.info_pair_video.setText(info_video)
        self.video_fps = fps_video
        
        self.btn_analyze.setEnabled(True)
        if self.audio_fps and self.video_fps:
            if abs(self.audio_fps - self.video_fps) < 0.01:
                self.combo_scale.setEnabled(False)
            else:
                self.combo_scale.setEnabled(True)

    def next_pair(self):
        self.current_index += 1
        self.load_current_pair()
        
    def prev_pair(self):
        self.current_index -= 1
        self.load_current_pair()

    def start_analysis(self):
        if not self.audio_files or not self.video_files:
            return
            
        audio_idx, video_idx = self.get_current_indices()
        audio_file = self.audio_files[audio_idx]
        video_file = self.video_files[video_idx]
            
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.clear()
        self.figure.clear()
        self.canvas.draw()
        
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.lbl_status.setText("Starting analysis...")
        
        if self.audio_fps and self.video_fps:
            if abs(self.audio_fps - self.video_fps) < 0.01:
                scale_factor = 1.0
                mode = 0
            else:
                mode = self.combo_scale.currentIndex()
                if mode == 0:
                    scale_factor = self.video_fps / self.audio_fps
                else:
                    scale_factor = self.audio_fps / self.video_fps
        else:
            scale_factor = 1.0
            mode = 0
            
        self.log(f"Using scale factor: {scale_factor:.5f}")
        
        frame_step = self.spin_frame_step.value()
        self.log(f"Frame sampling rate: 1 in {frame_step} frames")
        
        self.worker = AnalysisWorker(audio_file, video_file, scale_factor, mode, frame_step=frame_step)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.error_signal.connect(self.on_analysis_error)
        
        self.worker.start()
        
    def update_progress(self, current, total, status):
        self.lbl_status.setText(status)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setMaximum(0) # indeterminate
            
    def on_analysis_error(self, msg):
        self.log(f"ERROR: {msg}")
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Error")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
    def on_analysis_finished(self, data1, data2, scale_factor, shift, mode):
        self.log("Analysis complete. Plotting results...")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Finished")
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        self.last_shift = shift
        self.last_scale = scale_factor
        self.last_mode = mode
        self.audio_thumbnails = data1[5]
        self.video_thumbnails = data2[5]
        self.btn_sync_save.setEnabled(True)
        
        self.plot_results(data1, data2, scale_factor, shift, mode)
        
    def plot_results(self, data1, data2, scale_factor, shift, mode):
        self.figure.clear()
        
        t1, r1, g1, b1, h1 = data1[:5]
        t2, r2, g2, b2, h2 = data2[:5]
        
        t1 = np.array(t1)
        t2 = np.array(t2)
        grad1 = np.array(h1)
        grad2 = np.array(h2)
        
        ax3 = self.figure.add_subplot(111)
        
        ax3.set_facecolor('#303134')
        ax3.tick_params(colors='white')
        ax3.xaxis.label.set_color('white')
        ax3.yaxis.label.set_color('white')
        ax3.title.set_color('white')
        for spine in ax3.spines.values():
            spine.set_edgecolor('gray')
        
        # Plot Aligned Color Balance
        if mode == 0:
            ax3.plot(t1, grad1, color='white', label='Audio Source Color Tint', linewidth=2)
            t_aligned = np.array(t2) * scale_factor + shift
            ax3.plot(t_aligned, grad2, color='cyan', label=f'Video Source (x{scale_factor:.3f} + {shift:.2f}s)', linewidth=2, linestyle='--')
            ax3.set_xlabel('Time (Audio Source Timebase)')
            t_ref = t1
        else:
            ax3.plot(t2, grad2, color='white', label='Video Source Color Tint', linewidth=2)
            t_aligned = np.array(t1) * scale_factor + shift
            ax3.plot(t_aligned, grad1, color='cyan', label=f'Audio Source (x{scale_factor:.3f} + {shift:.2f}s)', linewidth=2, linestyle='--')
            ax3.set_xlabel('Time (Video Source Timebase)')
            t_ref = t2
            
        overall_start = min(t_ref[0], t_aligned[0])
        overall_end = max(t_ref[-1], t_aligned[-1])
        overlap_start = max(t_ref[0], t_aligned[0])
        overlap_end = min(t_ref[-1], t_aligned[-1])
        
        has_cut_label = False
        if overall_start < overlap_start:
            ax3.axvspan(overall_start, overlap_start, color='red', alpha=0.3, label='Cut Content')
            has_cut_label = True
        if overlap_end < overall_end:
            ax3.axvspan(overlap_end, overall_end, color='red', alpha=0.3, label='' if has_cut_label else 'Cut Content')
            
        ax3.set_ylabel('Color Tint (Cool ↔ Warm)')
        ax3.set_title('Aligned Color Balance Curves')
        ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax3.grid(True, alpha=0.3)
        
        # Calculate overlap region for diff
        if overlap_start < overlap_end:
            valid_indices = np.where((t_ref >= overlap_start) & (t_ref <= overlap_end))[0]
            self.last_t_common = t_ref[valid_indices]
            
            if mode == 0:
                ref_vals = grad1[valid_indices]
                interp_align = interpolate.interp1d(t_aligned, grad2, bounds_error=False, fill_value=0)
                align_vals = interp_align(self.last_t_common)
            else:
                ref_vals = grad2[valid_indices]
                interp_align = interpolate.interp1d(t_aligned, grad1, bounds_error=False, fill_value=0)
                align_vals = interp_align(self.last_t_common)
                
            self.last_diff = np.abs(ref_vals - align_vals)
            
            p5 = np.percentile(self.last_diff, 5)
            best_5_pct = self.last_diff[self.last_diff <= p5]
            if len(best_5_pct) > 0:
                self.baseline_diff = float(np.mean(best_5_pct))
            else:
                self.baseline_diff = float(np.min(self.last_diff))
                
            self.std_diff = float(np.std(self.last_diff))
            
            self.slider_diff.setEnabled(True)
        else:
            self.slider_diff.setEnabled(False)
            self.last_t_common = None
            self.last_diff = None
            self.baseline_diff = 0.0
            self.std_diff = 0.0
            
        global_min = min(t1[0], t2[0], t_aligned[0])
        global_max = max(t1[-1], t2[-1], t_aligned[-1])
        x_limit = global_max
        
        ax3.set_xlim(global_min, x_limit)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
        if self.slider_diff.isEnabled():
            self.update_diff_plot()

    def get_diff_mask(self, threshold, min_consecutive=2):
        if self.last_diff is None:
            return None
        is_above = self.last_diff > threshold
        if not np.any(is_above):
            return is_above
            
        mask = np.zeros_like(is_above, dtype=bool)
        padded = np.pad(is_above.astype(int), (1, 1), 'constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        for s, e in zip(starts, ends):
            if (e - s) >= min_consecutive:
                mask[s:e] = True
                
        return mask

    def get_removal_mask(self, threshold, min_density=0.50):
        if self.last_diff is None or self.last_t_common is None:
            return None

        blue_mask = self.get_diff_mask(threshold, min_consecutive=2)
        if blue_mask is None or not np.any(blue_mask):
            return None

        removal_mask = np.zeros_like(blue_mask, dtype=bool)
        n = len(self.last_t_common)

        for i in range(n):
            if blue_mask[i]:
                remaining = n - i
                if remaining > 0:
                    density = np.sum(blue_mask[i:]) / remaining
                    if density >= min_density:
                        removal_mask[i:] = True
                        break

        return removal_mask

    def update_diff_plot(self):
        sigma_multiplier = self.slider_diff.value() / 20.0
        threshold = self.baseline_diff + (sigma_multiplier * self.std_diff)
        self.lbl_diff_val.setText(f"{sigma_multiplier:.1f}σ (Thresh: {threshold:.2f})")
        
        if self.last_diff is not None and self.last_t_common is not None:
            if len(self.figure.axes) >= 1:
                ax3 = self.figure.axes[0]
                
                if self.diff_collection is not None:
                    try:
                        self.diff_collection.remove()
                    except Exception:
                        pass
                    self.diff_collection = None

                if hasattr(self, 'removal_collection') and self.removal_collection is not None:
                    try:
                        self.removal_collection.remove()
                    except Exception:
                        pass
                    self.removal_collection = None
                        
                mask = self.get_diff_mask(threshold)
                if mask is not None and np.any(mask):
                    self.diff_collection = ax3.fill_between(
                        self.last_t_common, 0, 1,
                        where=mask,
                        color='blue', alpha=1.0, zorder=10, transform=ax3.get_xaxis_transform(),
                        label='Diff > Threshold'
                    )

                self.last_removal_mask = self.get_removal_mask(threshold, min_density=0.50)
                if self.last_removal_mask is not None and np.any(self.last_removal_mask):
                    self.removal_collection = ax3.fill_between(
                        self.last_t_common, 0, 0.04,
                        where=self.last_removal_mask,
                        color='red', alpha=1.0, zorder=15, transform=ax3.get_xaxis_transform(),
                        label='To Be Removed'
                    )

                self.canvas.draw_idle()

    def on_canvas_hover(self, event):
        if event.inaxes is None or self.last_diff is None or self.last_t_common is None:
            self.preview_popup.hide()
            return
            
        if len(self.figure.axes) >= 1 and event.inaxes == self.figure.axes[0]:
            t_hover = event.xdata
            if t_hover is None:
                self.preview_popup.hide()
                return
                
            idx = np.searchsorted(self.last_t_common, t_hover)
            idx = np.clip(idx, 0, len(self.last_diff) - 1)
            
            diff_val = self.last_diff[idx]
            sigma_multiplier = self.slider_diff.value() / 20.0
            threshold = self.baseline_diff + (sigma_multiplier * self.std_diff)
            
            if True:
                time_str = format_time(t_hover)
                a_fps = self.audio_fps or 25.0
                v_fps = self.video_fps or 25.0
                
                shift = self.last_shift or 0.0
                scale = self.last_scale or 1.0
                mode = self.last_mode or 0
                
                if mode == 0:
                    t_audio = t_hover
                    t_video = (t_hover - shift) / scale
                else:
                    t_video = t_hover
                    t_audio = (t_hover - shift) / scale
                    
                a_frame = int(t_audio * a_fps)
                v_frame = int(t_video * v_fps)
                
                pix_audio = get_thumbnail(self.audio_thumbnails, t_audio)
                pix_video = get_thumbnail(self.video_thumbnails, t_video)
                
                diff_sigma = (diff_val - self.baseline_diff) / (self.std_diff + 1e-6)
                
                is_removal = hasattr(self, 'last_removal_mask') and self.last_removal_mask is not None and self.last_removal_mask[idx]
                status_icon = "   |   🛑 TO BE REMOVED" if is_removal else ""

                header_text = f"⏱ Time: {time_str} ({t_hover:.2f}s)   |   Diff: {diff_val:.3f} ({diff_sigma:.1f}σ){status_icon}"
                footer_text = f"🎵 Audio Frame: #{a_frame} ({t_audio:.2f}s)     •     🎬 Video Frame: #{v_frame} ({t_video:.2f}s)"
                
                self.preview_popup.update_preview(header_text, pix_audio, pix_video, footer_text)
                
                cursor_pos = QCursor.pos()
                self.preview_popup.move(cursor_pos + QPoint(15, 15))
                self.preview_popup.show()
                self.preview_popup.raise_()
            else:
                self.preview_popup.hide()
        else:
            self.preview_popup.hide()

    def start_sync_save(self):
        if not self.output_folder_path:
            self.log("ERROR: Please select an Output Folder first.")
            return
            
        audio_idx, video_idx = self.get_current_indices()
        audio_file = self.audio_files[audio_idx]
        video_file = self.video_files[video_idx]
        
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(self.output_folder_path, f"{base_name}_synced.mkv")
        
        shift = self.last_shift
        scale = self.last_scale
        mode = self.last_mode
        
        if mode == 0:
            if shift > 0:
                trim_audio = shift
                trim_video = 0.0
            else:
                trim_audio = 0.0
                trim_video = abs(shift) / scale
        else:
            if shift > 0:
                trim_video = shift
                trim_audio = 0.0
            else:
                trim_video = 0.0
                trim_audio = abs(shift) / scale
                
        cmd = ["ffmpeg", "-y"]
        
        slider_val = self.slider_diff.value()
        sigma_multiplier = slider_val / 20.0
        threshold = self.baseline_diff + (sigma_multiplier * self.std_diff)
        removal_mask = self.get_removal_mask(threshold, min_density=0.50)
        
        duration_to_encode = None
        if removal_mask is not None and np.any(removal_mask):
            cut_idx = np.argmax(removal_mask)
            t_cut = self.last_t_common[cut_idx]
            
            if mode == 0:
                duration_to_encode = max(0.0, t_cut - trim_audio)
            else:
                duration_to_encode = max(0.0, t_cut - trim_video)
        
        if trim_audio > 0:
            cmd.extend(["-ss", f"{trim_audio:.4f}"])
        cmd.extend(["-i", audio_file])
        
        if trim_video > 0:
            cmd.extend(["-ss", f"{trim_video:.4f}"])
            
        if abs(scale - 1.0) > 0.001 and mode == 0:
            cmd.extend(["-itsscale", f"{scale:.5f}"])
            
        cmd.extend(["-i", video_file])
        
        cmd.extend(["-map", "0:a:0", "-map", "1:v:0"])
        
        if abs(scale - 1.0) < 0.001:
            cmd.extend(["-c", "copy"])
        else:
            if mode == 0:
                cmd.extend(["-c", "copy"])
            else:
                cmd.extend(["-c:v", "copy", "-c:a", "aac", "-filter:a", f"atempo={1/scale:.5f}"])
                
        if duration_to_encode is not None:
            cmd.extend(["-t", f"{duration_to_encode:.4f}"])
            
        cmd.extend(["-shortest", output_file])
        
        self.btn_analyze.setEnabled(False)
        self.btn_sync_save.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log(f"Starting ffmpeg...")
        
        self.ffmpeg_worker = FFmpegWorker(cmd)
        self.ffmpeg_worker.log_signal.connect(self.log)
        self.ffmpeg_worker.finished_signal.connect(self.on_sync_finished)
        self.ffmpeg_worker.start()

    def on_sync_finished(self, success, msg):
        self.log(msg)
        self.btn_analyze.setEnabled(True)
        self.btn_sync_save.setEnabled(True)
        self.btn_stop.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAudioSyncApp()
    window.show()
    sys.exit(app.exec())
