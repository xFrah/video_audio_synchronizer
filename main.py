import sys
import os
import glob
import json
import av
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from natsort import natsorted

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, 
    QGroupBox, QFormLayout, QTextEdit, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def get_color_percentages(image):
    total_pixels = image.shape[0] * image.shape[1]
    max_intensity = total_pixels * 255.0
    b_sum = np.sum(image[:, :, 0])
    g_sum = np.sum(image[:, :, 1])
    r_sum = np.sum(image[:, :, 2])
    return r_sum / max_intensity, g_sum / max_intensity, b_sum / max_intensity


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


class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object, float, float, int)
    error_signal = pyqtSignal(str)

    def __init__(self, file1, file2, scale_factor, mode, max_frames=1000):
        super().__init__()
        self.file1 = file1
        self.file2 = file2
        self.scale_factor = scale_factor
        self.mode = mode
        self.max_frames = max_frames
        self._is_cancelled = False

    def analyze_video(self, video_path, label):
        self.log_signal.emit(f"Analyzing {label}: {os.path.basename(video_path)}")
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            timestamps = []
            reds = []
            greens = []
            blues = []

            total_frames = stream.frames
            if total_frames == 0 or total_frames is None:
                total_frames = self.max_frames
                
            total_to_process = min(total_frames, self.max_frames)
            
            frame_number = 0
            for frame in container.decode(stream):
                if self._is_cancelled:
                    break
                if frame_number >= self.max_frames:
                    break
                frame_number += 1
                
                image = frame.to_ndarray(format="bgr24")
                r, g, b = get_color_percentages(image)
                
                timestamps.append(frame.time)
                reds.append(r)
                greens.append(g)
                blues.append(b)
                
                if frame_number % 10 == 0:
                    self.progress_signal.emit(frame_number, total_to_process, f"Analyzing {label}...")
                    
            container.close()
            return timestamps, reds, greens, blues
        except Exception as e:
            self.error_signal.emit(f"Error analyzing {video_path}: {str(e)}")
            return None

    def run(self):
        data1 = self.analyze_video(self.file1, "Audio Source")
        if not data1 or self._is_cancelled: return
        
        data2 = self.analyze_video(self.file2, "Video Source")
        if not data2 or self._is_cancelled: return
        
        self.log_signal.emit("Calculating time shift...")
        self.progress_signal.emit(0, 0, "Calculating shift...")
        
        t1, r1, g1, b1 = data1
        t2, r2, g2, b2 = data2
        
        dr1 = np.gradient(r1)
        dg1 = np.gradient(g1)
        db1 = np.gradient(b1)
        grad1 = np.sqrt(dr1**2 + dg1**2 + db1**2)
        
        dr2 = np.gradient(r2)
        dg2 = np.gradient(g2)
        db2 = np.gradient(b2)
        grad2 = np.sqrt(dr2**2 + dg2**2 + db2**2)
        
        if self.mode == 0:
            shift = find_time_shift(t1, grad1, t2, grad2, self.scale_factor, log_callback=lambda m: self.log_signal.emit(m))
        else:
            shift = find_time_shift(t2, grad2, t1, grad1, self.scale_factor, log_callback=lambda m: self.log_signal.emit(m))
        
        if not self._is_cancelled:
            self.finished_signal.emit(data1, data2, self.scale_factor, shift, self.mode)

    def cancel(self):
        self._is_cancelled = True


class VideoAudioSyncApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Audio Synchronizer")
        self.resize(1000, 800)
        
        self.audio_folder_path = None
        self.video_folder_path = None
        
        self.audio_files = []
        self.video_files = []
        
        self.current_index = 0
        self.audio_fps = None
        self.video_fps = None
        self.worker = None
        self.settings_file = "settings.json"
        
        self.ratio_timer = QTimer()
        self.ratio_timer.setSingleShot(True)
        self.ratio_timer.timeout.connect(self.apply_ratio_change)
        
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Folders Selection
        files_group = QGroupBox("Source Selection")
        files_layout = QVBoxLayout()
        
        # Audio Folder
        h1 = QHBoxLayout()
        self.btn_audio = QPushButton("Select Audio Folder")
        self.btn_audio.clicked.connect(lambda: self.select_folder(1))
        self.lbl_audio = QLabel("No folder selected")
        h1.addWidget(self.btn_audio)
        h1.addWidget(self.lbl_audio, stretch=1)
        
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
        mapping_group = QGroupBox("Mapping Settings")
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
        
        mapping_group.setLayout(mapping_layout)
        main_layout.addWidget(mapping_group)
        
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
        settings_group = QGroupBox("Scaling Settings")
        settings_layout = QFormLayout()
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Scale Video to match Audio", "Scale Audio to match Video"])
        self.combo_scale.setEnabled(False)
        self.combo_scale.currentIndexChanged.connect(lambda: self.save_settings())
        
        settings_layout.addRow("Scaling Mode:", self.combo_scale)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Controls and Progress
        controls_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze && Sync")
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.lbl_status = QLabel("Ready")
        
        controls_layout.addWidget(self.btn_analyze)
        controls_layout.addWidget(self.progress_bar, stretch=1)
        controls_layout.addWidget(self.lbl_status)
        main_layout.addLayout(controls_layout)
        
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
        main_layout.addWidget(self.canvas, stretch=1)
        
    def log(self, msg):
        self.log_text.append(msg)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
            
        info_text = f"{count} valid files found | Resolution: {res} (from first file)"
        
        if is_audio:
            self.lbl_audio.setText(self.audio_folder_path if self.audio_folder_path else "No folder selected")
            self.info_audio.setText(info_text)
        else:
            self.lbl_video.setText(self.video_folder_path if self.video_folder_path else "No folder selected")
            self.info_video.setText(info_text)
        
    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    
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
                    
                scale_mode = settings.get("scale_mode", 0)
                self.combo_scale.setCurrentIndex(scale_mode)
                
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
            "audio_folder": self.audio_folder_path,
            "video_folder": self.video_folder_path,
            "scale_mode": self.combo_scale.currentIndex(),
            "ratio_a": self.spin_ratio_a.value(),
            "ratio_v": self.spin_ratio_v.value()
        }
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
        title = "Select Audio Folder" if folder_num == 1 else "Select Video Folder"
        folder_path = QFileDialog.getExistingDirectory(self, title, "")
        if folder_path:
            if folder_num == 1:
                self.audio_folder_path = folder_path
                self.audio_files = self.get_media_files(folder_path)
                self.set_folder_info(True)
            else:
                self.video_folder_path = folder_path
                self.video_files = self.get_media_files(folder_path)
                self.set_folder_info(False)
                
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
        self.log_text.clear()
        self.figure.clear()
        self.canvas.draw()
        
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
        
        self.worker = AnalysisWorker(audio_file, video_file, scale_factor, mode)
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
        self.lbl_status.setText("Error")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
    def on_analysis_finished(self, data1, data2, scale_factor, shift, mode):
        self.log("Analysis complete. Plotting results...")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Finished")
        self.btn_analyze.setEnabled(True)
        
        self.plot_results(data1, data2, scale_factor, shift, mode)
        
    def plot_results(self, data1, data2, scale_factor, shift, mode):
        self.figure.clear()
        
        t1, r1, g1, b1 = data1
        t2, r2, g2, b2 = data2
        
        dr1 = np.gradient(r1)
        dg1 = np.gradient(g1)
        db1 = np.gradient(b1)
        grad1 = np.sqrt(dr1**2 + dg1**2 + db1**2)
        
        dr2 = np.gradient(r2)
        dg2 = np.gradient(g2)
        db2 = np.gradient(b2)
        grad2 = np.sqrt(dr2**2 + dg2**2 + db2**2)
        
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312)
        ax3 = self.figure.add_subplot(313)
        
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor('#303134')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
        
        # Plot Video 1 (Gradient)
        ax1.plot(t1, r1, color='red', alpha=0.3, label='Red Raw')
        ax1.plot(t1, grad1, color='white', label='Gradient Magnitude', linewidth=1.5)
        ax1.set_ylabel('Change Intensity')
        ax1.set_title(f'Audio Source - Gradient')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot Video 2 (Gradient)
        ax2.plot(t2, r2, color='red', alpha=0.3, label='Red Raw')
        ax2.plot(t2, grad2, color='white', label='Gradient Magnitude', linewidth=1.5)
        ax2.set_ylabel('Change Intensity')
        ax2.set_title(f'Video Source - Gradient')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot Aligned Gradients
        if mode == 0:
            ax3.plot(t1, grad1, color='white', label='Audio Source Gradient', linewidth=2)
            t_aligned = np.array(t2) * scale_factor + shift
            ax3.plot(t_aligned, grad2, color='cyan', label=f'Video Source (x{scale_factor:.3f} + {shift:.2f}s)', linewidth=2, linestyle='--')
            ax3.set_xlabel('Time (Audio Source Timebase)')
            t_ref = t1
        else:
            ax3.plot(t2, grad2, color='white', label='Video Source Gradient', linewidth=2)
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
            
        ax3.set_ylabel('Gradient Magnitude')
        ax3.set_title('Aligned Gradient Curves')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAudioSyncApp()
    window.show()
    sys.exit(app.exec())
