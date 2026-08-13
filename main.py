import sys
import os
import glob
import json
import time
import av
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from scipy import signal, interpolate
from natsort import natsorted
import pickle
import hashlib
import concurrent.futures
import multiprocessing

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


def get_cache_path(cache_dir, filepath):
    h = hashlib.md5(os.path.abspath(filepath).encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{h}.pkl")

def process_video_file(video_path, output_path, max_frames=float('inf'), frame_step=1, progress_queue=None):
    """Module-level function for multiprocessing extraction."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        
        timestamps = []
        reds = []
        greens = []
        blues = []
        hues = []

        fps = None
        try:
            rate = stream.average_rate or stream.r_frame_rate
            if rate and rate.denominator != 0:
                fps = float(rate)
        except Exception:
            pass

        frame_number = 0
        first_frame_time = None
        last_frame_time = None
        
        for frame in container.decode(stream):
            if frame_number >= max_frames:
                break
            
            if first_frame_time is None:
                first_frame_time = frame.time
            last_frame_time = frame.time
            
            frame_number += 1
            if progress_queue and frame_number % 50 == 0:
                progress_queue.put(50)
            
            if (frame_number - 1) % frame_step != 0:
                continue
            
            small_frame = frame.reformat(width=200, height=112, format="rgb24")
            image = small_frame.to_ndarray()
            r, g, b, h = get_color_percentages(image)
            
            timestamps.append(frame.time)
            reds.append(r)
            greens.append(g)
            blues.append(b)
            hues.append(h)
                
        container.close()
        
        if progress_queue and frame_number % 50 != 0:
            progress_queue.put(frame_number % 50)
        
        if not fps or fps <= 0:
            if first_frame_time is not None and last_frame_time is not None and last_frame_time > first_frame_time:
                duration = last_frame_time - first_frame_time
                fps = (frame_number - 1) / duration
            else:
                raise ValueError("Could not determine FPS from metadata or frame timestamps")
                
        data = (timestamps, reds, greens, blues, hues, [], fps)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_path, output_path)
        return True
    except Exception as e:
        return e


class IndexWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, audio_files, video_files, output_dir, frame_step=1, max_frames=float('inf'), resume=True):
        super().__init__()
        self.resume = resume
        self.audio_files = audio_files
        self.video_files = video_files
        self.output_dir = output_dir
        self.frame_step = max(1, frame_step)
        self.max_frames = max_frames
        self._is_cancelled = False
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sync_cache")

    def cancel(self):
        self._is_cancelled = True
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False, cancel_futures=True)

    def run(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        all_files = self.audio_files + self.video_files
        total_files = len(all_files)
        
        self.progress_signal.emit(0, total_files, "Checking existing cache...")
        files_to_process = []
        for i, file_path in enumerate(all_files):
            self.progress_signal.emit(i, total_files, f"Checking cache {i}/{total_files}...")
            if self._is_cancelled:
                self.finished_signal.emit(False, "Indexing stopped by user.")
                return
                
            cache_path = get_cache_path(self.cache_dir, file_path)
            if self.resume and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, tuple) and len(data) >= 6:
                            continue
                except:
                    pass
            
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            
            files_to_process.append((file_path, cache_path))
            time.sleep(0.005)  # Yield GIL to prevent GUI freeze
            
        if not files_to_process:
            self.finished_signal.emit(True, "All files are already indexed.")
            return
                
        self.progress_signal.emit(0, len(files_to_process), "Calculating total frames...")
        total_frames_to_process = 0
        for i, (fp, cp) in enumerate(files_to_process):
            time.sleep(0.005)  # Yield GIL to prevent GUI freeze
            if self._is_cancelled:
                self.finished_signal.emit(False, "Indexing stopped by user.")
                return
                
            self.progress_signal.emit(i, len(files_to_process), f"Calculating total frames {i}/{len(files_to_process)}...")
            try:
                container = av.open(fp)
                stream = container.streams.video[0]
                frames = stream.frames
                if not frames or frames == 0:
                    rate = stream.average_rate or stream.r_frame_rate
                    fps = float(rate) if rate and rate.denominator != 0 else 25.0
                    dur = 0
                    if stream.duration and stream.time_base:
                        dur = float(stream.duration * stream.time_base)
                    elif container.duration:
                        dur = container.duration / 1000000.0
                    frames = int(dur * fps)
                total_frames_to_process += min(frames, self.max_frames)
                container.close()
            except Exception:
                pass
                
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        self.executor = concurrent.futures.ProcessPoolExecutor()
        futures = {
            self.executor.submit(process_video_file, fp, cp, self.max_frames, self.frame_step, progress_queue): (fp, cp)
            for fp, cp in files_to_process
        }
        
        processed_frames = 0
        self.progress_signal.emit(0, total_frames_to_process, f"Processed 0/{total_frames_to_process} frames")
        
        import queue
        while not all(f.done() for f in futures):
            if self._is_cancelled:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.finished_signal.emit(False, "Indexing stopped by user.")
                return
                
            try:
                msg = progress_queue.get(timeout=0.2)
                processed_frames += msg
                self.progress_signal.emit(processed_frames, total_frames_to_process, f"Processed {processed_frames}/{total_frames_to_process} frames")
            except queue.Empty:
                pass
                
        # Drain remaining items in the queue just in case
        while not progress_queue.empty():
            try:
                processed_frames += progress_queue.get_nowait()
            except queue.Empty:
                break
        self.progress_signal.emit(processed_frames, total_frames_to_process, f"Processed {processed_frames}/{total_frames_to_process} frames")
        
        has_errors = False
        for future in concurrent.futures.as_completed(futures):
            fp, cp = futures[future]
            try:
                result = future.result()
                if isinstance(result, Exception):
                    self.log_signal.emit(f"Error indexing {os.path.basename(fp)}: {result}")
                    has_errors = True
                elif result is not True:
                    self.log_signal.emit(f"Error indexing {os.path.basename(fp)}: {result}")
                    has_errors = True
            except Exception as e:
                self.log_signal.emit(f"Failed to process {os.path.basename(fp)}: {e}")
                has_errors = True

        self.executor.shutdown(wait=True)
        
        if not self._is_cancelled and not has_errors:
            self.finished_signal.emit(True, "Indexing complete.")
        else:
            self.finished_signal.emit(False, "Indexing failed due to errors.")


class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object, float, float, int)
    error_signal = pyqtSignal(str)

    def __init__(self, data1, data2, scale_factor, mode):
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.scale_factor = scale_factor
        self.mode = mode
        self._is_cancelled = False

    def run(self):
        self.log_signal.emit("Calculating time shift...")
        self.progress_signal.emit(0, 0, "Calculating shift...")
        
        t1, r1, g1, b1, h1 = self.data1[:5]
        t2, r2, g2, b2, h2 = self.data2[:5]
        
        grad1 = np.array(h1)
        grad2 = np.array(h2)
        
        if self.mode == 0:
            shift = find_time_shift(t1, grad1, t2, grad2, self.scale_factor, log_callback=lambda m: self.log_signal.emit(m))
        else:
            shift = find_time_shift(t2, grad2, t1, grad1, self.scale_factor, log_callback=lambda m: self.log_signal.emit(m))
        
        if not self._is_cancelled:
            self.finished_signal.emit(self.data1, self.data2, self.scale_factor, shift, self.mode)
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
        self.hover_container_audio = None
        self.hover_container_video = None
        
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
        
        # Index Files Button
        h_index = QHBoxLayout()
        self.btn_index = QPushButton("Index Files")
        self.btn_index.clicked.connect(self.toggle_indexing)
        self.btn_index.setEnabled(False)
        
        self.btn_reindex = QPushButton("Re-index Files")
        self.btn_reindex.clicked.connect(self.trigger_reindex)
        self.btn_reindex.setEnabled(False)
        
        self.progress_index = QProgressBar()
        self.progress_index.setValue(0)
        self.progress_index.setVisible(False)
        self.progress_index.setTextVisible(True)
        
        self.index_status = QLabel("")
        self.index_status.setStyleSheet("color: #fbbc04; font-weight: bold;")
        
        h_index.addWidget(self.btn_index)
        h_index.addWidget(self.btn_reindex)
        h_index.addWidget(self.progress_index, stretch=1)
        h_index.addWidget(self.index_status)
        
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
        files_layout.addLayout(h_index)
        files_group.setLayout(files_layout)
        main_layout.addWidget(files_group)
        
        # Lower Panel
        self.lower_panel = QWidget()
        lower_layout = QVBoxLayout(self.lower_panel)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        
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
        lower_layout.addWidget(pair_group)
        
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
        lower_layout.addWidget(settings_group)
        
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
        lower_layout.addLayout(controls_layout)
        
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
        lower_layout.addLayout(diff_layout)
        
        # Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        lower_layout.addWidget(self.log_text)
        
        # Plot Canvas
        self.figure = Figure(figsize=(8, 6))
        self.figure.patch.set_facecolor('#202124')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_hover)
        self.canvas.mpl_connect("axes_leave_event", lambda e: self.preview_popup.hide())
        lower_layout.addWidget(self.canvas)
        
        main_layout.addWidget(self.lower_panel)
        self.lower_panel.setEnabled(False)
        
    def toggle_indexing(self):
        if self.btn_index.text() == "Stop Indexing":
            self.stop_indexing()
        else:
            self.start_indexing(resume=True)
            
    def trigger_reindex(self):
        self.reset_ui()
        self.start_indexing(resume=False)
            
    def stop_indexing(self):
        if hasattr(self, 'index_worker') and self.index_worker.isRunning():
            self.index_worker.cancel()
            self.index_status.setText("Stopping...")
            self.btn_index.setEnabled(False)
            
    def reset_ui(self):
        self.lower_panel.setEnabled(False)
        if hasattr(self, 'matched_pairs'):
            self.matched_pairs = []
        self.current_pair_idx = 0
        self.lbl_nav.setText("Pair 0 of 0")
        self.lbl_pair_audio.setText("No audio file")
        self.lbl_pair_video.setText("No video file")
        self.info_pair_audio.setText("")
        self.info_pair_video.setText("")
        self.combo_scale.setEnabled(False)
        self.figure.clear()
        self.canvas.draw()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Ready")
        
    def start_indexing(self, resume=True):
        self.btn_index.setText("Stop Indexing")
        self.btn_index.setStyleSheet("""
            QPushButton { background-color: #d32f2f; color: white; }
            QPushButton:disabled { background-color: #555555; color: #aaaaaa; }
        """)
        self.btn_reindex.setEnabled(False)
        self.progress_index.setVisible(True)
        self.progress_index.setValue(0)
        
        self.lower_panel.setEnabled(False)
        self.index_status.setText("Indexing...")
        
        self.index_worker = IndexWorker(
            self.audio_files,
            self.video_files,
            self.output_folder_path,
            self.spin_frame_step.value(),
            float('inf'),
            resume=resume
        )
        self.index_worker.progress_signal.connect(self.update_index_progress)
        self.index_worker.log_signal.connect(self.log)
        self.index_worker.finished_signal.connect(self.on_index_finished)
        self.index_worker.start()

    def update_index_progress(self, val, total, status):
        self.progress_index.setMaximum(total)
        self.progress_index.setValue(val)
        self.index_status.setText(status)

    def on_index_finished(self, success, msg):
        self.btn_index.setStyleSheet("")
        self.btn_index.setText("Index Files" if success else "Resume Indexing")
        self.btn_index.setEnabled(True)
        self.btn_reindex.setEnabled(self.has_cache_files())
        self.progress_index.setVisible(False)
        self.log(msg)
        
        if success:
            self.index_status.setText("Indexed ✅")
            self.lower_panel.setEnabled(True)
            self.perform_matching()
        else:
            self.index_status.setText("Failed")

    def perform_matching(self):
        self.log("Loading indexed curves into RAM...")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sync_cache")
        
        video_curves = {}
        for vf in self.video_files:
            cache_path = get_cache_path(cache_dir, vf)
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    video_curves[vf] = data
            except:
                continue

        self.log("Matching audio to video files...")
        self.matched_pairs = []
        
        for af in self.audio_files:
            a_cache_path = get_cache_path(cache_dir, af)
            try:
                with open(a_cache_path, 'rb') as f:
                    a_data = pickle.load(f)
            except:
                continue
                
            t1, r1, g1, b1, h1 = a_data[:5]
            a_fps = a_data[6]
            y1 = np.array(h1)
            
            best_match_file = None
            best_match_peak = -float('inf')
            
            for vf, v_data in video_curves.items():
                t2, r2, g2, b2, h2 = v_data[:5]
                v_fps = v_data[6]
                y2 = np.array(h2)
                
                scale_factor = 1.0
                mode = self.combo_scale.currentIndex()
                if mode == 0:
                    scale_factor = a_fps / v_fps if v_fps else 1.0
                else:
                    scale_factor = v_fps / a_fps if a_fps else 1.0
                
                t2_scaled = np.array(t2) * scale_factor
                
                fs = 200.0
                try:
                    t1_uniform = np.arange(t1[0], t1[-1], 1/fs)
                    interp1 = interpolate.interp1d(t1, y1, kind='linear', fill_value="extrapolate")
                    y1_uniform = interp1(t1_uniform)
                    y1_centered = y1_uniform - np.mean(y1_uniform)
                    
                    t2_uniform = np.arange(t2_scaled[0], t2_scaled[-1], 1/fs)
                    interp2 = interpolate.interp1d(t2_scaled, y2, kind='linear', fill_value="extrapolate")
                    y2_uniform = interp2(t2_uniform)
                    y2_centered = y2_uniform - np.mean(y2_uniform)
                    
                    corr = signal.correlate(y1_centered, y2_centered, mode='valid')
                    if len(corr) == 0:
                        continue
                    peak = np.max(corr)
                    
                    if peak > best_match_peak:
                        best_match_peak = peak
                        best_match_file = vf
                except:
                    continue
            
            if best_match_file:
                self.matched_pairs.append((af, best_match_file))
                self.log(f"Matched Audio: {os.path.basename(af)} -> Video: {os.path.basename(best_match_file)}")
            else:
                self.log(f"Failed to find match for {os.path.basename(af)}")
                
        self.current_pair_idx = 0
        self.load_current_pair()

    def has_cache_files(self):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sync_cache")
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith('.pkl'):
                    return True
        return False

    def update_files_list(self):
        valid = len(self.audio_files) > 0 and len(self.video_files) > 0 and self.output_folder_path is not None
        self.btn_index.setEnabled(valid)
        self.btn_reindex.setEnabled(valid and self.has_cache_files())
        if valid and self.btn_index.text() not in ["Stop Indexing", "Resume Indexing"]:
            self.btn_index.setText("Index Files")
        elif not valid:
            self.btn_index.setText("Index Files")

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
            self.btn_prev.setVisible(True)
            self.btn_next.setVisible(True)
            self.lbl_nav.setVisible(True)
        else:
            self.btn_audio.setText("Select Audio File")
            self.btn_video.setText("Select Video File")
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.lbl_nav.setVisible(False)
            
        self.set_folder_info(True)
        self.set_folder_info(False)
        self.load_current_pair()
        self.save_settings()
        self.update_files_list()

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
                
                self.current_index = 0
                self.load_current_pair()
            except Exception as e:
                print(f"Failed to load settings: {e}")
        self.update_files_list()

    def save_settings(self):
        settings = {
            "app_mode": self.app_mode,
            "output_folder": self.output_folder_path,
            "scale_mode": self.combo_scale.currentIndex(),
            "frame_step": self.spin_frame_step.value()
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

    def load_current_pair(self):
        self.figure.clear()
        self.canvas.draw()
        
        if self.hover_container_audio:
            try: self.hover_container_audio.close()
            except: pass
        if self.hover_container_video:
            try: self.hover_container_video.close()
            except: pass
            
        self.hover_container_audio = None
        self.hover_container_video = None
        self.hover_stream_audio = None
        self.hover_stream_video = None
        
        if not hasattr(self, 'matched_pairs') or len(self.matched_pairs) == 0:
            self.lbl_nav.setText("Pair 0 of 0")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.lbl_pair_audio.setText("No audio file")
            self.lbl_pair_video.setText("No video file")
            self.info_pair_audio.setText("")
            self.info_pair_video.setText("")
            self.combo_scale.setEnabled(False)
            return
            
        total = len(self.matched_pairs)
        self.lbl_nav.setText(f"Pair {self.current_pair_idx + 1} of {total}")
        self.btn_prev.setEnabled(self.current_pair_idx > 0)
        self.btn_next.setEnabled(self.current_pair_idx < total - 1)
        
        audio_file, video_file = self.matched_pairs[self.current_pair_idx]
        
        self.lbl_pair_audio.setText(f"🎵 Audio: {os.path.basename(audio_file)}")
        self.lbl_pair_video.setText(f"🎬 Video: {os.path.basename(video_file)}")
        
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sync_cache")
        try:
            with open(get_cache_path(cache_dir, audio_file), 'rb') as f:
                a_data = pickle.load(f)
            with open(get_cache_path(cache_dir, video_file), 'rb') as f:
                v_data = pickle.load(f)
                
            self.audio_fps = a_data[6]
            self.video_fps = v_data[6]
            
            a_dur = a_data[0][-1] if len(a_data[0]) > 0 else 0
            v_dur = v_data[0][-1] if len(v_data[0]) > 0 else 0
            a_frames = len(a_data[0])
            v_frames = len(v_data[0])
            
            self.info_pair_audio.setText(f"FPS: {self.audio_fps:.2f} | Duration: {a_dur:.2f}s | Frames: {a_frames}")
            self.info_pair_video.setText(f"FPS: {self.video_fps:.2f} | Duration: {v_dur:.2f}s | Frames: {v_frames}")
            
            try:
                self.hover_container_audio = av.open(audio_file)
                self.hover_stream_audio = self.hover_container_audio.streams.video[0]
                self.hover_stream_audio.thread_type = "AUTO"
            except Exception:
                pass
                
            try:
                self.hover_container_video = av.open(video_file)
                self.hover_stream_video = self.hover_container_video.streams.video[0]
                self.hover_stream_video.thread_type = "AUTO"
            except Exception:
                pass
            
            self.btn_analyze.setEnabled(True)
            self.combo_scale.setEnabled(True)
            
            if self.audio_fps and self.video_fps and abs(self.audio_fps - self.video_fps) > 0.1:
                self.info_pair_audio.setText(self.info_pair_audio.text() + " ⚠️ FPS MISMATCH")
                self.info_pair_video.setText(self.info_pair_video.text() + " ⚠️ FPS MISMATCH")
        except Exception as e:
            self.log(f"Error loading cache for pair: {e}")
            self.btn_analyze.setEnabled(False)

    def next_pair(self):
        self.current_pair_idx += 1
        self.load_current_pair()
        
    def prev_pair(self):
        self.current_pair_idx -= 1
        self.load_current_pair()

    def start_analysis(self):
        if not hasattr(self, 'matched_pairs') or not self.matched_pairs:
            return
            
        audio_file, video_file = self.matched_pairs[self.current_pair_idx]
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sync_cache")
        
        try:
            with open(get_cache_path(cache_dir, audio_file), 'rb') as f:
                data1 = pickle.load(f)
            with open(get_cache_path(cache_dir, video_file), 'rb') as f:
                data2 = pickle.load(f)
        except Exception as e:
            self.log(f"Cache missing for analysis: {e}")
            return
            
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.clear()
        self.figure.clear()
        self.canvas.draw()
        self.progress_bar.setValue(0)
        
        mode = self.combo_scale.currentIndex()
        if mode == 0:
            scale_factor = self.audio_fps / self.video_fps if self.video_fps else 1.0
        else:
            scale_factor = self.video_fps / self.audio_fps if self.audio_fps else 1.0
            
        self.lbl_status.setText("Analyzing...")
        self.log(f"Using scale factor: {scale_factor:.5f}")
        
        self.worker = AnalysisWorker(
            data1,
            data2,
            scale_factor,
            mode
        )
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

    def get_dynamic_thumbnail(self, container, stream, target_time):
        if not container or not stream:
            return None
        try:
            target_pts = int(target_time / float(stream.time_base))
            container.seek(target_pts, stream=stream)
            for frame in container.decode(stream):
                small_frame = frame.reformat(width=200, height=112, format="rgb24")
                image = small_frame.to_ndarray()
                h, w, ch = image.shape
                qimg = QImage(image.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(qimg)
        except Exception:
            pass
        return None

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
                
                pix_audio = self.get_dynamic_thumbnail(self.hover_container_audio, self.hover_stream_audio, max(0, t_audio))
                pix_video = self.get_dynamic_thumbnail(self.hover_container_video, self.hover_stream_video, max(0, t_video))
                
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
            
        if not hasattr(self, 'matched_pairs') or not self.matched_pairs:
            return
            
        audio_file, video_file = self.matched_pairs[self.current_pair_idx]
        
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
