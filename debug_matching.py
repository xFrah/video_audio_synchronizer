import os
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate

def get_cache_path(cache_dir, filepath):
    h = hashlib.md5(os.path.abspath(filepath).encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{h}.pkl")

audio_file = r"M:\Hero 108\Stagione 1\Hero-108---1x01---Ita.mp4"
video_s01e01 = r"M:\temp_downloads\vixsrc\Hero 108\tt1674117_S01E01 - Rabbit Castle; Elephant Castle.mp4"
video_s02e03 = r"M:\temp_downloads\vixsrc\Hero 108\tt1674117_S02E03 - Stingray  Pangolin Castle.mp4"

cache_dir = r"E:\video_audio_synchronizer\.sync_cache"

audio_cache = get_cache_path(cache_dir, audio_file)
video1_cache = get_cache_path(cache_dir, video_s01e01)
video2_cache = get_cache_path(cache_dir, video_s02e03)

def load_data(cp):
    if not os.path.exists(cp):
        print(f"Cache file not found: {cp}")
        return None
    with open(cp, 'rb') as f:
        return pickle.load(f)

print("Loading audio...")
a_data = load_data(audio_cache)
print("Loading video S01E01...")
v1_data = load_data(video1_cache)
print("Loading video S02E03...")
v2_data = load_data(video2_cache)

if not (a_data and v1_data and v2_data):
    print("Missing cache files, make sure they are indexed.")
    exit(1)

def compute_correlation(a_data, v_data, title):
    t1, r1, g1, b1, h1 = a_data[:5]
    a_fps = a_data[6]
    y1 = np.array(h1)
    
    t2, r2, g2, b2, h2 = v_data[:5]
    v_fps = v_data[6]
    y2 = np.array(h2)
    
    print(f"[{title}] Audio FPS: {a_fps:.3f} | Video FPS: {v_fps:.3f}")
    
    # Apply correct PAL speedup scale factor (v_fps / a_fps)
    if a_fps and v_fps:
        scale_factor = v_fps / a_fps
    else:
        scale_factor = 1.0
        
    print(f"Applying Scale Factor: {scale_factor:.5f}")
    t2_scaled = np.array(t2) * scale_factor
    
    if len(t1) > 1 and t1[-1] > t1[0]:
        fs = len(t1) / (t1[-1] - t1[0])
    else:
        fs = 25.0
        
    t1_uniform = np.arange(t1[0], t1[-1], 1/fs)
    interp1 = interpolate.interp1d(t1, y1, kind='nearest', bounds_error=False, fill_value=0)
    y1_uniform = interp1(t1_uniform)
    y1_centered = y1_uniform - np.mean(y1_uniform)
    
    t2_uniform = np.arange(t2_scaled[0], t2_scaled[-1], 1/fs)
    interp2 = interpolate.interp1d(t2_scaled, y2, kind='nearest', bounds_error=False, fill_value=0)
    y2_uniform = interp2(t2_uniform)
    y2_centered = y2_uniform - np.mean(y2_uniform)
    
    if len(y1_centered) <= len(y2_centered):
        y_short, y_long = y1_centered, y2_centered
    else:
        y_short, y_long = y2_centered, y1_centered
        
    corr = signal.correlate(y_long, y_short, mode='full')
    lags = signal.correlation_lags(len(y_long), len(y_short), mode='full')
    if len(corr) == 0:
        return 0
        
    peak = np.max(corr)
    best_lag_idx = np.argmax(corr)
    best_lag_frames = lags[best_lag_idx]
    shift_seconds = best_lag_frames / fs
    
    print(f"[{title}] Perfect Shift: {shift_seconds:.4f}s ({best_lag_frames} frames)")
    
    # Create the overlap graph
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: The correlation peak
    plt.subplot(2, 1, 1)
    plt.plot(lags, corr, color='purple')
    plt.axvline(x=best_lag_frames, color='red', linestyle='--', label=f'Best Shift: {shift_seconds:.3f}s')
    plt.title(f"{title} | Peak Score: {peak:.2f}")
    plt.xlabel("Shift (frames)")
    plt.ylabel("Correlation Sum")
    plt.legend()
    plt.grid(True)
    
    # Shift the short clip to its best position
    # best_lag_frames can be negative with mode='full'
    if best_lag_frames >= 0:
        max_len = max(len(y_long), best_lag_frames + len(y_short))
    else:
        max_len = max(abs(best_lag_frames) + len(y_long), len(y_short))
        
    aligned_long = np.full(max_len, np.nan)
    aligned_short = np.full(max_len, np.nan)
    
    if best_lag_frames >= 0:
        aligned_long[:len(y_long)] = y_long
        aligned_short[best_lag_frames:best_lag_frames+len(y_short)] = y_short
    else:
        # short starts before long
        start_idx = abs(best_lag_frames)
        aligned_long[start_idx:start_idx+len(y_long)] = y_long
        aligned_short[:len(y_short)] = y_short
        
    plt.subplot(2, 1, 2)
    plt.plot(aligned_long, label='Long Video (Hue)', color='blue', alpha=0.5)
    plt.plot(aligned_short, label='Audio Clip (Hue)', color='orange', alpha=0.9)
    plt.title("Best Fit Alignment")
    plt.xlabel("Frame Index")
    plt.ylabel("Hue (Centered)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    return peak

p1 = compute_correlation(a_data, v1_data, "Audio 1x01 vs Video S01E01")
p2 = compute_correlation(a_data, v2_data, "Audio 1x01 vs Video S02E03")

print(f"Peak S01E01: {p1:.2f}")
print(f"Peak S02E03: {p2:.2f}")

plt.show()
