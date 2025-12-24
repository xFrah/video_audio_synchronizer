import av
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal, interpolate


def get_color_percentages(image):
    """
    Calculates the intensity of Red, Green, and Blue channels as a fraction of
    the maximum possible intensity (num_pixels * 255).
    
    Args:
        image: Numpy array of shape (H, W, 3). Assumes BGR format (standard for OpenCV/PyAV).
    
    Returns:
        tuple: (red_percentage, green_percentage, blue_percentage) as floats between 0.0 and 1.0.
    """

    # Calculate total pixels (Height * Width)
    total_pixels = image.shape[0] * image.shape[1]

    # Maximum possible sum for a single channel (all pixels = 255)
    max_intensity = total_pixels * 255.0

    # Calculate sum of each channel
    # Image is BGR (Blue=0, Green=1, Red=2)
    b_sum = np.sum(image[:, :, 0])
    g_sum = np.sum(image[:, :, 1])
    r_sum = np.sum(image[:, :, 2])

    # Calculate fractions
    b_pct = b_sum / max_intensity
    g_pct = g_sum / max_intensity
    r_pct = r_sum / max_intensity

    return r_pct, g_pct, b_pct


def analyze_video(video_path, MAX_FRAMES=1000):
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    timestamps = []
    reds = []
    greens = []
    blues = []

    # Get total frames for progress bar
    total_frames = stream.frames
    
    print("Analyzing frames...")
    frame_number = 0
    for frame in tqdm(container.decode(stream), total=min(total_frames, MAX_FRAMES or total_frames), unit="frame"):
        if frame_number >= MAX_FRAMES:
            break
        frame_number += 1
        # Convert to numpy array (BGR format)
        image = frame.to_ndarray(format="bgr24")
        
        # Calculate percentages
        r, g, b = get_color_percentages(image)
        
        # Store data (frame.time is the presentation timestamp in seconds)
        timestamps.append(frame.time)
        reds.append(r)
        greens.append(g)
        blues.append(b)
        
    container.close()
    return timestamps, reds, greens, blues


def find_time_shift(t1, y1, t2, y2, scale):
    """
    Finds the optimal time shift to align y2 to y1, given a fixed scale factor.
    Model: y1(t) ~ y2(scale * t + shift)
    
    Args:
        scale: Fixed scaling factor applied to t2.
        
    Returns:
        float: best_shift
    """
    t1 = np.array(t1)
    y1 = np.array(y1)
    t2 = np.array(t2)
    y2 = np.array(y2)
    
    # Apply scale immediately to t2
    t2_scaled = t2 * scale
    
    # Common sampling for comparison
    fs = 1000.0
    
    # Pre-interpolate y1 once
    t1_uniform = np.arange(t1[0], t1[-1], 1/fs)
    interp1 = interpolate.interp1d(t1, y1, kind='linear', fill_value="extrapolate")
    y1_uniform = interp1(t1_uniform)
    y1_centered = y1_uniform - np.mean(y1_uniform)
    
    # Create a uniform grid for the scaled video 2
    # We want the sample spacing to match y1's spacing (1/fs)
    t2_uniform = np.arange(t2_scaled[0], t2_scaled[-1], 1/fs)
    interp2 = interpolate.interp1d(t2_scaled, y2, kind='linear', fill_value="extrapolate")
    y2_uniform = interp2(t2_uniform)
    y2_centered = y2_uniform - np.mean(y2_uniform)
    
    # Cross-correlation
    corr = signal.correlate(y1_centered, y2_centered, mode='full')
    lags = signal.correlation_lags(len(y1_centered), len(y2_centered), mode='full')
    
    # Find max correlation
    lag_idx = np.argmax(corr)
    lag_samples = lags[lag_idx]
    coarse_shift = lag_samples / fs
    
    print(f"Coarse shift detected: {coarse_shift:.4f}s")

    # Fine-tuning step: minimize MSE locally
    # Search window: +/- 0.1 seconds around coarse shift
    # Step size: 0.001 seconds (1ms) is standard, but let's go finer if needed
    fine_shifts = np.linspace(coarse_shift - 0.1, coarse_shift + 0.1, 201)
    
    best_mse = float('inf')
    best_shift = coarse_shift
    
    # Pre-calculate y1 function for fast evaluation
    f1 = interpolate.interp1d(t1, y1, kind='linear', fill_value="extrapolate")
    
    # We will evaluate error on the overlapping interval
    # t2_scaled + shift must be within [t1_min, t1_max]
    
    for s in fine_shifts:
        # Time points for video 2 aligned
        t2_shifted = t2_scaled + s
        
        # Find overlap
        start_t = max(t1[0], t2_shifted[0])
        end_t = min(t1[-1], t2_shifted[-1])
        
        if end_t <= start_t:
            continue
            
        # Evaluate on a grid in the overlap
        t_eval = np.arange(start_t, end_t, 0.01) # 100Hz eval grid for MSE
        if len(t_eval) < 10: continue
        
        # y1 values
        v1 = f1(t_eval)
        
        # y2 values: need to interpolate y2 at (t_eval - s) / scale (but t2_scaled is already scaled)
        # So we need y2 value at time t on the t2_scaled axis, which is t - s
        f2 = interpolate.interp1d(t2_scaled, y2, kind='linear', fill_value="extrapolate")
        v2 = f2(t_eval - s)
        
        mse = np.mean((v1 - v2)**2)
        
        if mse < best_mse:
            best_mse = mse
            best_shift = s
            
    print(f"Fine-tuned shift: {best_shift:.4f}s")
    return best_shift


def plot_results(data1, data2, scale_factor):
    """
    Plots the color intensity results for two videos in subplots.
    
    Args:
        data1: Tuple (timestamps, reds, greens, blues) for video 1
        data2: Tuple (timestamps, reds, greens, blues) for video 2
        scale_factor: Manual scale factor (Video 1 FPS / Video 2 FPS)
    """
    
    # Unpack data
    t1, r1, g1, b1 = data1
    t2, r2, g2, b2 = data2
    
    # Calculate Gradients (Derivatives) instead of simple Average
    # We want a signal that represents "rate of change"
    # Gradient of each channel
    dr1 = np.gradient(r1)
    dg1 = np.gradient(g1)
    db1 = np.gradient(b1)
    # Combine gradients: L2 norm of the change vector (magnitude of change)
    grad1 = np.sqrt(dr1**2 + dg1**2 + db1**2)
    
    dr2 = np.gradient(r2)
    dg2 = np.gradient(g2)
    db2 = np.gradient(b2)
    grad2 = np.sqrt(dr2**2 + dg2**2 + db2**2)
    
    # Calculate time shift given the manual scale using GRADIENTS
    shift = find_time_shift(t1, grad1, t2, grad2, scale_factor)
    
    print(f"Using Manual Scale Factor: {scale_factor:.4f}")
    print(f"Detected Time Shift: {shift:.4f} seconds")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
    
    # Plot Video 1 (Gradient)
    ax1.plot(t1, r1, color='red', alpha=0.3, label='Red Raw') # Show raw faint
    ax1.plot(t1, grad1, color='black', label='Gradient Magnitude', linewidth=1.5)
    ax1.set_ylabel('Change Intensity')
    ax1.set_title(f'Video 1 (Reference) - Gradient')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot Video 2 (Gradient)
    ax2.plot(t2, r2, color='red', alpha=0.3, label='Red Raw')
    ax2.plot(t2, grad2, color='black', label='Gradient Magnitude', linewidth=1.5)
    ax2.set_xlabel('Original Time (seconds)')
    ax2.set_ylabel('Change Intensity')
    ax2.set_title(f'Video 2 (Original) - Gradient')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot Aligned Gradients
    ax3.plot(t1, grad1, color='black', label='Video 1 Gradient', linewidth=2)
    
    # Apply transform to t2
    t2_aligned = np.array(t2) * scale_factor + shift
    
    ax3.plot(t2_aligned, grad2, color='blue', label=f'Video 2 Gradient (x{scale_factor:.3f} + {shift:.2f}s)', linewidth=2, linestyle='--')
    ax3.set_xlabel('Time (Video 1 Timebase)')
    ax3.set_ylabel('Gradient Magnitude')
    ax3.set_title('Aligned Gradient Curves')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace with your video paths
    video_fixed_audio = "rabbitsd.mp4"
    video_stretch = "rabbit1080.mp4" 
    
    # Manual Scale Factor
    # If Video 2 (25fps) is sped up relative to Video 1 (24fps),
    # we need to STRETCH Video 2 to match Video 1.
    # Scale = Target Duration / Current Duration
    # Scale = 25 / 24 ~= 1.041666...
    # Or enter 1.0 if no scaling is needed.
    SCALE_FACTOR = 24.0 / 25.0 
    
    print(f"Processing {video_fixed_audio}...")
    data1 = analyze_video(video_fixed_audio)
    
    print(f"Processing {video_stretch}...")
    data2 = analyze_video(video_stretch)
    
    plot_results(data1, data2, SCALE_FACTOR)
