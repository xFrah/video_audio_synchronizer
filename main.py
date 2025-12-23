import av
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def plot_results(data1, data2):
    """
    Plots the color intensity results for two videos in subplots.
    
    Args:
        data1: Tuple (timestamps, reds, greens, blues) for video 1
        data2: Tuple (timestamps, reds, greens, blues) for video 2
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Unpack data
    t1, r1, g1, b1 = data1
    t2, r2, g2, b2 = data2
    
    # Calculate Average Curves
    avg1 = [(r + g + b) / 3 for r, g, b in zip(r1, g1, b1)]
    avg2 = [(r + g + b) / 3 for r, g, b in zip(r2, g2, b2)]
    
    # Plot Video 1
    ax1.plot(t1, r1, color='red', label='Red', linewidth=1)
    ax1.plot(t1, g1, color='green', label='Green', linewidth=1)
    ax1.plot(t1, b1, color='blue', label='Blue', linewidth=1)
    ax1.plot(t1, avg1, color='black', label='Average', linewidth=2, linestyle='--')
    ax1.set_ylabel('Intensity (0-1)')
    ax1.set_title('Video 1: Color Intensity')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot Video 2
    ax2.plot(t2, r2, color='red', label='Red', linewidth=1)
    ax2.plot(t2, g2, color='green', label='Green', linewidth=1)
    ax2.plot(t2, b2, color='blue', label='Blue', linewidth=1)
    ax2.plot(t2, avg2, color='black', label='Average', linewidth=2, linestyle='--')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Intensity (0-1)')
    ax2.set_title('Video 2: Color Intensity')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace with your video paths
    video_path_1 = "rabbit1080.mp4" 
    video_path_2 = "rabbitsd.mp4"
    
    print(f"Processing {video_path_1}...")
    data1 = analyze_video(video_path_1)
    
    print(f"Processing {video_path_2}...")
    data2 = analyze_video(video_path_2)
    
    plot_results(data1, data2)
