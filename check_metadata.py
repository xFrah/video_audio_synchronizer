import av

def print_metadata(video_path):
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        print(f"File: {video_path}")
        print(f"  Duration: {float(stream.duration * stream.time_base) if stream.duration else 'Unknown'} s")
        print(f"  Frame Rate: {stream.average_rate} fps")
        print(f"  Time Base: {stream.time_base}")
        print(f"  Total Frames: {stream.frames}")
        container.close()
    except Exception as e:
        print(f"Error reading {video_path}: {e}")

if __name__ == "__main__":
    print_metadata("rabbit1080.mp4")
    print_metadata("rabbitsd.mp4")

