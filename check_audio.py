import av

def check_audio(video_path):
    try:
        container = av.open(video_path)
        if len(container.streams.audio) > 0:
            audio_stream = container.streams.audio[0]
            print(f"File: {video_path}")
            print(f"  Audio Stream found: {audio_stream}")
            print(f"  Sample Rate: {audio_stream.rate}")
            print(f"  Channels: {audio_stream.channels}")
        else:
            print(f"File: {video_path} - No audio stream found.")
        container.close()
    except Exception as e:
        print(f"Error reading {video_path}: {e}")

if __name__ == "__main__":
    check_audio("rabbit1080.mp4")
    check_audio("rabbitsd.mp4")

