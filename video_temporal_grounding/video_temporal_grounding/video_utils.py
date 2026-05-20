import cv2
import numpy as np


def sample_video_frames_sliding_window(
    video_path,
    window_size=2.0,   # 每个窗口2秒
    stride=2.0,        # 步长2秒
    num_frames=16      # 每个窗口采样16帧
):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print("========== VIDEO INFO ==========")
    print("Video path:", video_path)
    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Duration:", duration)
    print("Window size:", window_size)
    print("Stride:", stride)
    print("Frames per window:", num_frames)
    print("================================")

    windows = []

    start_time = 0.0

    while start_time < duration:

        end_time = start_time + window_size
        if end_time > duration:
            end_time = duration

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        print("\n---- New Window ----")
        print("Start time:", start_time)
        print("End time:", end_time)
        print("Start frame:", start_frame)
        print("End frame:", end_frame)

        frame_indices = np.linspace(
            start_frame,
            end_frame,
            num_frames
        ).astype(int)

        #print("Sampled frame indices:", frame_indices)

        frames = []
        timestamps = []

        for idx in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

            ts = idx / fps
            timestamps.append(ts)

        #print("Collected frames:", len(frames))
        #print("Timestamps:", timestamps)

        windows.append({
            "start": start_time,
            "end": end_time,
            "frames": frames,
            "timestamps": timestamps
        })

        start_time += stride

    cap.release()

    #print("\n========== DONE ==========")
    #print("Total windows:", len(windows))

    return windows, duration