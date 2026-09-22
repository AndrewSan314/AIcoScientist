import os
import sys
import json
import shutil
import cv2
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(base_dir, 'temp_video_frames')
    metadata_path = os.path.join(frames_dir, 'metadata.json')

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        sys.exit(1)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    if not metadata:
        print("Error: No frames recorded in metadata.")
        sys.exit(1)

    print(f"Loaded {len(metadata)} recorded frames.")
    t_start = metadata[0]['time']
    t_end = metadata[-1]['time']
    actual_duration = t_end - t_start
    print(f"Recorded actual time span: {actual_duration:.2f} seconds ({t_start:.2f}s to {t_end:.2f}s)")

    target_fps = 30
    target_duration_s = 60.0
    total_output_frames = int(target_fps * target_duration_s) # 1800 frames

    # Verify first frame to get dimensions
    first_frame_path = metadata[0]['file']
    test_img = cv2.imread(first_frame_path)
    if test_img is None:
        print(f"Error: Unable to read first frame at {first_frame_path}")
        sys.exit(1)

    height, width, _ = test_img.shape
    print(f"Video resolution: {width}x{height} at {target_fps} FPS (total target: {total_output_frames} frames)")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'docs', 'exhibition', 'recordings')
    os.makedirs(output_dir, exist_ok=True)
    output_mp4 = os.path.join(output_dir, 'aicoscientist_exhibition_showcase_1min_30fps.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, target_fps, (width, height))

    # Pre-extract frame times and files for fast binary search
    frame_times = np.array([m['time'] for m in metadata])
    frame_files = [m['file'] for m in metadata]

    # Map each output frame index (0..1799) to target timestamp in recorded timeline
    cached_frame = test_img
    last_file_idx = -1

    for i in range(total_output_frames):
        # Normalized progress [0, 1]
        progress = i / (total_output_frames - 1)
        target_t = t_start + progress * (t_end - t_start)

        # Find closest recorded frame
        idx = int(np.searchsorted(frame_times, target_t))
        if idx >= len(frame_times):
            idx = len(frame_times) - 1
        elif idx > 0 and abs(frame_times[idx - 1] - target_t) < abs(frame_times[idx] - target_t):
            idx = idx - 1

        if idx != last_file_idx:
            file_to_read = frame_files[idx]
            if os.path.exists(file_to_read):
                img = cv2.imread(file_to_read)
                if img is not None:
                    cached_frame = img
                    last_file_idx = idx

        out.write(cached_frame)
        if (i + 1) % 300 == 0 or i == total_output_frames - 1:
            print(f"Encoded {i + 1}/{total_output_frames} frames ({(i + 1) / target_fps:.1f}s)...")

    out.release()
    print("Video encoding completed!")

    file_size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
    print(f"Output saved to: {output_mp4} ({file_size_mb:.2f} MB)")

    # Copy to brain artifact directory as well
    artifact_rec_dir = r'C:\Users\ADMIN\.gemini\antigravity\brain\1da2a9bb-bc6b-4531-a202-a5d44499c205\recordings'
    os.makedirs(artifact_rec_dir, exist_ok=True)
    artifact_mp4 = os.path.join(artifact_rec_dir, 'aicoscientist_exhibition_showcase_1min_30fps.mp4')
    shutil.copyfile(output_mp4, artifact_mp4)
    print(f"Copied to artifact directory: {artifact_mp4}")

    # Clean up temp frames to preserve disk space
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("Cleaned up temporary frames directory.")

if __name__ == '__main__':
    main()
