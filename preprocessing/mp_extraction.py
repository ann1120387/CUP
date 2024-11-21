import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from tqdm import tqdm
import os
import sys

def create_output_directories(base_dir, save_failures):
    success_frame_dir = f"{base_dir}"
    os.makedirs(success_frame_dir, exist_ok=True)

    if save_failures:
        failure_frame_dir = f"{base_dir}/failure"
        os.makedirs(failure_frame_dir, exist_ok=True)

def update_dict(pkl_file, new_data):
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        data.update(new_data)
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
    except FileNotFoundError:
        with open(pkl_file, 'wb') as f:
            pickle.dump(new_data, f)

def annotate_and_save_video(video_path, output_dir, model_path, frames_per_minute, hand_label, dump_landmarks=False, save_failures=False, start_time=0, end_time=None):
    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}\n")
        with open("error_log.txt", 'a') as f:
            f.write(f"Error: Cannot open video {video_path}\n")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate start and end frames based on provided times
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else total_frames

    # Validate start and end frames
    if start_frame >= total_frames or start_frame < 0:
        print("Error: Start time is out of video duration range.")
        return
    if end_frame > total_frames or end_frame <= start_frame:
        print("Error: End time is out of video duration range or before start time.")
        return

    # Frame interval to achieve the desired frames per minute
    frame_interval = int(fps * 60 / frames_per_minute)

    create_output_directories(output_dir, save_failures)

    # Process each frame with a progress bar
    frame_index = 0
    landmarks_success = dict()
    landmarks_failure = dict()
    for frame_count in tqdm(range(start_frame, end_frame, frame_interval), desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Cannot read frame at position {frame_count}")
            continue

        # Convert the frame to the format required by MediaPipe
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect hand landmarks
        detection_result = detector.detect(image)

        # Annotate the image with landmarks
        if detection_result.hand_landmarks:
            hand = detection_result.handedness[0][0].category_name
            if (hand_label == "L" and hand == "Left") or (hand_label == "R" and hand == "Right"):
                landmarks_success[frame_index] = detection_result
                frame_path = f"{output_dir}/{frame_index:04d}.png"
                cv2.imwrite(frame_path, frame)
        else:
            if save_failures:
                frame_path = f"{output_dir}/failure/{frame_index:04d}.png"
                cv2.imwrite(frame_path, frame)
            landmarks_failure[frame_index] = 'error'
        frame_index += frame_interval
    cap.release()

    if dump_landmarks:
        landmarks_success_path = f'{output_dir}/landmarks_success.pkl'
        update_dict(landmarks_success_path, landmarks_success)
        landmarks_failure_path = f'{output_dir}/landmarks_failure.pkl'
        update_dict(landmarks_failure_path, landmarks_failure)
