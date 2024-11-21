import os
import pickle
import numpy as np
import cv2

def load_landmarks(pkl_file, frame_index):
    required_landmarks = [0, 1, 2, 3, 5, 9, 13, 17]
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        if frame_index in data:
            return np.array([[data[frame_index].hand_world_landmarks[0][lm].x,
                              data[frame_index].hand_world_landmarks[0][lm].y,
                              data[frame_index].hand_world_landmarks[0][lm].z] 
                              for lm in required_landmarks])
    except Exception as e:
        # print(f"Error loading landmarks from {pkl_file}: {e}")
        return None
    return None

def calculate_transformation_matrix(ground_truth, target):
    retval, matrix, inliers = cv2.estimateAffine3D(ground_truth, target)
    if retval:
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
        return matrix
    else:
        print("Transformation estimation failed.")
        return None

def decompose_matrix(matrix):
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    try:
        pitch, yaw, roll = cv2.RQDecomp3x3(rotation_matrix)[0]
        return pitch, yaw, roll, translation
    except Exception as e:
        print(f"Error decomposing matrix: {e}")
        return None, None, None, None

def get_first_png_per_user(frame_paths):
    user_first_png = {}
    for path in frame_paths:
        user_dir = os.path.dirname(path)
        if user_dir not in user_first_png:
            user_first_png[user_dir] = path
    return user_first_png

def get_rotates(frame_index, profile_index, pkl_file):
    target_landmarks = load_landmarks(pkl_file, frame_index)
    ground_truth = load_landmarks(pkl_file, profile_index)

    if target_landmarks is None:
        # print(f"Skipping as target landmarks not found.")
        return None

    transformation_matrix = calculate_transformation_matrix(ground_truth, target_landmarks)
    if transformation_matrix is not None:
        pitch, yaw, roll, translation = decompose_matrix(transformation_matrix)
        return pitch, yaw, roll, translation
    
    return None

def process_rigid_deformations(frames_file, output_all_file, pitch_file, yaw_file, roll_file, translation_file):
    with open(frames_file, 'r') as f:
        frame_paths = [line.strip() for line in f if line.strip()]

    # Identify the first PNG for each user
    user_first_png = get_first_png_per_user(frame_paths)
    results_all = []
    results_pitch = []
    results_yaw = []
    results_roll = []
    results_translation = []
    
    for frame_path in frame_paths:
        user_dir = os.path.dirname(frame_path)
        
        # Get the baseline path for the current user
        baseline_path = user_first_png.get(user_dir)
        if not baseline_path:
            continue
        
        # Load baseline landmarks from the corresponding `landmarks_success.pkl`
        pkl_file = os.path.join(user_dir, 'landmarks_success.pkl')
        baseline_index = int(os.path.basename(baseline_path).split('.')[0])
        ground_truth = load_landmarks(pkl_file, baseline_index)
        if ground_truth is None:
            continue
        
        # Extract current frame's landmarks
        frame_index = int(os.path.basename(frame_path).split('.')[0])
        target_landmarks = load_landmarks(pkl_file, frame_index)
        
        if target_landmarks is None:
            continue

        # Calculate transformation matrix
        transformation_matrix = calculate_transformation_matrix(ground_truth, target_landmarks)
        if transformation_matrix is not None:
            pitch, yaw, roll, translation = decompose_matrix(transformation_matrix)
            if pitch is not None:
                # Append all results to respective lists
                results_all.append(f"{frame_path}: Pitch = {pitch}, Yaw = {yaw}, Roll = {roll}, Translation = {translation}")
                results_pitch.append(f"{frame_path}: Pitch = {pitch}")
                results_yaw.append(f"{frame_path}: Yaw = {yaw}")
                results_roll.append(f"{frame_path}: Roll = {roll}")
                results_translation.append(f"{frame_path}: Translation = {translation}")
            else:
                print(f"Decomposition failed for {frame_path}")
    
    # with open(output_all_file, 'w') as out_all_file:
    #     for result in results_all:
    #         out_all_file.write(result + "\n")
    
    # with open(pitch_file, 'w') as out_pitch_file:
    #     for result in results_pitch:
    #         out_pitch_file.write(result + "\n")

    # with open(yaw_file, 'w') as out_yaw_file:
    #     for result in results_yaw:
    #         out_yaw_file.write(result + "\n")

    # with open(roll_file, 'w') as out_roll_file:
    #     for result in results_roll:
    #         out_roll_file.write(result + "\n")

    # with open(translation_file, 'w') as out_translation_file:
    #     for result in results_translation:
    #         out_translation_file.write(result + "\n")
    
    # print(f"Rigid deformation results saved to {output_all_file}")
    # print(f"Pitch results saved to {pitch_file}")
    # print(f"Yaw results saved to {yaw_file}")
    # print(f"Roll results saved to {roll_file}")
    # print(f"Translation results saved to {translation_file}")
