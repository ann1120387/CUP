import os
import random
import shutil
import pickle
import re
import sys

sys.path.append(r"../preprocessing")

from non_rigid import calculate_frame_bending, calculate_frame_stretching
from homo_roi import get_profile_images
from transformation_m import get_rotates

def get_image_paths(user_dir, hand, wavelengths, conditions):
    image_paths = []
    profile_index = "0000"
    for wavelength in wavelengths:
        for condition in conditions:
            target_dir = rf"{user_dir}\{wavelength}\{hand}\{condition}"
            try:
                (profile_index, _) = get_profile_images(target_dir)
                # print(profile_index)
                for img in os.listdir(target_dir):
                    if img.endswith('.png'):
                        img_path = os.path.join(target_dir, img)
                        image_paths.append(img_path)
            except Exception as err:
                continue
    return image_paths, profile_index

def split_list(train_ratio, val_ratio, test_ratio, data):
    n = len(data)
    size1 = int(n * train_ratio)
    size2 = int(n * val_ratio)
    size3 = int(n * test_ratio)

    # Randomly shuffle the data
    random.shuffle(data)

    # Split the list
    list1 = data[:size1]
    list2 = data[size1:size1 + size2]
    list3 = data[size1 + size2:size1 + size2 + size3]

    return list1, list2, list3

def check_standard(pkl_file, frame_index, profile_index):
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        landmarks = data[int(frame_index)].hand_world_landmarks[0]
        angle1, angle2, angle3, angle4, total_angle = calculate_frame_stretching(landmarks)
        
        bend_indices = [0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        cur_bend = calculate_frame_bending(int(frame_index), bend_indices, pkl_file)
        profile_bend = calculate_frame_bending(int(profile_index), bend_indices, pkl_file)
        bend = cur_bend-profile_bend

        if total_angle > 25 or total_angle < 15:
            return False
        
        if bend < -3 or bend > 3:
            return False
        return True
    
    except Exception as err:
        return False

def check_not_standard(pkl_file, frame_index, profile_index, pitch, yaw, roll):
    try:
        if any(angle < -35 or angle > 35 for angle in (pitch, yaw, roll)):
            return True
    
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        landmarks = data[int(frame_index)].hand_world_landmarks[0]
        angle1, angle2, angle3, angle4, total_angle = calculate_frame_stretching(landmarks)
        
        bend_indices = [0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        cur_bend = calculate_frame_bending(int(frame_index), bend_indices, pkl_file)
        profile_bend = calculate_frame_bending(int(profile_index), bend_indices, pkl_file)
        bend = cur_bend-profile_bend

        if total_angle > 60:
            return True 
        if bend > -20:
            return True
        return False
    except Exception as err:
        return False

def large_angles(pitch, yaw, roll, translation):
    if any(angle < -60 or angle > 60 for angle in (pitch, yaw, roll)):
        return True
    return False
    
def prep_dataset(users, src_roots, dst_root, hands, wavelengths, conditions, train_ratio, val_ratio, test_ratio, num_images, standard):
    os.makedirs(rf"{dst_root}\train", exist_ok=True)
    os.makedirs(rf"{dst_root}\val", exist_ok=True)
    os.makedirs(rf"{dst_root}\test", exist_ok=True)
    for user in users:
        for hand in hands:
            # treat each hand of a user as a class, 
            # since the left hand and right hand are reported to be unrelated
            train_dir = rf"{dst_root}\train\{user}_{hand}"
            val_dir = rf"{dst_root}\val\{user}_{hand}"
            test_dir = rf"{dst_root}\test\{user}_{hand}"
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            image_paths = []

            for src_root in src_roots:
                user_dir = rf"{src_root}\{user}"
                tmp_paths, profile_index = get_image_paths(user_dir, hand, wavelengths, conditions)
                # print(len(tmp_paths))
                for tmp_path in tmp_paths:
                    image_index = os.path.splitext(os.path.basename(tmp_path))[0]
                    path_until_lowest_folder = os.path.dirname(tmp_path)
                    path_until_lowest_folder = path_until_lowest_folder.replace("_ROI", "")
                    path_until_lowest_folder = path_until_lowest_folder.replace(r"\10", "")
                    pkl_path = rf"{path_until_lowest_folder}\landmarks_success.pkl"
                    try:
                        pitch, yaw, roll, translation = get_rotates(int(image_index), int(profile_index), pkl_path)
                    except Exception as err:
                        continue
                    image_paths.append(tmp_path)
                    if large_angles(pitch, yaw, roll, translation):
                        # if the rotation angles are extreme, skip those frames
                        continue
                    if standard and check_standard(pkl_path, image_index, profile_index):
                        image_paths.append(tmp_path)
                    if not standard and check_not_standard(pkl_path, image_index, profile_index, pitch, yaw, roll):
                        image_paths.append(tmp_path)
            
            print(f"{user}_{hand}", len(image_paths))

            random.shuffle(image_paths)
            image_paths = image_paths[:num_images]
            # print(image_paths)
            train_images, val_images, test_images = split_list(train_ratio, val_ratio, test_ratio, image_paths)

            print(len(train_images), len(val_images), len(test_images))
            for image_path in train_images:
                shutil.copy(image_path, train_dir)
            for image_path in val_images:
                shutil.copy(image_path, val_dir)
            for image_path in test_images:
                shutil.copy(image_path, test_dir)

def combine_dirs(source_dirs, target_dir, subdirs):
    for source_dir in source_dirs:
        for subdir in subdirs:
            src_subdir_path = os.path.join(source_dir, subdir)
            dest_subdir_path = os.path.join(target_dir, subdir)
            os.makedirs(dest_subdir_path, exist_ok=True)

            for filename in os.listdir(src_subdir_path):
                src_file_path = os.path.join(src_subdir_path, filename)
                dest_file_path = os.path.join(dest_subdir_path, filename)

                if os.path.isdir(src_file_path):
                    continue

                # Rename if necessary to avoid overwriting files
                if os.path.exists(dest_file_path):
                    base, ext = os.path.splitext(filename)
                    dest_file_path = os.path.join(dest_subdir_path, f"{base}_{os.path.basename(source_dir)}{ext}")

                shutil.copy2(src_file_path, dest_file_path)
