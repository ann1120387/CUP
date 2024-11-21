import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_angle(A, B, C):
    AB = A - B
    BC = C - B
    cosine_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def load_bend_landmarks(pkl_file, frame_index, landmark_indices):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    if frame_index in data:
        return {idx: np.array([data[frame_index].hand_world_landmarks[0][idx].x,
                               data[frame_index].hand_world_landmarks[0][idx].y,
                               data[frame_index].hand_world_landmarks[0][idx].z])
                for idx in landmark_indices}
    else:
        return None

def calculate_finger_angles(landmarks, finger_joints):
    angles = {}
    for finger, joints in finger_joints.items():
        A, B, C = landmarks[joints[0]], landmarks[joints[1]], landmarks[joints[2]]
        angle = calculate_angle(A, B, C)
        angles[finger] = angle
    return angles

def visualize_bending_angles(image_path, angles, landmarks, finger_joints):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Convert BGR image (OpenCV format) to RGB format for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    
    # Mark the joints and display the angles
    for finger, joints in finger_joints.items():
        # Scale the normalized coordinates by image dimensions
        A = landmarks[joints[0]] * [width, height, 1]
        B = landmarks[joints[1]] * [width, height, 1]
        C = landmarks[joints[2]] * [width, height, 1]
        
        # Plot joints in scaled coordinates
        plt.plot([A[0], B[0]], [A[1], B[1]], 'ro-')
        plt.plot([B[0], C[0]], [B[1], C[1]], 'ro-')
        
        # Display angle in scaled coordinates
        angle = angles[finger]
        plt.text(B[0], B[1], f'{finger}: {angle:.1f}°', color='yellow', fontsize=12, ha='center')

    plt.axis('off')
    plt.show()

def calculate_frame_bending(frame_index, landmark_indices, pkl_file, image_path=None):
    finger_joints1 = {
        'index': [5, 6, 7],
        'middle': [9, 10, 11],
        'ring': [13, 14, 15],
        'pinky': [17, 18, 19]
    }

    finger_joints2 = {
        'index': [0, 5, 6],
        'middle': [0, 9, 10],
        'ring': [0, 13, 14],
        'pinky': [0, 17, 18]
    }

    landmarks = load_bend_landmarks(pkl_file, frame_index, landmark_indices)
                    
    if landmarks is not None:
        # Calculate angles for both sets of finger joints
        angles1 = calculate_finger_angles(landmarks, finger_joints1)
        angles2 = calculate_finger_angles(landmarks, finger_joints2)

        if image_path:
            visualize_bending_angles(image_path, angles1, landmarks, finger_joints1)
        
        sum_angles1 = sum(angles1.values())
        real_sum = sum_angles1

    return real_sum
                        
def calculate_angle_between_lines(p1, p2, p3, p4, reference_line1=None, reference_line2=None, check_proximity=False):
    line1 = np.array([p1.x - p2.x, p1.y - p2.y])
    line2 = np.array([p3.x - p4.x, p3.y - p4.y])
    
    cross_product = np.cross(line1, line2)
    if np.linalg.norm(cross_product) == 0:  # Parallel lines, no intersection
        return 0
    
    dot_product = np.dot(line1, line2)
    magnitude1 = np.linalg.norm(line1)
    magnitude2 = np.linalg.norm(line2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    if check_proximity:
        intersection_point = find_intersection(p1, p2, p3, p4)
        
        if intersection_point is not None:
            dist_to_ref1 = distance_to_line(intersection_point, reference_line1[0], reference_line1[1])
            dist_to_ref2 = distance_to_line(intersection_point, reference_line2[0], reference_line2[1])
            
            if dist_to_ref1 >= dist_to_ref2:
                return 0  
    
    return angle

def find_intersection(p1, p2, p3, p4):
    A = np.array([p1.x, p1.y])
    B = np.array([p2.x, p2.y])
    C = np.array([p3.x, p3.y])
    D = np.array([p4.x, p4.y])

    AB = B - A
    CD = D - C

    if np.cross(AB, CD).all() == 0:
        return None

    t = np.dot(np.cross(C - A, CD), np.cross(AB, CD)) / np.linalg.norm(np.cross(AB, CD)) ** 2
    intersection_point = A + t * AB

    return intersection_point

def distance_to_line(point, line_point1, line_point2):
    P = np.array([point[0], point[1]])
    A = np.array([line_point1.x, line_point1.y])
    B = np.array([line_point2.x, line_point2.y])

    AP = P - A
    AB = B - A

    distance = np.linalg.norm(np.cross(AP, AB)) / np.linalg.norm(AB)
    return distance

def calculate_frame_stretching(landmarks):
    try:
        # Step 1: Calculate angle without proximity checks
        angle1 = calculate_angle_between_lines(landmarks[2], landmarks[1], landmarks[5], landmarks[1], check_proximity=False)
        
        # Step 2: Check proximity with lines 5-9 and 6-10
        angle2 = calculate_angle_between_lines(
            landmarks[6], landmarks[5], landmarks[10], landmarks[9],
            reference_line1=(landmarks[5], landmarks[9]), reference_line2=(landmarks[6], landmarks[10]), check_proximity=True)
        
        # Step 3: Check proximity with lines 9-13 and 10-14
        angle3 = calculate_angle_between_lines(
            landmarks[10], landmarks[9], landmarks[14], landmarks[13],
            reference_line1=(landmarks[9], landmarks[13]), reference_line2=(landmarks[10], landmarks[14]), check_proximity=True)
        
        # Step 4: Check proximity with lines 13-17 and 14-18
        angle4 = calculate_angle_between_lines(
            landmarks[14], landmarks[13], landmarks[18], landmarks[17],
            reference_line1=(landmarks[13], landmarks[17]), reference_line2=(landmarks[14], landmarks[18]), check_proximity=True)
        
        # Sum of the four angles
        total_angle = angle1 + angle2 + angle3 + angle4
        # print(f"Angle 1: {angle1}")
        # print(f"Angle 2: {angle2}")
        # print(f"Angle 3: {angle3}")
        # print(f"Angle 4: {angle4}")
        # print(f"Sum of Angles: {total_angle}")
        
        return angle1, angle2, angle3, angle4, total_angle
    except KeyError as e:
        return 0, 0, 0, 0, 0

def visualize_landmarks_and_angles(image_path, pkl_file):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height_1, width_1, _ = image_rgb.shape

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    landmarks = data[2929].hand_landmarks[0]
    angle1, angle2, angle3, angle4, total_angle = calculate_frame_stretching(landmarks)

    # Set up plot
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    
    # Define line segments and calculate extensions
    line_segments = [
        (landmarks[2], landmarks[1]),  # First angle line
        (landmarks[5], landmarks[1]),
        (landmarks[6], landmarks[5]),  # Second angle line
        (landmarks[10], landmarks[9]),
        (landmarks[10], landmarks[9]), # Third angle line
        (landmarks[14], landmarks[13]),
        (landmarks[14], landmarks[13]), # Fourth angle line
        (landmarks[18], landmarks[17])
    ]

    for idx, landmark in enumerate(landmarks):
        if idx in [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]:
            x_pixel = landmark.x * width_1
            y_pixel = landmark.y * height_1
            plt.scatter(x_pixel, y_pixel, color='red')
            plt.text(x_pixel, y_pixel, str(idx), color="yellow", fontsize=12)

    for (p1, p2) in line_segments:
        plt.plot([p1.x * width_1, p2.x * width_1], [p1.y * height_1, p2.y * height_1], color='blue', linewidth=2)

    return angle1, angle2, angle3, angle4, total_angle
