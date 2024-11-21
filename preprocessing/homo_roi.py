import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from non_rigid import calculate_frame_bending, calculate_frame_stretching

def load_landmarks(pkl_file, frame_index, landmark_indices=None):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    if frame_index in data:
        hand = data[frame_index].handedness[0][0].category_name
        landmarks = data[frame_index].hand_landmarks[0]  # Assuming the first hand's landmarks
        if landmark_indices:
            extracted_landmarks = [landmarks[i] for i in landmark_indices]
        else:
            extracted_landmarks = landmarks
        src_pts = np.array([[lm.x, lm.y] for lm in extracted_landmarks], dtype=np.float32)
        return hand, src_pts
    else:
        return None

def combine_images_with_connections(image1, image2, src_pts, dst_pts):
    # Combine image1 and image2 horizontally
    combined_image = np.hstack((image1, image2))
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    # Offset for the second image (image2 starts after image1 in the combined image)
    offset_x = image1.shape[1]

    # Draw landmarks and connections between image1 and image2 with different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Colors for connections
    for i, (src_pt, dst_pt) in enumerate(zip(src_pts, dst_pts)):
        color = colors[i % len(colors)]  # Cycle through the colors
        # print(i, int(src_pt[0]), int(src_pt[1]))
        cv2.circle(combined_image_rgb, (int(src_pt[0]), int(src_pt[1])), 5, color, -1)
        cv2.circle(combined_image_rgb, (int(dst_pt[0]) + offset_x, int(dst_pt[1])), 5, color, -1)
        cv2.line(combined_image_rgb, (int(src_pt[0]), int(src_pt[1])), (int(dst_pt[0]) + offset_x, int(dst_pt[1])), color, 2)

    return combined_image_rgb

def visualize_combined_images(combined_image, combined_corrected_image):
    # Plot the combined images (original and corrected) in a 2x2 grid
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    axes[0].imshow(combined_image)
    axes[0].set_title("Original Images")
    axes[0].axis('off')

    axes[1].imshow(combined_corrected_image)
    axes[1].set_title("Corrected Images")
    axes[1].axis('off')

    plt.show()

def apply_homography_to_points(H, points):
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    # Apply the homography transformation
    transformed_points = H.dot(points_homogeneous.T).T
    # Convert back to Cartesian coordinates
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)
    return transformed_points[:, :2]

def dump_corrected_landmarks(pkl_file, frame_index, H, output_pkl_file, width_1, height_1, MCP_dist):
    hand, src_pts = load_landmarks(pkl_file, frame_index)
    pkl_extra_file = pkl_file.replace("landmarks", "meta")

    for i in range(len(src_pts)):
        src_pts[i] = (int(src_pts[i][0]*width_1), int(src_pts[i][1]*height_1))

    # Apply homography to all points
    if H is None:
        corrected_pts = src_pts
    else:
        corrected_pts = apply_homography_to_points(H, src_pts)
        # tmp = extract_pose_from_matrix(H)

    try:
        with open(output_pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as err:
        data = dict()

    try:
        with open(pkl_extra_file, 'rb') as f:
            H = pickle.load(f)
            MCP_dist = pickle.load(f)
            # translations = pickle.load(f)
    except Exception as err:
        H = dict()
        MCP_dist = dict()

    data[frame_index] = corrected_pts
    H[frame_index] = H
    MCP_dist[frame_index] = MCP_dist

    with open(output_pkl_file, 'wb') as f:
        pickle.dump(data, f)
    
    with open(pkl_extra_file, 'wb') as f:
        pickle.dump(H, f)
        pickle.dump(MCP_dist, f)

def get_homography_from_images(cur_path, profile_path, frame_index, profile_index, pkl_file, landmark_indices):
    hand1, src_pts = load_landmarks(pkl_file, int(frame_index), landmark_indices)
    hand2, dst_pts = load_landmarks(pkl_file, int(profile_index), landmark_indices)

    image1 = cv2.imread(cur_path)
    image2 = cv2.imread(profile_path)

    height_1, width_1, _ = image1.shape
    height_2, width_2, _ = image2.shape
    for i in range(len(src_pts)):
        src_pts[i] = (int(src_pts[i][0]*width_1), int(src_pts[i][1]*height_1))
    for i in range(len(dst_pts)):
        dst_pts[i] = (int(dst_pts[i][0]*width_2), int(dst_pts[i][1]*height_2))

    # calculate MCP_distance (directly decide the size of ROI)
    MCP_dist = np.linalg.norm(src_pts[3] - src_pts[2])
    # print(MCP_dist)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    landmarks = data[int(frame_index)].hand_world_landmarks[0]
    angle1, angle2, angle3, angle4, total_angle = calculate_frame_stretching(landmarks)
    
    bend_indices = [0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    cur_bend = calculate_frame_bending(int(frame_index), bend_indices, pkl_file)
    profile_bend = calculate_frame_bending(int(profile_index), bend_indices, pkl_file)
    bend = cur_bend-profile_bend

    if total_angle > 75:
        output_pkl_file = pkl_file.replace(".pkl", "_corrected.pkl")
        corrected_path = cur_path.replace(".png", "_corrected.png")
        cv2.imwrite(corrected_path, image1)
        dump_corrected_landmarks(pkl_file, int(frame_index), None, output_pkl_file, width_1, height_1, MCP_dist)
        return
    
    if bend < -20:
        output_pkl_file = pkl_file.replace(".pkl", "_corrected.pkl")
        corrected_path = cur_path.replace(".png", "_corrected.png")
        cv2.imwrite(corrected_path, image1)
        dump_corrected_landmarks(pkl_file, int(frame_index), None, output_pkl_file, width_1, height_1, MCP_dist)
        return

    # Calculate the homography matrix
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # print(H, status)
    # Apply homography to correct image2
    corrected_image1 = cv2.warpPerspective(image1, H, (width_1, height_1))

    corrected_path = cur_path.replace(".png", "_corrected.png")
    cv2.imwrite(corrected_path, corrected_image1)
    # Update src_pts using homography matrix H
    corrected_src_pts = apply_homography_to_points(H, src_pts)
    # Save the corrected landmark points
    output_pkl_file = pkl_file.replace(".pkl", "_corrected.pkl")
    dump_corrected_landmarks(pkl_file, int(frame_index), H, output_pkl_file, width_1, height_1, MCP_dist)


def calculate_square_roi(image_path, frame_index, corrected_pkl_path, output_dir, fraction, visualization, profile=False):
    frame = cv2.imread(image_path)

    height, width, _ = frame.shape
    frame_index = int(frame_index)
    frame_with_box = frame.copy()
    # print(height, width)

    with open(corrected_pkl_path, "rb") as file:
        corrected_points = pickle.load(file)

    x1, y1 = int(corrected_points[frame_index][5][0]), int(corrected_points[frame_index][5][1])  # Coordinates of first landmark (point 5)
    x2, y2 = int(corrected_points[frame_index][17][0]), int(corrected_points[frame_index][17][1])  # Coordinates of second landmark (point 17)


    # print((x1, y1), (x2, y2))
    cv2.circle(frame_with_box, (x1, y1), 5, (0, 255, 0), -1)  
    cv2.circle(frame_with_box, (x2, y2), 5, (0, 255, 0), -1)  

    # Calculate the fractional points
    p1 = (int(((fraction - 1) * x1 + x2) / fraction), int(((fraction - 1) * y1 + y2) / fraction))  # (fraction-1)/fraction point
    p2 = (int((x1 + (fraction - 1) * x2) / fraction), int((y1 + (fraction - 1) * y2) / fraction))  # fraction/(fraction-1) point
    
    # Mark the points on the frame and display labels
    cv2.circle(frame_with_box, p1, 5, (0, 0, 255), -1)  
    cv2.putText(frame_with_box, f"P1 (1/{fraction})", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.circle(frame_with_box, p2, 5, (0, 0, 255), -1)  
    cv2.putText(frame_with_box, f"P2 ({fraction-1}/{fraction})", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Calculate the side length of the square
    side_length = int(np.linalg.norm(np.array(p2) - np.array(p1)))
    
    # Calculate the perpendicular direction from p1p2 to get p3
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Calculate the direction of the perpendicular vector (90 degrees rotation)
    perp_dx = -dy
    perp_dy = dx
    
    # Ensure the perpendicular direction points leftward
    if perp_dx > 0:  # If the perpendicular vector points rightward, flip it to point leftward
        perp_dx = -perp_dx
        perp_dy = -perp_dy
    
    # Normalize the perpendicular vector and scale it to the length of the side
    norm = np.sqrt(perp_dx ** 2 + perp_dy ** 2)
    perp_dx = int(perp_dx / norm * side_length)
    perp_dy = int(perp_dy / norm * side_length)
    
    # Calculate the coordinates of p3 and p4
    p3 = (p2[0] + perp_dx, p2[1] + perp_dy)
    p4 = (p1[0] + perp_dx, p1[1] + perp_dy)
    # print(p3, p4)
    
    # Mark the square corners and display labels
    cv2.circle(frame_with_box, p3, 5, (255, 0, 0), -1)  # Blue circle at p3
    cv2.putText(frame_with_box, "P3", p3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.circle(frame_with_box, p4, 5, (255, 0, 0), -1)  # Blue circle at p4
    cv2.putText(frame_with_box, "P4", p4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Draw the square by connecting the points
    cv2.line(frame_with_box, p1, p2, (0, 255, 0), 2)
    cv2.line(frame_with_box, p2, p3, (0, 255, 0), 2)
    cv2.line(frame_with_box, p3, p4, (0, 255, 0), 2)
    cv2.line(frame_with_box, p4, p1, (0, 255, 0), 2)

    # crop and rotate the roi
    src_pts = np.array([p1, p2, p3, p4], dtype=np.float32)
    width = int(np.linalg.norm(np.array(p1) - np.array(p2)))
    height = int(np.linalg.norm(np.array(p2) - np.array(p3)))
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(frame, H, (width, height))

    # Save annotated frame and ROI frame
    os.makedirs(output_dir, exist_ok=True)
    roi_frame_path = os.path.join(output_dir, f"{frame_index:04d}.png")
    if warped_image.size != 0:  # Check if ROI is not empty
        cv2.imwrite(roi_frame_path, warped_image)
    if visualization:
        cv2.imshow('ROI marked on frame', frame_with_box)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_profile_images(directory):
    files = os.listdir(directory)
    png_files = [f for f in files if (f.endswith('.png') and "corrected" not in f)]
    png_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    profile_index = png_files[0].split(".")[0]

    return profile_index, os.path.join(directory, png_files[0]) if png_files else None
