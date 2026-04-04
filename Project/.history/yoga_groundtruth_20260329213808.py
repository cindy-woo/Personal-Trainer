import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose for static images
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_normalized_pose_matrices(image_path):
    """Extracts, centers, and scales both 2D and 3D poses from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  -> Error: Could not load image at {image_path}")
        return None, None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print(f"  -> Error: No pose detected in {image_path}")
        return None, None
        
    # Extract landmarks for both 2D and 3D
    landmarks_2d = []
    landmarks_3d = []
    for landmark in results.pose_landmarks.landmark:
        landmarks_2d.append([landmark.x, landmark.y])
        landmarks_3d.append([landmark.x, landmark.y, landmark.z])
    
    A_2d = np.array(landmarks_2d)
    A_3d = np.array(landmarks_3d)
    
    # --- 2D Normalization ---
    centroid_2d = np.mean(A_2d, axis=0)
    A_centered_2d = A_2d - centroid_2d
    scale_2d = np.linalg.norm(A_centered_2d, 'fro')
    A_normalized_2d = A_centered_2d / scale_2d if scale_2d != 0 else A_centered_2d
    
    # --- 3D Normalization ---
    centroid_3d = np.mean(A_3d, axis=0)
    A_centered_3d = A_3d - centroid_3d
    scale_3d = np.linalg.norm(A_centered_3d, 'fro')
    A_normalized_3d = A_centered_3d / scale_3d if scale_3d != 0 else A_centered_3d
    
    return A_normalized_2d, A_normalized_3d

def calculate_average_ground_truth(image_paths):
    """Calculates the average normalized matrices from a list of images."""
    matrices_2d = []
    matrices_3d = []
    
    for path in image_paths:
        print(f"  Processing: {os.path.basename(path)}")
        mat_2d, mat_3d = get_normalized_pose_matrices(path)
        
        if mat_2d is not None and mat_3d is not None:
            matrices_2d.append(mat_2d)
            matrices_3d.append(mat_3d)
            
    if not matrices_2d or not matrices_3d:
        return None, None
        
    # Average the normalized matrices together
    average_matrix_2d = np.mean(np.array(matrices_2d), axis=0)
    average_matrix_3d = np.mean(np.array(matrices_3d), axis=0)
    
    return average_matrix_2d, average_matrix_3d

if __name__ == "__main__":
    # Define the base directory and the specific poses
    base_dir = "Yoga Poses"
    
    # Dictionary mapping the pose name to its folder and file prefix
    pose_configs = {
        "Tree Pose": {
            "folder": "tree_pose",
            "prefix": "tree_pose_"
        },
        "One-Legged King Pigeon Pose": {
            "folder": "olkp_pose",
            "prefix": "olkp_pose_"
        }
    }
    
    # Loop through each configuration and process the images
    for pose_name, config in pose_configs.items():
        print(f"\n--- Extracting Ground Truth for {pose_name} ---")
        
        image_paths = []
        # Construct paths for images 1, 2, and 3
        for i in range(1, 4):
            base_name = f"{config['prefix']}{i}"
            
            # Check for both .jpg and .jpeg
            path_jpg = os.path.join(base_dir, config['folder'], f"{base_name}.jpg")
            path_jpeg = os.path.join(base_dir, config['folder'], f"{base_name}.jpeg")
            
            if os.path.exists(path_jpg):
                image_paths.append(path_jpg)
            elif os.path.exists(path_jpeg):
                image_paths.append(path_jpeg)
            else:
                print(f"  -> Missing File: Could not find {base_name}.jpg or .jpeg")
            
        # Only calculate if we found all 3 images
        if len(image_paths) > 0:
            gt_2d, gt_3d = calculate_average_ground_truth(image_paths)
            
            if gt_2d is not None and gt_3d is not None:
                # Save the 2D matrix
                save_filename_2d = f"{config['folder']}_gt_2d.npy"
                np.save(save_filename_2d, gt_2d)
                
                # Save the 3D matrix
                save_filename_3d = f"{config['folder']}_gt_3d.npy"
                np.save(save_filename_3d, gt_3d)
                
                print(f"Success! Saved 2D matrix to {save_filename_2d} (Shape: {gt_2d.shape})")
                print(f"Success! Saved 3D matrix to {save_filename_3d} (Shape: {gt_3d.shape})")
        else:
            print(f"Failed to generate ground truth for {pose_name}. No images found.")