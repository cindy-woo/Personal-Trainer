import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose for static images
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_normalized_pose_matrix(image_path):
    """Extracts, centers, and scales the pose from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print(f"No pose detected in {image_path}")
        return None
        
    # 1. Extract the 33 landmarks into a 33x2 matrix (using x and y)
    # You can easily add landmark.z here if you pivot to 3D SVD
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y])
    
    A = np.array(landmarks)
    
    # 2. Translate: Center the pose at the origin (0,0)
    centroid = np.mean(A, axis=0)
    A_centered = A - centroid
    
    # 3. Scale: Normalize the size using the Frobenius norm
    scale = np.linalg.norm(A_centered, 'fro')
    A_normalized = A_centered / scale
    
    return A_normalized

def calculate_average_ground_truth(image_paths):
    """Calculates the average normalized matrix from a list of images."""
    matrices = []
    for path in image_paths:
        matrix = get_normalized_pose_matrix(path)
        if matrix is not None:
            matrices.append(matrix)
            
    if not matrices:
        return None
        
    # Average the normalized matrices together
    # This results in a mathematically stable "Grand Ground Truth"
    average_matrix = np.mean(np.array(matrices), axis=0)
    
    return average_matrix

if __name__ == "__main__":
    base_dir = "Yoga Poses"
    
    pose_configs = {
        "Tree Pose": {
            "folder": "tree_pose",
            "prefix": "tree_pose_"
        }
    }

# --- Execution Example ---
# List your 3 image paths for a specific pose (e.g., Downward Dog)
pose_images = ["pose1_img1.jpg", "pose1_img2.jpg", "pose1_img3.jpg"]

grand_ground_truth = calculate_average_ground_truth(pose_images)

if grand_ground_truth is not None:
    print("Successfully generated average Ground Truth Matrix:")
    print(f"Shape: {grand_ground_truth.shape}") # Should be (33, 2)