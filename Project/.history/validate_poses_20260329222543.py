import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_normalized_pose_matrices(image_path):
    """Extracts, centers, and scales both 2D and 3D poses from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, None
        
    # Define the 12 core structural joints
    CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    landmarks_2d, landmarks_3d = [], []
    for :
        landmark in results.pose_landmarks.landmark
        landmarks_2d.append([landmark.x, landmark.y])
        landmarks_3d.append([landmark.x, landmark.y, landmark.z])
    
    A_2d, A_3d = np.array(landmarks_2d), np.array(landmarks_3d)
    
    # Normalize 2D
    A_centered_2d = A_2d - np.mean(A_2d, axis=0)
    scale_2d = np.linalg.norm(A_centered_2d, 'fro')
    A_normalized_2d = A_centered_2d / scale_2d if scale_2d != 0 else A_centered_2d
    
    # Normalize 3D
    A_centered_3d = A_3d - np.mean(A_3d, axis=0)
    scale_3d = np.linalg.norm(A_centered_3d, 'fro')
    A_normalized_3d = A_centered_3d / scale_3d if scale_3d != 0 else A_centered_3d
    
    return A_normalized_2d, A_normalized_3d

def procrustes_error(test_matrix, gt_matrix):
    """
    Calculates the optimal rotation to align test_matrix to gt_matrix using SVD,
    then returns the Frobenius norm of the difference (the error).
    """
    # 1. Calculate Covariance Matrix: C = Test^T * GT
    C = np.dot(test_matrix.T, gt_matrix)
    
    # 2. Perform SVD on the Covariance Matrix
    # NOTE: This is the built-in function you will replace with your custom solver later!
    U, S, Vt = np.linalg.svd(C)
    
    # 3. Calculate Optimal Rotation Matrix: R = U * Vt
    R = np.dot(U, Vt)
    
    # 4. Handle Reflection (Ensure it's a pure rotation, not a mirrored flip)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
        
    # 5. Rotate the test matrix to align with ground truth
    test_rotated = np.dot(test_matrix, R)
    
    # 6. Calculate Euclidean error (Frobenius norm)
    error = np.linalg.norm(gt_matrix - test_rotated, 'fro')
    return error

if __name__ == "__main__":
    test_dir = "Test Poses"
    
    pose_configs = {
        "Tree Pose": {
            "test_folder": "tree_pose",
            "gt_2d_file": "tree_pose_gt_2d.npy",
            "gt_3d_file": "tree_pose_gt_3d.npy"
        },
        "One-Legged King Pigeon Pose": {
            "test_folder": "olkp_pose",
            "gt_2d_file": "olkp_pose_gt_2d.npy",
            "gt_3d_file": "olkp_pose_gt_3d.npy"
        }
    }
    
    for pose_name, config in pose_configs.items():
        print(f"\n========================================")
        print(f"Validating: {pose_name}")
        
        # Load Ground Truths
        try:
            gt_2d = np.load(config["gt_2d_file"])
            gt_3d = np.load(config["gt_3d_file"])
        except FileNotFoundError:
            print(f"  -> Error: Could not find ground truth .npy files for {pose_name}.")
            continue
            
        # Get Test Images
        folder_path = os.path.join(test_dir, config["test_folder"])
        if not os.path.exists(folder_path):
            print(f"  -> Error: Could not find test folder {folder_path}")
            continue
            
        valid_extensions = ('.jpg', '.jpeg', '.png')
        test_images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        if not test_images:
            print(f"  -> No test images found in {folder_path}")
            continue
            
        for img_name in test_images:
            img_path = os.path.join(folder_path, img_name)
            test_2d, test_3d = get_normalized_pose_matrices(img_path)
            
            if test_2d is not None and test_3d is not None:
                error_2d = procrustes_error(test_2d, gt_2d)
                error_3d = procrustes_error(test_3d, gt_3d)
                
                print(f"\n  Image: {img_name}")
                print(f"    2D Alignment Error: {error_2d:.4f}")
                print(f"    3D Alignment Error: {error_3d:.4f}")
            else:
                print(f"\n  Image: {img_name} -> Failed to detect pose.")