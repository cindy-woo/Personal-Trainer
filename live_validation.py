import cv2
import numpy as np
import time
import mediapipe as mp
import os

# Import your custom math functions
from validate_poses import procrustes_error

# --- Configuration ---
CAMERA_INDEX = 0
STABLE_SECONDS = 5
MOTION_THRESHOLD = 25
MOTION_PERCENT = 2.0
GT_DIR = "Ground Truth" 

# Define the 12 core structural joints
CORE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_motion(frame1, frame2):
    """Calculates pixel difference to detect movement."""
    gray1 = cv2.GaussianBlur(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (21,21), 0)
    gray2 = cv2.GaussianBlur(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (21,21), 0)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    pct = 100 * np.count_nonzero(thresh) / thresh.size
    return pct > MOTION_PERCENT

def draw_core_skeleton(image, results):
    """Draws only the 12 core joints and their connecting lines."""
    if not results or not results.pose_landmarks:
        return
        
    h, w = image.shape[:2]
    
    # Skeletal connections specifically for the 12 core joints
    CONNECTIONS = [
        (11, 12), # Shoulders
        (11, 13), (13, 15), # Left Arm
        (12, 14), (14, 16), # Right Arm
        (11, 23), (12, 24), # Torso
        (23, 24), # Hips
        (23, 25), (25, 27), # Left Leg
        (24, 26), (26, 28)  # Right Leg
    ]
    
    # Draw green lines
    for start_idx, end_idx in CONNECTIONS:
        start_lm = results.pose_landmarks.landmark[start_idx]
        end_lm = results.pose_landmarks.landmark[end_idx]
        
        pt1 = (int(start_lm.x * w), int(start_lm.y * h))
        pt2 = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
            
    # Draw red joint circles
    for idx in CORE_JOINTS:
        lm = results.pose_landmarks.landmark[idx]
        pt = (int(lm.x * w), int(lm.y * h))
        cv2.circle(image, pt, 6, (0, 0, 255), -1)

def extract_and_normalize_frame(frame):
    """Extracts joints and returns the matrices AND the raw MediaPipe results for drawing."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if not results.pose_landmarks:
        return None, None, None
        
    landmarks_2d, landmarks_3d = [], []
    for idx in CORE_JOINTS:
        lm = results.pose_landmarks.landmark[idx]
        landmarks_2d.append([lm.x, lm.y])
        landmarks_3d.append([lm.x, lm.y, lm.z])
        
    A_2d, A_3d = np.array(landmarks_2d), np.array(landmarks_3d)
    
    # Normalize 2D
    A_centered_2d = A_2d - np.mean(A_2d, axis=0)
    scale_2d = np.linalg.norm(A_centered_2d, 'fro')
    A_norm_2d = A_centered_2d / scale_2d if scale_2d != 0 else A_centered_2d
    
    # Normalize 3D
    A_centered_3d = A_3d - np.mean(A_3d, axis=0)
    scale_3d = np.linalg.norm(A_centered_3d, 'fro')
    A_norm_3d = A_centered_3d / scale_3d if scale_3d != 0 else A_centered_3d
    
    return A_norm_2d, A_norm_3d, results

def main():
    # 1. Terminal UI for Pose Selection
    print("\n" + "="*50)
    print("🧘 YOGA POSE ALIGNMENT SYSTEM")
    print("="*50)
    pose_configs = {
        "1": {
            "name": "Tree Pose", 
            "gt_3d": "tree_pose_gt_3d.npy",
            "ref_img": "instructor_tree.jpg" 
        },
        "2": {
            "name": "One-Legged King Pigeon Pose", 
            "gt_3d": "olkp_pose_gt_3d.npy",
            "ref_img": "instructor_olkp.jpg" 
        }
    }
    
    for key, val in pose_configs.items():
        print(f"[{key}] {val['name']}")
        
    choice = input("\nEnter the number of the pose you want to perform: ")
    if choice not in pose_configs:
        print("Invalid choice. Exiting.")
        return
        
    target_pose = pose_configs[choice]["name"]
    gt_file = pose_configs[choice]["gt_3d"]
    ref_img_name = pose_configs[choice]["ref_img"]
    
    gt_file_path = os.path.join(GT_DIR, target_pose, gt_file)
    ref_img_path = os.path.join(GT_DIR, target_pose, ref_img_name)
    
    # 2. Load Ground Truth Matrix
    try:
        gt_3d_matrix = np.load(gt_file_path) 
        print(f"\n✅ Successfully loaded Ground Truth for {target_pose}")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find ground truth file '{gt_file_path}'.")
        return

    # 3. Start Webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): 
        print("Error: Could not open webcam.")
        return
        
    # Read a single frame to get the webcam height
    ret, initial_frame = cap.read()
    cam_h, cam_w = initial_frame.shape[:2]

    # 4. Load and Resize Reference Image
    ref_image = cv2.imread(ref_img_path)
    if ref_image is not None:
        ref_h, ref_w = ref_image.shape[:2]
        aspect_ratio = ref_w / ref_h
        new_w = int(cam_h * aspect_ratio)
        ref_image = cv2.resize(ref_image, (new_w, cam_h))
    else:
        # Fallback if image doesn't exist yet
        print(f"⚠️ Warning: Reference image '{ref_img_name}' not found. Using placeholder.")
        ref_image = np.zeros((cam_h, int(cam_w * 0.75), 3), dtype=np.uint8)
        cv2.putText(ref_image, "Reference Image", (50, cam_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    print(f"\nStarting webcam... Get into {target_pose}!")
    
    prev_frame = None
    stable_since = None
    captured = False
    result_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        h, w = frame.shape[:2]
        now = time.time()

        # --- 1. DETERMINE WHAT TO DISPLAY ---
        if result_frame is not None:
            # We are in the "Frozen" Evaluation State
            display = result_frame.copy()
            cv2.rectangle(display, (0, h-40), (w, h), (0,0,0), -1)
            cv2.putText(display, "PRESS SPACE to try again, or Q to quit", (10, h-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Check for reset key
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                result_frame = None
                captured = False
                stable_since = None
                continue
        else:
            # We are in the "Live" Tracking State
            if prev_frame is not None:
                is_moving = detect_motion(prev_frame, frame)
                if is_moving: stable_since = None
                elif stable_since is None: stable_since = now
            prev_frame = frame.copy()

            display = frame.copy()

            if stable_since is not None:
                elapsed = now - stable_since
                remaining = max(0, STABLE_SECONDS - elapsed)
                bar_w = min(int((elapsed/STABLE_SECONDS)*(w-40)), w-40)
                cv2.rectangle(display, (20, h-40), (w-20, h-20), (60, 60, 60), -1)
                cv2.rectangle(display, (20, h-40), (20+bar_w, h-20), (0, 220, 0), -1)
                cv2.putText(display, f"Hold still! Analyzing in {remaining:.1f}s...", 
                            (25, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # THE TRIGGER: 5 seconds reached
                if elapsed >= STABLE_SECONDS and not captured:
                    captured = True
                    print(f"\n📸 Snapshot taken! Calculating Jacobi SVD...")
                    
                    test_norm_2d, test_norm_3d, mp_results = extract_and_normalize_frame(frame)
                    
                    if test_norm_3d is not None:
                        result_frame = frame.copy() 
                        
                        np_err, jacobi_err = procrustes_error(test_norm_3d, gt_3d_matrix)
                        
                        print("-" * 40)
                        print(f"POSE: {target_pose}")
                        print(f"JACOBI SVD ERROR: {jacobi_err:.4f}")
                        print(f"Analytical (NumPy) Error: {np_err:.4f}")
                        print("-" * 40)
                        
                        # Draw joints and display score on the frozen frame
                        draw_core_skeleton(result_frame, mp_results)
                        cv2.rectangle(result_frame, (5, 5), (350, 70), (0,0,0), -1)
                        cv2.putText(result_frame, f"Error: {jacobi_err:.4f}", (15, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if jacobi_err > 0.4 else (0, 255, 0), 3)
                    else:
                        print("❌ No body detected in snapshot. Try again.")
                        captured = False
                        stable_since = None
            else:
                cv2.putText(display, "Move into position and hold still", (10, h-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

        # --- 2. THE SCALING ENGINE ---
        SCALE_FACTOR = 2.5 # Increase this to make the window larger
        
        # Scale the active display (Live or Frozen)
        display_scaled = cv2.resize(display, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        new_h, new_w = display_scaled.shape[:2]
        
        # Scale the reference image to match the NEW height
        ref_h_orig, ref_w_orig = ref_image.shape[:2]
        aspect_ratio = ref_w_orig / ref_h_orig
        ref_scaled = cv2.resize(ref_image, (int(new_h * aspect_ratio), new_h))
        
        # Stitch them together
        combined_display = cv2.hconcat([display_scaled, ref_scaled])
        
        # --- 3. SHOW THE WINDOW ---
        # Allow the window to be resized by the OS without snapping back
        cv2.namedWindow("Yoga Alignment Tracker", cv2.WINDOW_NORMAL)
        cv2.imshow("Yoga Alignment Tracker", combined_display)
        
        # Check for quit key in the live loop as well
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()