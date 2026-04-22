import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- Configuration ---
GT_DIR = "Yoga Poses" # Or wherever your files are saved
POSE_NAME = "triangle" # Change this to test different poses

# The 12 core structural joints (mapped to indices 0-11 in your matrix)
# 0: L Shoulder, 1: R Shoulder
# 2: L Elbow,    3: R Elbow
# 4: L Wrist,    5: R Wrist
# 6: L Hip,      7: R Hip
# 8: L Knee,     9: R Knee
# 10: L Ankle,   11: R Ankle

CONNECTIONS = [
    (0, 1),           # Shoulders
    (0, 2), (2, 4),   # Left Arm
    (1, 3), (3, 5),   # Right Arm
    (0, 6), (1, 7),   # Torso (Shoulders to Hips)
    (6, 7),           # Hips
    (6, 8), (8, 10),  # Left Leg
    (7, 9), (9, 11)   # Right Leg
]

def validate_pose(pose_name):
    npy_path = f"{pose_name}_gt.npy"
    json_path = f"{pose_name}_gt_angles.json"
    
    # 1. Verify and Load JSON Angles
    print(f"\n--- VALIDATING: {pose_name.upper()} ---")
    try:
        with open(json_path, 'r') as f:
            angles = json.load(f)
            print("\n📋 Averaged Biomechanical Angles:")
            for joint, angle in angles.items():
                print(f"  {joint}: {angle}°" if angle is not None else f"  {joint}: [HIDDEN]")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_path}")
        return

    # 2. Verify and Load Matrix
    try:
        matrix = np.load(npy_path)
        num_dims = matrix.shape[1]
        print(f"\n📏 Matrix Shape: {matrix.shape} (Detected as {num_dims}D Pose)")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {npy_path}")
        return

    is_2d = (num_dims == 2)

    # 3. Render the Skeleton
    fig = plt.figure(figsize=(8, 8))
    
    # --- 2D RENDERER ---
    if is_2d:
        ax = fig.add_subplot(111)
        
        # Extract X and Y (Invert Y because MediaPipe Y goes down)
        xs = matrix[:, 0]
        ys = -matrix[:, 1]
        
        # Plot the joints
        ax.scatter(xs, ys, c='red', s=50, label='Joints')

        # Draw the bones
        for start_idx, end_idx in CONNECTIONS:
            start_pt = matrix[start_idx]
            end_pt = matrix[end_idx]
            
            if np.all(start_pt == 0) or np.all(end_pt == 0):
                continue 
                
            ax.plot([xs[start_idx], xs[end_idx]], 
                    [ys[start_idx], ys[end_idx]], 'b-', linewidth=2)

        # Label the joints
        labels = ["LS", "RS", "LE", "RE", "LW", "RW", "LH", "RH", "LK", "RK", "LA", "RA"]
        for i, txt in enumerate(labels):
            if not np.all(matrix[i] == 0):
                # Add a slight offset to the text so it doesn't overlap the red dot
                ax.text(xs[i] + 0.02, ys[i], txt, size=8, zorder=1, color='k')

        # Formatting the plot
        ax.set_title(f"2D Ground Truth: {pose_name}")
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Up/Down)')
        ax.set_aspect('equal', adjustable='datalim') # Lock aspect ratio for accurate geometry

    # --- 3D RENDERER ---
    else:
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract X, Y, Z
        xs = matrix[:, 0]
        ys = matrix[:, 2] 
        zs = -matrix[:, 1] 

        # Plot the joints
        ax.scatter(xs, ys, zs, c='red', s=50, label='Joints')

        # Draw the bones
        for start_idx, end_idx in CONNECTIONS:
            start_pt = matrix[start_idx]
            end_pt = matrix[end_idx]
            
            if np.all(start_pt == 0) or np.all(end_pt == 0):
                continue 
                
            ax.plot([xs[start_idx], xs[end_idx]], 
                    [ys[start_idx], ys[end_idx]], 
                    [zs[start_idx], zs[end_idx]], 'b-', linewidth=2)

        # Label the joints
        labels = ["LS", "RS", "LE", "RE", "LW", "RW", "LH", "RH", "LK", "RK", "LA", "RA"]
        for i, txt in enumerate(labels):
            if not np.all(matrix[i] == 0):
                ax.text(xs[i], ys[i], zs[i], txt, size=8, zorder=1, color='k')

        # Formatting the plot
        ax.set_title(f"3D Ground Truth: {pose_name}")
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Up/Down)')
        
        # Set consistent aspect ratio
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print("\n✅ Launching Viewer! Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    # Change POSE_NAME at the top of the file to test 
    validate_pose(POSE_NAME)