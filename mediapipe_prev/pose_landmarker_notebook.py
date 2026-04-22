#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2023 The MediaPipe Authors. All Rights Reserved.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Pose Landmarks Detection with MediaPipe Tasks
# 
# This notebook shows you how to use MediaPipe Tasks Python API to detect pose landmarks from images.

# ## Preparation
# 
# Let's start with installing MediaPipe.
# 

# In[2]:


get_ipython().system('pip3 install -q mediapipe')


# Then download an off-the-shelf model bundle. Check out the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models) for more information about this model bundle.

# In[3]:


import urllib.request
import os

model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
model_path = 'pose_landmarker.task'

if not os.path.exists(model_path):
    print("Downloading pose_landmarker model...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Model downloaded successfully to {model_path}")
else:
    print(f"Model file already exists at {model_path}")


# ## Visualization utilities

# In[4]:


import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


# ## Download test image
# 
# To demonstrate the Pose Landmarker API, you can download a sample image using the follow code. The image is from [Pixabay](https://pixabay.com/photos/girl-woman-fitness-beautiful-smile-4051811/).

# In[5]:


import cv2
import matplotlib.pyplot as plt

def cv2_imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

IMAGE_FILE = "images/cobra1.jpg"
img = cv2.imread(IMAGE_FILE)
if img is not None:
    cv2_imshow(img)
else:
    print(f"Could not read {IMAGE_FILE}. Please make sure it's downloaded correctly.")


# ## Running inference and visualizing the results
# 
# The final step is to run pose landmark detection on your selected image. This involves creating your PoseLandmarker object, loading your image, running detection, and finally, the optional step of displaying the image with visualizations.
# 
# Check out the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python) to learn more about configuration options that this solution supports.
# 

# In[6]:


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 # Ensure cv2 is imported here too
import numpy as np

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

print("completed creating PoseLandmarker object")


# In[7]:


def output_joints_image(IMAGE_FILE, detector):
    # Read image with OpenCV (converts BGR to RGB)
    cv_image = cv2.imread(IMAGE_FILE)
    if cv_image is None:
        print(f"Error: Could not load image from {IMAGE_FILE}")
    else:
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Create MediaPipe Image from numpy array
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(image)
        
        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        return detection_result

# STEP 3: Load the input image using OpenCV
IMAGE_FILE_1 = "images/balancing_table/balancing_table1.jpg"
IMAGE_FILE_2 = "images/balancing_table/balancing_table2.jpg"
output_joints_image(IMAGE_FILE_1, detector)
output_joints_image(IMAGE_FILE_2, detector)


# Visualize the pose segmentation mask.

# In[8]:


# segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
# segmentation_mask = np.squeeze(segmentation_mask)

# # Convert to 3‑channel uint8 image for visualization.
# visualized_mask = (segmentation_mask * 255).astype(np.uint8)
# visualized_mask = np.stack([visualized_mask]*3, axis=-1)

# # Display the segmentation mask as an image
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.title('Segmentation Mask (grayscale)')
# plt.imshow(segmentation_mask, cmap='gray')
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.title('Visualized Mask (RGB)')
# plt.imshow(visualized_mask)
# plt.axis('off')
# plt.show()

# # Print a small part of the mask array
# print('Segmentation mask sample:', segmentation_mask[:5,:5])
# print('Visualized mask sample:', visualized_mask[:2,:2,:])


# In[9]:


# --- Pose-Specific Joint Angle Extraction with Visibility Filtering ---
import math

# ALL MediaPipe JOINTS
ALL_JOINTS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

# POSE-SPECIFIC IMPORTANT JOINTS (8 per pose)
POSE_SPECIFIC_JOINTS = {
    'balancing_table': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
    },
    'cobra': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
    },
    'downward_facing_dog': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    },
    'tree': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    },
    'triangle': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_elbow': 13,
        'right_elbow': 14,
    },
    'one_legged_king_pigeon': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
}

# POSE-SPECIFIC ANGLE DEFINITIONS (which 3 joints form each angle)
POSE_SPECIFIC_ANGLES = {
    'balancing_table': [
        ('left_shoulder', ('left_hip', 'left_shoulder', 'left_elbow')),
        ('right_shoulder', ('right_hip', 'right_shoulder', 'right_elbow')),
        ('left_elbow', ('left_shoulder', 'left_elbow', 'left_wrist')),
        ('right_elbow', ('right_shoulder', 'right_elbow', 'right_wrist')),
        ('left_hip', ('left_shoulder', 'left_hip', 'left_knee')),
        ('right_hip', ('right_shoulder', 'right_hip', 'right_knee')),
        ('left_knee', ('left_hip', 'left_knee', 'left_ankle')),
        ('right_knee', ('right_hip', 'right_knee', 'right_ankle')),
    ],
    'cobra': [
        ('left_shoulder', ('left_hip', 'left_shoulder', 'left_elbow')),
        ('right_shoulder', ('right_hip', 'right_shoulder', 'right_elbow')),
        ('left_elbow', ('left_shoulder', 'left_elbow', 'left_wrist')),
        ('right_elbow', ('right_shoulder', 'right_elbow', 'right_wrist')),
        ('left_hip', ('left_shoulder', 'left_hip', 'left_knee')),
        ('right_hip', ('right_shoulder', 'right_hip', 'right_knee')),
        ('left_knee', ('left_hip', 'left_knee', 'left_ankle')),
        ('right_knee', ('right_hip', 'right_knee', 'right_ankle')),
    ],
    'downward_facing_dog': [
        ('left_shoulder', ('left_hip', 'left_shoulder', 'left_elbow')),
        ('right_shoulder', ('right_hip', 'right_shoulder', 'right_elbow')),
        ('left_elbow', ('left_shoulder', 'left_elbow', 'left_wrist')),
        ('right_elbow', ('right_shoulder', 'right_elbow', 'right_wrist')),
        ('left_hip', ('left_shoulder', 'left_hip', 'left_knee')),
        ('right_hip', ('right_shoulder', 'right_hip', 'right_knee')),
        ('shoulder_hip_dist_left', ('left_shoulder', 'left_hip', 'left_hip')),  # pseudo angle
        ('shoulder_hip_dist_right', ('right_shoulder', 'right_hip', 'right_hip')),  # pseudo angle
    ],
    'tree': [
        ('left_shoulder', ('left_hip', 'left_shoulder', 'left_elbow')),
        ('right_shoulder', ('right_hip', 'right_shoulder', 'right_elbow')),
        ('left_hip', ('left_shoulder', 'left_hip', 'left_knee')),
        ('right_hip', ('right_shoulder', 'right_hip', 'right_knee')),
        ('left_knee', ('left_hip', 'left_knee', 'left_ankle')),
        ('right_knee', ('right_hip', 'right_knee', 'right_ankle')),
        ('left_ankle', ('left_knee', 'left_ankle', 'left_ankle')),  # pseudo angle
        ('right_ankle', ('right_knee', 'right_ankle', 'right_ankle')),  # pseudo angle
    ],
    'triangle': [
        ('left_shoulder', ('left_hip', 'left_shoulder', 'left_elbow')),
        ('right_shoulder', ('right_hip', 'right_shoulder', 'right_elbow')),
        ('left_hip', ('left_shoulder', 'left_hip', 'left_knee')),
        ('right_hip', ('right_shoulder', 'right_hip', 'right_knee')),
        ('left_ankle', ('left_knee', 'left_ankle', 'left_ankle')),  # pseudo
        ('right_ankle', ('right_knee', 'right_ankle', 'right_ankle')),  # pseudo
        ('left_elbow', ('left_wrist', 'left_elbow', 'left_shoulder')),
        ('right_elbow', ('right_wrist', 'right_elbow', 'right_shoulder')),
    ],
    'one_legged_king_pigeon': [
        ('left_shoulder', ('left_hip', 'left_shoulder', 'left_elbow')),
        ('right_shoulder', ('right_hip', 'right_shoulder', 'right_elbow')),
        ('left_hip', ('left_shoulder', 'left_hip', 'left_knee')),
        ('right_hip', ('right_shoulder', 'right_hip', 'right_knee')),
        ('left_knee', ('left_hip', 'left_knee', 'left_ankle')),
        ('right_knee', ('right_hip', 'right_knee', 'right_ankle')),
        ('left_ankle', ('left_knee', 'left_ankle', 'left_knee')),  # pseudo
        ('right_ankle', ('right_knee', 'right_ankle', 'right_knee')),  # pseudo
    ],
}

# Helper: angle between three points (in degrees)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Extract pose-specific joint coordinates with visibility filtering
def get_pose_landmark_coords(landmarks, image_shape, pose_name, min_visibility=0.5):
    """Extract only the pose-specific joints that are clearly visible."""
    h, w = image_shape[:2]
    coords = {}
    visibility_scores = {}
    joints_to_use = POSE_SPECIFIC_JOINTS.get(pose_name, ALL_JOINTS)
    
    for name, idx in joints_to_use.items():
        lm = landmarks[idx]
        visibility_scores[name] = lm.visibility
        if lm.visibility >= min_visibility:
            coords[name] = (lm.x * w, lm.y * h)
    
    return coords, visibility_scores

# Calculate pose-specific joint angles
def get_pose_joint_angles(landmarks, image_shape, pose_name, min_visibility=0.5):
    """Calculate only the angles relevant to this specific pose."""
    coords, visibility_scores = get_pose_landmark_coords(landmarks, image_shape, pose_name, min_visibility)
    angles = {}
    angle_visibility = {}
    
    angle_definitions = POSE_SPECIFIC_ANGLES.get(pose_name, [])
    
    for angle_name, (point_a_name, point_b_name, point_c_name) in angle_definitions:
        if point_a_name in coords and point_b_name in coords and point_c_name in coords:
            angle = calculate_angle(coords[point_a_name], coords[point_b_name], coords[point_c_name])
            angles[angle_name] = angle
            min_vis = min(visibility_scores[point_a_name], 
                         visibility_scores[point_b_name], 
                         visibility_scores[point_c_name])
            angle_visibility[angle_name] = min_vis
    
    return angles, angle_visibility

# Helper to run pose detection and get pose-specific angles
def get_pose_angles_for_image(image_path, detector, pose_name, min_visibility=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None, None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result = detector.detect(mp_image)
    if not result.pose_landmarks:
        print(f"No pose detected in {image_path}")
        return None, None, None, None
    angles, visibility = get_pose_joint_angles(result.pose_landmarks[0], rgb_img.shape, pose_name, min_visibility)
    return angles, visibility, img, result

print("Pose-specific joint angle extraction utilities loaded")


# In[10]:


# --- Create detector and utilities for all poses ---
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure detector is created (reuse from previous cell if possible)
try:
    detector
except NameError:
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    print("Detector created successfully")

# Define available poses in the images directory
AVAILABLE_POSES = ['balancing_table', 'cobra', 'downward_facing_dog', 'one_legged_king_pigeon', 'tree', 'triangle']

print(f"\nAvailable poses for analysis: {', '.join(AVAILABLE_POSES)}")


# In[11]:


# --- Ground Truth Calculation (from first 3 images) ---
def calculate_ground_truth(pose_name, detector, min_visibility=0.5):
    """
    Calculate ground truth joint angles using the first 3 images.
    Only averages angles using visible joints for consistency.
    """
    image_paths = [
        f"images/{pose_name}/{pose_name}1.jpg",
        f"images/{pose_name}/{pose_name}2.jpg",
        f"images/{pose_name}/{pose_name}3.jpg"
    ]
    
    all_angles = []
    all_visibility = []
    
    for path in image_paths:
        result = get_angles_for_image(path, detector, min_visibility)
        if result[0] is not None:  # angles exist
            angles, visibility, _, _ = result
            all_angles.append(angles)
            all_visibility.append(visibility)
            print(f"✓ Loaded from {path}")
        else:
            print(f"✗ Could not load from {path}")
    
    if not all_angles:
        print(f"Could not calculate ground truth for {pose_name}")
        return None, None
    
    # Calculate average for each joint angle
    ground_truth = {}
    average_visibility = {}
    
    for angle_name in all_angles[0].keys():
        # Get all values for this joint angle (only from images where it was detected)
        valid_angles = [angles[angle_name] for angles in all_angles if angle_name in angles]
        valid_visibilities = [vis[angle_name] for vis in all_visibility if angle_name in vis]
        
        if valid_angles:
            ground_truth[angle_name] = sum(valid_angles) / len(valid_angles)
            average_visibility[angle_name] = sum(valid_visibilities) / len(valid_visibilities)
    
    return ground_truth, average_visibility

# Example for a pose (update this to cycle through all poses)
pose_name = 'balancing_table'
ground_truth_angles, gt_visibility = calculate_ground_truth(pose_name, detector, min_visibility=0.5)

if ground_truth_angles:
    print(f"\n{'='*70}")
    print(f"Ground Truth Angles for {pose_name.upper()} (from images 1-3)")
    print(f"{'='*70}")
    print(f"{'Joint Angle':<20} | {'Avg Angle (°)':<15} | {'Avg Visibility':<15}")
    print(f"{'-'*70}")
    
    for joint, angle in sorted(ground_truth_angles.items()):
        visibility = gt_visibility[joint]
        print(f"{joint:<20} | {angle:>13.1f} | {visibility:>13.2f}")
else:
    print(f"Failed to calculate ground truth for {pose_name}")


# In[ ]:


# --- Compare Images 4 & 5 with Ground Truth ---
def compare_with_ground_truth(pose_name, ground_truth, detector, threshold=10, min_visibility=0.5):
    """
    Compare images 4 and 5 against the ground truth calculated from images 1-3.
    Only compares angles where both ground truth and test image have visible joints.
    """
    user_image_paths = [
        f"images/{pose_name}/{pose_name}4.jpg",
        f"images/{pose_name}/{pose_name}5.jpg"
    ]
    
    for path in user_image_paths:
        result = get_angles_for_image(path, detector, min_visibility)
        
        if result[0] is None:
            print(f"\n✗ Could not get angles for {path}")
            continue
        
        user_angles, user_visibility, _, _ = result
        
        print(f"\n{'='*80}")
        print(f"Comparison: {path.split('/')[-1]} vs {pose_name.upper()} Ground Truth")
        print(f"{'='*80}")
        print(f"{'Joint Angle':<18} | {'User (°)':<10} | {'Target (°)':<10} | {'Diff (°)':<10} | {'Status':<20}")
        print(f"{'-'*80}")
        
        corrections_needed = []
        
        for joint_name in sorted(ground_truth.keys()):
            # Only compare if angle was detected in both images
            if joint_name not in user_angles:
                print(f"{joint_name:<18} | {'N/A':<10} | {ground_truth[joint_name]:>8.1f}  | {'N/A':<10} | {'NOT VISIBLE':<20}")
                continue
            
            ref = ground_truth[joint_name]
            test = user_angles[joint_name]
            diff = test - ref
            
            # Determine status
            if abs(diff) > threshold:
                status = f"⚠ NEEDS CORRECTION"
                corrections_needed.append((joint_name, ref, test, -diff))
            else:
                status = "✓ OK"
            
            print(f"{joint_name:<18} | {test:>8.1f}  | {ref:>8.1f}  | {diff:>+8.1f} | {status:<20}")
        
        # Summary
        print(f"\n{'SUMMARY':<18} | {'Status':<50}")
        print(f"{'-'*80}")
        if corrections_needed:
            print(f"⚠ {len(corrections_needed)} joint(s) need correction:\n")
            for joint, target, current, correction in corrections_needed:
                print(f"  • {joint}: Adjust by {correction:+.1f}° (from {current:.1f}° to {target:.1f}°)")
        else:
            print(f"✓ All joint angles are within tolerance!")

# Compare test images against ground truth
if ground_truth_angles:
    compare_with_ground_truth(pose_name, ground_truth_angles, detector, threshold=10, min_visibility=0.5)
else:
    print("Cannot compare: ground truth not available.")


# In[ ]:


# --- Visual Comparison with Visibility Indicators ---
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_comparison_with_visibility(pose_name, ground_truth, detector, threshold=10, min_visibility=0.5):
    """
    Visualize test images (4, 5) with:
    - Pose landmarks overlaid
    - Visibility indicators
    - Side-by-side angle comparison
    """
    if not ground_truth:
        print("Ground truth not available.")
        return

    user_image_paths = [
        f"images/{pose_name}/{pose_name}4.jpg",
        f"images/{pose_name}/{pose_name}5.jpg"
    ]
    
    for path in user_image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error reading {path}")
            continue
        
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect(mp_image)
        
        if not result.pose_landmarks:
            print(f"No pose detected in {path}.")
            continue
        
        # Draw landmarks
        annotated_image = draw_landmarks_on_image(rgb_image, result)
        
        # Get angles and visibility
        user_angles, user_visibility, _, _ = get_angles_for_image(path, detector, min_visibility)
        
        # Create figure with image and comparison table
        fig = plt.figure(figsize=(18, 8))
        
        # Image subplot
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(annotated_image)
        ax1.axis('off')
        ax1.set_title(f"{path.split('/')[-1]}\n(Joints filtered by visibility)", fontsize=14, fontweight='bold')
        
        # Comparison table subplot
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        
        # Build comparison text
        text_str = "JOINT ANGLE COMPARISON\n"
        text_str += "="*70 + "\n\n"
        text_str += f"{'Joint Angle':<18} | {'User':<9} | {'Target':<9} | {'Diff':<9} | {'Status':<15}\n"
        text_str += "-"*70 + "\n"
        
        corrections_list = []
        
        for joint in sorted(ground_truth.keys()):
            if joint not in user_angles:
                text_str += f"{joint:<18} | {'N/A':<9} | {ground_truth[joint]:>7.1f}° | {'N/A':<9} | {'NOT VISIBLE':<15}\n"
                continue
            
            ref = ground_truth[joint]
            test = user_angles[joint]
            diff = test - ref
            user_vis = user_visibility[joint]
            
            if abs(diff) > threshold:
                status = f"⚠ {(-diff):+.1f}°"
                corrections_list.append(joint)
            else:
                status = "✓ OK"
            
            text_str += f"{joint:<18} | {test:>7.1f}° | {ref:>7.1f}° | {diff:>+7.1f}° | {status:<15}\n"
        
        text_str += "-"*70 + "\n"
        if corrections_list:
            text_str += f"\n⚠ CORRECTIONS NEEDED for {len(corrections_list)} joint(s):\n"
            for joint in corrections_list:
                text_str += f"   • {joint}\n"
        else:
            text_str += f"\n✓ ALL JOINTS WITHIN TOLERANCE\n"
        
        ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=11, 
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()

# Visualize comparison
if ground_truth_angles:
    visualize_comparison_with_visibility(pose_name, ground_truth_angles, detector, threshold=10, min_visibility=0.5)
else:
    print("Cannot visualize: ground truth not available.")


# In[ ]:


# --- Run full analysis pipeline for a specific pose ---
def run_full_analysis(pose_name, detector, min_visibility=0.5, angle_threshold=10):
    """
    Complete analysis pipeline:
    1. Calculate ground truth from images 1-3 (using only visible joints)
    2. Compare images 4-5 against the ground truth
    3. Display results
    """
    print(f"\n{'#'*80}")
    print(f"FULL ANALYSIS: {pose_name.upper()}")
    print(f"{'#'*80}\n")
    
    # Step 1: Calculate ground truth
    print(f"Step 1: Calculating ground truth from images 1-3...")
    ground_truth, visibility = calculate_ground_truth(pose_name, detector, min_visibility = min_visibility)
    
    if not ground_truth:
        print(f"Failed to calculate ground truth for {pose_name}")
        return
    
    print(f"✓ Ground truth calculated with {len(ground_truth)} visible joint angles\n")
    
    # Step 2: Compare with test images
    print(f"Step 2: Comparing images 4-5 with ground truth...\n")
    compare_with_ground_truth(pose_name, ground_truth, detector, threshold=angle_threshold, min_visibility=min_visibility)
    
    # Step 3: Visualize
    print(f"\nStep 3: Generating visualization...\n")
    visualize_comparison_with_visibility(pose_name, ground_truth, detector, threshold=angle_threshold, min_visibility=min_visibility)
    
    return ground_truth, visibility

# Run analysis for balancing_table pose
all_poses = ['balancing_table', 'cobra', 'downward_facing_dog', 'one_legged_king_pigeon', 'tree', 'triangle']
for i in range(len(all_poses)):
    pose_to_analyze = all_poses[i]
    gt_angles, gt_vis = run_full_analysis(pose_to_analyze, detector, min_visibility=0.5, angle_threshold=10)

