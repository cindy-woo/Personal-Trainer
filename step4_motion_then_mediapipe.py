import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH   = "/Users/kyungyoon-choung/Downloads/cpp files/project/pose_landmarker_full.task"
CAMERA_INDEX = 0
STABLE_SECONDS   = 5
MOTION_THRESHOLD = 25
MOTION_PERCENT   = 2.0
SAVE_DIR = "/Users/kyungyoon-choung/Downloads/cpp files/project"

LANDMARK_NAMES = ["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]

def detect_motion(frame1, frame2):
    gray1 = cv2.GaussianBlur(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (21,21), 0)
    gray2 = cv2.GaussianBlur(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (21,21), 0)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    pct = 100 * np.count_nonzero(thresh) / thresh.size
    return pct > MOTION_PERCENT, pct

def compute_angle(a, b, c):
    ba, bc = a-b, c-b
    n = np.linalg.norm(ba)*np.linalg.norm(bc)
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/n,-1,1)))) if n>1e-8 else 0.0

def run_mediapipe(frame):
    ANGLE_DEFS = {"L elbow":(11,13,15),"R elbow":(12,14,16),"L shoulder":(13,11,23),"R shoulder":(14,12,24),"L hip":(11,23,25),"R hip":(12,24,26),"L knee":(23,25,27),"R knee":(24,26,28)}
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(base_options=base_opts, running_mode=mp_vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.pose_landmarks: return None, None, None
    lms    = result.pose_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])   # ALL 33 joints
    vis    = np.array([lm.visibility for lm in lms])          # ALL 33 visibilities
    angles = {name: compute_angle(coords[a], coords[b], coords[c]) for name,(a,b,c) in ANGLE_DEFS.items() if vis[a]>0.5 and vis[b]>0.5 and vis[c]>0.5}

    # ── Save all 33 joints ──
    np.save(f"{SAVE_DIR}/user_pose.npy", coords)
    np.save(f"{SAVE_DIR}/user_pose_visibility.npy", vis)
    print(f"\nSaved all 33 joints → user_pose.npy")

    # ── Print all 33 joints ──
    print("\n" + "="*60)
    print("  ALL 33 JOINT COORDINATES")
    print("="*60)
    print(f"{'#':<4} {'Landmark':<22} {'x':>8} {'y':>8} {'z':>8} {'vis':>6}")
    print("-"*60)
    for i in range(33):
        x,y,z = coords[i]; v = vis[i]
        flag = "  ✓" if v > 0.5 else "  ✗ (low)"
        print(f"{i:<4} {LANDMARK_NAMES[i]:<22} {x:>8.4f} {y:>8.4f} {z:>8.4f} {v:>6.2f}{flag}")

    print("\n  Joint Angles:")
    for name,angle in angles.items(): print(f"  {name:<15} {angle:.1f}°")
    print("="*60)
    return coords, vis, angles

def draw_skeleton(frame, coords, vis):
    h,w = frame.shape[:2]
    for i,j in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
        if vis[i]<0.5 or vis[j]<0.5: continue
        cv2.line(frame,(int(coords[i,0]*w),int(coords[i,1]*h)),(int(coords[j,0]*w),int(coords[j,1]*h)),(0,220,0),2)
    for idx in range(33):
        if vis[idx]<0.5: continue
        cv2.circle(frame,(int(coords[idx,0]*w),int(coords[idx,1]*h)),5,(0,220,0),-1)
    return frame

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened(): print("Error: Could not open webcam."); exit()
print(f"Hold still for {STABLE_SECONDS} seconds to capture pose. Press Q to quit.")

prev_frame=None; stable_since=None; captured=False; result_frame=None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    if prev_frame is not None:
        is_moving, _ = detect_motion(prev_frame, frame)
        if is_moving: stable_since = None
        elif stable_since is None: stable_since = now
    prev_frame = frame.copy()

    display = result_frame.copy() if result_frame is not None else frame.copy()

    if result_frame is not None:
        cv2.putText(display,"POSE CAPTURED — move to reset or Q to quit",(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,0),2)
        if stable_since is None:
            result_frame=None; captured=False
            print("Movement detected — reset. Hold still again.")

    elif stable_since is not None:
        elapsed   = now - stable_since
        remaining = max(0, STABLE_SECONDS - elapsed)
        bar_w     = min(int((elapsed/STABLE_SECONDS)*(w-40)), w-40)
        cv2.rectangle(display,(20,h-40),(w-20,h-20),(60,60,60),-1)
        cv2.rectangle(display,(20,h-40),(20+bar_w,h-20),(0,220,0),-1)
        cv2.putText(display,f"Still! Capturing in {remaining:.1f}s...",(10,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,0),2)

        if elapsed >= STABLE_SECONDS and not captured:
            captured = True
            print(f"\nStable for {STABLE_SECONDS}s — running MediaPipe...")
            coords, vis, angles = run_mediapipe(frame.copy())
            if coords is not None:
                result_frame = draw_skeleton(frame.copy(), coords, vis)
                cv2.imwrite(f"{SAVE_DIR}/captured_pose.png", result_frame)
                print(f"Image saved → captured_pose.png")
            else:
                print("No pose detected. Move and try again.")
                captured=False; stable_since=None
    else:
        cv2.putText(display,"Move detected — hold still to start timer",(10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,140,255),2)

    cv2.putText(display,f"Hold still {STABLE_SECONDS}s to capture | Q=quit",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
    cv2.imshow("Pose Capture", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
