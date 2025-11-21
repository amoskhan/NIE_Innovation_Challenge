# ...existing code...
import argparse
import time
import cv2
import numpy as np

# Try MediaPipe Tasks (PoseLandmarker) first, fall back to mediapipe.solutions.pose
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        VisionRunningMode,
    )

    HAS_TASKS = True
except Exception:
    HAS_TASKS = False
    import mediapipe as mp
    mp_pose = mp.solutions.pose

MODEL_PATH = "models/pose_landmarker_full.task"


def hip_centroid_pixels(landmarks, img_w, img_h):
    # compute centroid between left and right hip if available
    # MediaPipe indices: LEFT_HIP=23, RIGHT_HIP=24 (if using standard pose)
    try:
        l = landmarks[23]
        r = landmarks[24]
        if hasattr(l, "x") and hasattr(r, "x"):
            x = int(((l.x + r.x) / 2.0) * img_w)
            y = int(((l.y + r.y) / 2.0) * img_h)
            return (x, y)
    except Exception:
        pass
    # fallback to average of all landmarks
    xs = []
    ys = []
    for lm in landmarks:
        if getattr(lm, "x", None) is not None and getattr(lm, "y", None) is not None:
            xs.append(lm.x)
            ys.append(lm.y)
    if not xs:
        return None
    return (int(np.mean(xs) * img_w), int(np.mean(ys) * img_h))


def draw_landmarks(frame, result):
    """
    Draw expressive skeleton: white backbone, colored inner stroke (cyan for left, orange for right),
    and larger filled joint circles with white borders. Accepts either a MediaPipe Tasks result
    (with `pose_landmarks`) or a mediapipe.solutions pose result.
    """
    h, w = frame.shape[:2]

    # prepare landmarks list from result (support Tasks API and solutions API)
    landmarks = None
    connections = None
    left_ids = set()
    right_ids = set()

    if HAS_TASKS:
        # MediaPipe Tasks PoseLandmarkerResult -> .pose_landmarks (list of NormalizedLandmarkList)
        if result and getattr(result, "pose_landmarks", None):
            if len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0].landmarks
    else:
        # mediapipe.solutions result -> .pose_landmarks (NormalizedLandmarkList)
        if result and getattr(result, "pose_landmarks", None):
            landmarks = result.pose_landmarks.landmark

    # build side id sets using MediaPipe enum if available (for coloring)
    try:
        import mediapipe as _mp

        for lm in _mp.solutions.pose.PoseLandmark:
            if lm.name.startswith("LEFT_"):
                left_ids.add(lm.value)
            elif lm.name.startswith("RIGHT_"):
                right_ids.add(lm.value)
        connections = _mp.solutions.pose.POSE_CONNECTIONS
    except Exception:
        connections = None

    # draw skeleton connections
    if landmarks and connections:
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                pa = landmarks[a]
                pb = landmarks[b]
                if None in (getattr(pa, "x", None), getattr(pa, "y", None), getattr(pb, "x", None), getattr(pb, "y", None)):
                    continue
                xa = max(0, min(w - 1, int(pa.x * w)))
                ya = max(0, min(h - 1, int(pa.y * h)))
                xb = max(0, min(w - 1, int(pb.x * w)))
                yb = max(0, min(h - 1, int(pb.y * h)))

                # white base
                cv2.line(frame, (xa, ya), (xb, yb), (255, 255, 255), 6, cv2.LINE_AA)

                # colored inner stroke for same-side segments
                seg_color = None
                if left_ids and a in left_ids and b in left_ids:
                    seg_color = (255, 255, 0)  # cyan (BGR)
                elif right_ids and a in right_ids and b in right_ids:
                    seg_color = (0, 165, 255)  # orange (BGR)
                if seg_color:
                    cv2.line(frame, (xa, ya), (xb, yb), seg_color, 3, cv2.LINE_AA)

    # draw joints
    if landmarks:
        for i, lm in enumerate(landmarks):
            if getattr(lm, "x", None) is None or getattr(lm, "y", None) is None:
                continue
            cx = max(0, min(w - 1, int(lm.x * w)))
            cy = max(0, min(h - 1, int(lm.y * h)))

            joint_color = (0, 0, 255)
            if left_ids and i in left_ids:
                joint_color = (255, 255, 0)
            elif right_ids and i in right_ids:
                joint_color = (0, 165, 255)

            cv2.circle(frame, (cx, cy), 7, joint_color, -1)
            cv2.circle(frame, (cx, cy), 9, (255, 255, 255), 2)
            cv2.putText(frame, str(i), (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)


def run_video(cam_index=0, mode="VIDEO"):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open camera", cam_index)
        return

    if HAS_TASKS:
        # create PoseLandmarker in VIDEO mode
        mode_enum = VisionRunningMode.VIDEO if mode.upper() == "VIDEO" else VisionRunningMode.LIVE_STREAM
        options = PoseLandmarkerOptions(
            model_asset_path=MODEL_PATH,
            running_mode=mode_enum,
        )
        landmarker = PoseLandmarker.create_from_options(options)
    else:
        # fallback: mediapipe.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        # tasks API expects RGB image wrapped in vision.Image
        if HAS_TASKS:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = vision.Image.create_from_array(rgb)
            # VIDEO mode: use detect; LIVE_STREAM would use a callback mechanism
            try:
                result = landmarker.detect(mp_image)
            except Exception:
                result = None
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            result = results  # pass through to draw_landmarks

        draw_landmarks(display, result)

        # draw hip centroid for tracking
        if HAS_TASKS and result and getattr(result, "pose_landmarks", None):
            if len(result.pose_landmarks) > 0:
                center = hip_centroid_pixels(result.pose_landmarks[0].landmarks, frame.shape[1], frame.shape[0])
                if center:
                    cv2.circle(display, center, 6, (0, 255, 0), -1)
        elif not HAS_TASKS and result and getattr(result, "pose_landmarks", None):
            center = hip_centroid_pixels(result.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
            if center:
                cv2.circle(display, center, 6, (0, 255, 0), -1)

        # FPS
        cur = time.time()
        fps = 1.0 / (cur - prev) if prev != 0 else 0.0
        prev = cur
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Pose Landmarker", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="webcam index")
    parser.add_argument("--mode", type=str, default="VIDEO", choices=["VIDEO", "LIVE_STREAM"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video(cam_index=args.cam, mode=args.mode)
# ...existing code...