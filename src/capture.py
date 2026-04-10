import cv2
import mediapipe as mp
import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
from features import FrameAggregator, extract_features
from head_pose import _build_default_camera_matrix

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe import Image, ImageFormat
import urllib.request


# ─── Model download ───────────────────────────────────────────────────────────

def download_model(model_path="face_landmarker.task"):
    if not os.path.exists(model_path):
        print("[INFO] Downloading face landmarker model (~30MB)...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Model downloaded.")
    return model_path


# ─── Session save ─────────────────────────────────────────────────────────────

def _save_session(data: list, session_name: str) -> str:
    os.makedirs("../data/sessions", exist_ok=True)
    path = f"../data/sessions/{session_name}.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ─── Frozen frame detection ───────────────────────────────────────────────────

def _is_frozen(frame: np.ndarray, prev_frame) -> bool:
    """
    Returns True if the current frame is pixel-identical to the previous one.
    A frozen webcam feed produces identical frames indefinitely.
    """
    if prev_frame is None:
        return False
    diff = cv2.absdiff(frame, prev_frame)
    return bool(diff.max() == 0)


def _is_frozen_row(row: dict) -> bool:
    """
    Returns True if a per-second feature row looks like it came from
    frozen frames — zero blink count, zero PERCLOS, and ear_mean == ear_min
    (no variation whatsoever across the second).
    """
    return (
        row["blink_count"] == 0
        and row["perclos"]  == 0.0
        and row["ear_mean"] == row["ear_min"]
    )


# ─── Main capture loop ────────────────────────────────────────────────────────

def run_capture(save_csv=True, session_name=None, show_overlay=True):
    """
    Opens webcam, runs MediaPipe face landmarker, extracts features every second.

    Args:
        save_csv:     Save session data to CSV on exit
        session_name: Label for this recording e.g. 'alert_1', 'tired_1'
        show_overlay: Draw eye/mouth dots on the webcam feed
    """

    model_path   = download_model()
    session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregator   = FrameAggregator()
    session_data = []
    second_timer = time.time()

    # Frozen frame tracking
    prev_frame        = None
    frozen_count      = 0
    MAX_FROZEN_FRAMES = 5    # consecutive identical frames before we warn + skip

    # Autosave tracking
    last_autosave_count = 0

    print(f"\n[INFO] Session: {session_name}")
    print("[INFO] Press 'q' to quit\n")

    # ── MediaPipe landmarker ───────────────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam. "
            "Check System Preferences → Privacy & Security → Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read one frame to get dimensions, build camera matrix once
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from webcam.")

    img_h, img_w  = frame.shape[:2]
    camera_matrix = _build_default_camera_matrix(img_w, img_h)
    dist_coeffs   = np.zeros((4, 1), dtype=np.float64)

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            # ── Frozen frame check ─────────────────────────────────────────
            if _is_frozen(frame, prev_frame):
                frozen_count += 1
                if frozen_count > MAX_FROZEN_FRAMES:
                    if frozen_count == MAX_FROZEN_FRAMES + 1:
                        print("[WARN] Webcam appears frozen — waiting for new frames...")
                    # Still show the last good frame so window stays open
                    cv2.imshow("Fatigue Detection — press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    continue
            else:
                if frozen_count > MAX_FROZEN_FRAMES:
                    print("[INFO] Webcam feed resumed.")
                frozen_count = 0

            prev_frame = frame.copy()
            img_h, img_w = frame.shape[:2]

            # ── Face detection ─────────────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect(mp_image)

            face_detected = bool(
                result.face_landmarks and len(result.face_landmarks) > 0
            )

            if face_detected:
                landmarks = result.face_landmarks[0]

                if show_overlay:
                    _draw_landmarks(frame, landmarks, img_w, img_h)

                feats = extract_features(
                    landmarks, frame, img_w, img_h,
                    camera_matrix, dist_coeffs
                )

                if feats is not None:
                    aggregator.add_frame(feats)

                    # Live HUD
                    ear_color = (0, 255, 0) if feats["ear_avg"] > 0.20 else (0, 0, 255)
                    cv2.putText(frame, f"EAR:   {feats['ear_avg']:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)
                    cv2.putText(frame, f"MAR:   {feats['mar']:.3f}",
                                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"Pitch:{feats['pitch']:+.1f}  "
                                f"Nod:{feats['nod_detected']}  "
                                f"Droop:{feats['head_droop']}",
                                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

            else:
                cv2.putText(frame, "NO FACE DETECTED",
                            (img_w // 2 - 130, img_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # ── Per-second aggregation ─────────────────────────────────────
            if time.time() - second_timer >= 1.0:
                second_features = aggregator.get_second_features()

                if second_features and face_detected:

                    # Discard rows that look like frozen frame artifacts
                    if _is_frozen_row(second_features):
                        print("[SKIP] Frozen frame artifact detected — row discarded")

                    else:
                        second_features["timestamp"] = datetime.now().isoformat()
                        second_features["session"]   = session_name
                        session_data.append(second_features)

                        print(
                            f"[{second_features['timestamp'][11:19]}]  "
                            f"EAR={second_features['ear_mean']:.3f}  "
                            f"PERCLOS={second_features['perclos']:.2f}  "
                            f"Blinks={second_features['blink_count']}  "
                            f"Pitch={second_features['pitch_mean']:+.1f}°  "
                            f"Nod={second_features['nod_detected']}  "
                            f"Droop={second_features['head_droop']}"
                        )

                # Autosave every 30 rows
                if (
                    save_csv
                    and len(session_data) > 0
                    and len(session_data) % 30 == 0
                    and len(session_data) != last_autosave_count
                ):
                    _save_session(session_data, session_name + "_autosave")
                    last_autosave_count = len(session_data)
                    print(f"[AUTOSAVE] {len(session_data)} rows saved.")

                second_timer = time.time()

            cv2.imshow("Fatigue Detection — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Read next frame at end of loop
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed, retrying...")
                continue

    cap.release()
    cv2.destroyAllWindows()

    if save_csv and session_data:
        path = _save_session(session_data, session_name)
        # Clean up autosave file now that we have the real save
        autosave_path = f"../data/sessions/{session_name}_autosave.csv"
        if os.path.exists(autosave_path):
            os.remove(autosave_path)
        print(f"\n[DONE] Saved {len(session_data)} rows → {path}")
    else:
        print("\n[DONE] Session ended. No data saved.")

    return session_data


# ─── Landmark overlay ─────────────────────────────────────────────────────────

def _draw_landmarks(frame, landmarks, img_w, img_h):
    from features import LEFT_EYE, RIGHT_EYE, MOUTH
    for idx in LEFT_EYE + RIGHT_EYE:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 200), -1)
    for idx in MOUTH:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(frame, (x, y), 2, (200, 200, 0), -1)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fatigue detection — capture session")
    parser.add_argument("--session",    type=str,  default=None,
                        help="Session name e.g. 'alert_1' or 'tired_1'")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable landmark overlay (slightly faster)")
    parser.add_argument("--no-save",    action="store_true",
                        help="Don't write CSV (useful for quick tests)")
    args = parser.parse_args()

    run_capture(
        save_csv=not args.no_save,
        session_name=args.session,
        show_overlay=not args.no_overlay,
    )