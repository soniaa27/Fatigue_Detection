"""
alert.py — Real-time fatigue detection system.

Connects:
  webcam → features.py → baseline.py → inference.py → alert

Press 'q' to quit.
Press 'c' to recalibrate baseline.

Usage:
    python alert.py
    python alert.py --threshold 0.45   # lower = more sensitive
    python alert.py --no-gru           # rule-based only (no model)
"""

import cv2
import time
import os
import numpy as np
import argparse
from datetime import datetime
from collections import deque
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe import Image, ImageFormat

from features import FrameAggregator, extract_features
from head_pose import _build_default_camera_matrix
from baseline import BaselineMonitor, build_baseline_from_csv
from inference import FatigueInferenceEngine

import urllib.request

print("ALERT.PY STARTED")

import sys
print("ARGS:", sys.argv)


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_DIR    = "../models"
SESSION_DIR  = "../data/sessions"
USER_ID      = "sonia"

# Alert fires if ANY of these are true:
#   1. GRU probability > GRU_THRESHOLD
#   2. Baseline deviation_score > DEVIATION_THRESHOLD
#   3. Any absolute threshold triggered (always-on)
GRU_THRESHOLD       = 0.50
DEVIATION_THRESHOLD = 0.65

# Minimum seconds of consecutive fatigue signal before alert fires
# Prevents single-frame false positives
ALERT_SUSTAIN_SEC = 8


def _fmt_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─── Audio alert ──────────────────────────────────────────────────────────────

def _play_alert():
    """Play a beep. Falls back gracefully if audio unavailable."""
    try:
        import subprocess
        subprocess.run(['afplay', '/System/Library/Sounds/Basso.aiff'],
                       capture_output=True, timeout=2)
    except Exception:
        try:
            print('\a', end='', flush=True)   # terminal bell
        except Exception:
            pass


# ─── Overlay drawing ──────────────────────────────────────────────────────────

def _draw_hud(frame, state: dict):
    """Draw the fatigue monitoring HUD on the frame."""
    h, w = frame.shape[:2]

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (w, 110), (60, 60, 60), 1)

    # EAR and PERCLOS
    ear_col = (0, 255, 0) if state['ear'] > 0.22 else (0, 100, 255)
    cv2.putText(frame, f"EAR: {state['ear']:.3f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_col, 2)
    cv2.putText(frame, f"PERCLOS: {state['perclos']:.2f}",
                (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # GRU probability bar
    gru_prob = state.get('gru_prob', 0.0)
    bar_w    = int((w - 20) * gru_prob)
    bar_col  = (0, 200, 0) if gru_prob < 0.4 else (0, 140, 255) if gru_prob < 0.65 else (0, 0, 255)
    cv2.rectangle(frame, (10, 38), (w - 10, 58), (50, 50, 50), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (10, 38), (10 + bar_w, 58), bar_col, -1)
    cv2.putText(frame, f"GRU: {gru_prob:.2f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Deviation score bar
    dev_score = state.get('deviation', 0.0)
    dev_w     = int((w - 20) * dev_score)
    dev_col   = (0, 200, 0) if dev_score < 0.4 else (0, 140, 255) if dev_score < 0.65 else (0, 0, 255)
    cv2.rectangle(frame, (10, 62), (w - 10, 78), (50, 50, 50), -1)
    if dev_w > 0:
        cv2.rectangle(frame, (10, 62), (10 + dev_w, 78), dev_col, -1)
    cv2.putText(frame, f"DEV: {dev_score:.2f}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Buffer fill progress
    buf = state.get('buffer_fill', 0.0)
    buf_w = int((w - 20) * buf)
    cv2.rectangle(frame, (10, 82), (w - 10, 90), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 82), (10 + buf_w, 90), (100, 100, 200), -1)
    cv2.putText(frame, f"Buffer: {buf*100:.0f}%  Nod:{state.get('nod',0)}  Droop:{state.get('droop',0)}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Alerts from absolute thresholds
    alerts = state.get('alerts', [])
    if alerts:
        for i, alert_msg in enumerate(alerts[:2]):
            cv2.putText(frame, f"! {alert_msg}",
                        (10, h - 50 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)

    # FATIGUE ALERT banner
    if state.get('alert_active'):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 50), (w, h//2 + 50), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "⚠  FATIGUE DETECTED  ⚠",
                    (w//2 - 200, h//2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # Calibration progress
    if state.get('calibrating'):
        pct = int(state.get('cal_progress', 0) * 100)
        cv2.putText(frame, f"Calibrating baseline... {pct}%",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)


def _draw_landmarks(frame, landmarks, img_w, img_h):
    from features import LEFT_EYE, RIGHT_EYE, MOUTH
    for idx in LEFT_EYE + RIGHT_EYE:
        lm = landmarks[idx]
        cv2.circle(frame, (int(lm.x*img_w), int(lm.y*img_h)), 2, (0,255,200), -1)
    for idx in MOUTH:
        lm = landmarks[idx]
        cv2.circle(frame, (int(lm.x*img_w), int(lm.y*img_h)), 2, (200,200,0), -1)


# ─── Download model helper ────────────────────────────────────────────────────

def _download_face_model(model_path="face_landmarker.task"):
    if not os.path.exists(model_path):
        print("[INFO] Downloading face landmarker model...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
    return model_path


# ─── Main system ──────────────────────────────────────────────────────────────

def run_alert_system(
    threshold: float = GRU_THRESHOLD,
    use_gru: bool = True,
    show_overlay: bool = True,
    headless: bool = False,
    web_frame_path: str | None = None,
    input_video: str | None = None,
):
    print("\n" + "="*50)
    print("  Fatigue Detection System")
    print("="*50)

    # ── Load baseline ──────────────────────────────────────────────────────
    import glob
    alert_csvs = sorted(glob.glob(f"{SESSION_DIR}/alert*.csv"))
    monitor    = build_baseline_from_csv(alert_csvs, user_id=USER_ID,
                                         profile_dir=MODEL_DIR)
    print(f"[SYSTEM] Baseline loaded for '{USER_ID}'")

    # ── Load GRU inference engine ──────────────────────────────────────────
    engine = None
    if use_gru:
        try:
            engine = FatigueInferenceEngine(
                model_dir=MODEL_DIR,
                decision_threshold=threshold,
            )
            print(f"[SYSTEM] GRU engine ready  (threshold={threshold:.2f})")
        except FileNotFoundError as e:
            print(f"[WARN] GRU model not found — running rule-based only\n  {e}")
            engine = None

    # ── MediaPipe setup ────────────────────────────────────────────────────
    model_path   = _download_face_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    mp_options   = FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    # ── Video source ───────────────────────────────────────────────────────
    video_mode = input_video is not None
    source = input_video if video_mode else 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps = cap.get(cv2.CAP_PROP_FPS) if video_mode else 0.0
    if not fps or fps <= 1:
        fps = 30.0
    frames_per_second = max(int(round(fps)), 1)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    img_h, img_w  = first_frame.shape[:2]
    camera_matrix = _build_default_camera_matrix(img_w, img_h)
    dist_coeffs   = np.zeros((4, 1), dtype=np.float64)

    # ── State ──────────────────────────────────────────────────────────────
    aggregator       = FrameAggregator()
    second_timer     = time.time()
    prev_frame       = None
    frozen_count     = 0
    alert_sustained  = 0          # consecutive seconds of fatigue signal
    alert_active     = False
    last_alert_time  = -999.0
    ALERT_COOLDOWN   = 30         # seconds between repeated alerts

    hud_state = {
        'ear': 0.0, 'perclos': 0.0, 'gru_prob': 0.0,
        'deviation': 0.0, 'buffer_fill': 0.0,
        'nod': 0, 'droop': 0, 'alerts': [],
        'alert_active': False, 'calibrating': False,
    }

    print("\n[SYSTEM] Running — press Q to quit, C to recalibrate\n")
    if video_mode:
        print(f"[SYSTEM] Video input: {input_video}")
    frame = first_frame
    frame_index = 1
    last_second_frame = 0

    frame_out_path = Path(web_frame_path).resolve() if web_frame_path else None
    if frame_out_path:
        frame_out_path.parent.mkdir(parents=True, exist_ok=True)

    with FaceLandmarker.create_from_options(mp_options) as landmarker:
        while True:

            # ── Frozen frame check ─────────────────────────────────────────
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                if diff.max() == 0:
                    frozen_count += 1
                    if frozen_count > 5:
                        if not headless:
                            cv2.imshow("Fatigue Monitor — Q to quit", frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        continue
                else:
                    frozen_count = 0
            prev_frame = frame.copy()

            img_h, img_w = frame.shape[:2]

            # ── Face detection ─────────────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect(mp_image)

            face_detected = bool(result.face_landmarks and
                                  len(result.face_landmarks) > 0)

            if face_detected:
                landmarks = result.face_landmarks[0]
                if show_overlay:
                    _draw_landmarks(frame, landmarks, img_w, img_h)

                feats = extract_features(landmarks, frame, img_w, img_h,
                                         camera_matrix, dist_coeffs)
                if feats is not None:
                    aggregator.add_frame(feats)
                    hud_state['ear']   = feats['ear_avg']
                    hud_state['nod']   = feats['nod_detected']
                    hud_state['droop'] = feats['head_droop']
            else:
                cv2.putText(frame, "NO FACE",
                            (img_w//2 - 60, img_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # ── Per-second processing ──────────────────────────────────────
            should_process_second = (
                (frame_index - last_second_frame) >= frames_per_second
                if video_mode
                else (time.time() - second_timer >= 1.0)
            )

            if should_process_second:
                second_features = aggregator.get_second_features()

                if second_features and face_detected:
                    # Frozen row check
                    if not (second_features['blink_count'] == 0
                            and second_features['perclos'] == 0.0
                            and second_features['ear_mean'] == second_features['ear_min']):

                        hud_state['perclos'] = second_features['perclos']

                        # Baseline deviation score
                        baseline_result = monitor.score(second_features)
                        dev_score       = baseline_result['deviation_score']
                        abs_alerts      = baseline_result['alerts']
                        hud_state['deviation'] = dev_score
                        hud_state['alerts']    = abs_alerts

                        # GRU inference
                        gru_prob = 0.0
                        if engine is not None:
                            prob = engine.update(second_features)
                            if prob is not None:
                                gru_prob = prob
                            else:
                                gru_prob = engine.fatigue_probability
                            hud_state['gru_prob']     = gru_prob
                            hud_state['buffer_fill']  = engine.buffer_fill

                        # ── Alert logic ────────────────────────────────────
                        # Three ways to trigger fatigue:
                        gru_trigger      = engine is not None and gru_prob > threshold
                        baseline_trigger = dev_score > DEVIATION_THRESHOLD
                        absolute_trigger = len(abs_alerts) > 0

                        fatigue_signal = gru_trigger or baseline_trigger or absolute_trigger

                        if fatigue_signal:
                            alert_sustained += 1
                        else:
                            alert_sustained = max(0, alert_sustained - 1)

                        # Fire alert only after sustained signal
                        now = time.time()
                        if (alert_sustained >= ALERT_SUSTAIN_SEC and
                                (now - last_alert_time) > ALERT_COOLDOWN):
                            alert_active    = True
                            last_alert_time = now
                            _play_alert()
                            alert_stamp = _fmt_hms(frame_index / fps) if video_mode else datetime.now().strftime('%H:%M:%S')
                            print(f"\n{'='*40}")
                            print(f"⚠  FATIGUE ALERT  [{'VIDEO ' if video_mode else ''}{alert_stamp}]")
                            print(f"   GRU prob:   {gru_prob:.3f}  (>{threshold:.2f}? {gru_trigger})")
                            print(f"   Deviation:  {dev_score:.3f}  (>{DEVIATION_THRESHOLD}? {baseline_trigger})")
                            print(f"   Absolutes:  {abs_alerts}")
                            print(f"{'='*40}\n")
                        elif alert_sustained < ALERT_SUSTAIN_SEC // 2:
                            alert_active = False

                        hud_state['alert_active'] = alert_active

                        if video_mode:
                            stamp = _fmt_hms(frame_index / fps)
                            prefix = f"[VIDEO {stamp}]"
                        else:
                            stamp = datetime.now().strftime('%H:%M:%S')
                            prefix = f"[{stamp}]"

                        # Console readout
                        print(
                            f"{prefix}  "
                            f"EAR={second_features['ear_mean']:.3f}  "
                            f"PERCLOS={second_features['perclos']:.2f}  "
                            f"GRU={gru_prob:.2f}  "
                            f"DEV={dev_score:.2f}  "
                            f"sustained={alert_sustained}/{ALERT_SUSTAIN_SEC}"
                        )

                        if video_mode:
                            last_second_frame = frame_index

                if not video_mode:
                    second_timer = time.time()
                else:
                    last_second_frame = frame_index

            # ── Draw HUD and show ──────────────────────────────────────────
            _draw_hud(frame, hud_state)

            if frame_out_path is not None:
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    try:
                        frame_out_path.write_bytes(encoded.tobytes())
                    except Exception:
                        pass

            if not headless:
                cv2.imshow("Fatigue Monitor — Q to quit", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("\n[SYSTEM] Recalibrating baseline — stay alert for 5 minutes...")
                    # Trigger fresh calibration
                    monitor = build_baseline_from_csv(
                        sorted(__import__('glob').glob(f"{SESSION_DIR}/alert*.csv")),
                        user_id=USER_ID, profile_dir=MODEL_DIR
                    )
                    print("[SYSTEM] Recalibration complete.")

            ret, frame = cap.read()
            if not ret:
                if video_mode:
                    break
                continue
            frame_index += 1

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print("\n[SYSTEM] Session ended.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fatigue Detection Alert System")
    parser.add_argument('--threshold', type=float, default=GRU_THRESHOLD,
                        help=f'GRU fatigue threshold (default: {GRU_THRESHOLD})')
    parser.add_argument('--no-gru',    action='store_true',
                        help='Disable GRU — use rule-based detection only')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Disable landmark overlay')
    parser.add_argument('--headless', action='store_true',
                        help='Run without OpenCV window (for web streaming)')
    parser.add_argument('--web-frame-path', type=str, default=None,
                        help='Optional JPEG output path for latest frame')
    parser.add_argument('--input-video', type=str, default=None,
                        help='Process a video file instead of the webcam')
    args = parser.parse_args()

    run_alert_system(
        threshold    = args.threshold,
        use_gru      = not args.no_gru,
        show_overlay = not args.no_overlay,
        headless     = args.headless,
        web_frame_path = args.web_frame_path,
        input_video  = args.input_video,
    )