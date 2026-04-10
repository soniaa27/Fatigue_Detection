from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


# MediaPipe Face Mesh indices for the 6 canonical head-pose anchor points.
# 1: nose tip, 152: chin, 33/263: eye corners, 61/291: mouth corners.
ANCHOR_INDEX_MAP: Dict[str, int] = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Standard 3D facial model points in millimeters.
# Order must match ANCHOR_INDEX_MAP insertion order above.
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass
class HeadPoseState:
    alpha: float = 0.4
    nod_threshold_deg: float = 10.0
    nod_min_rebound_total_deg: float = 12.0
    nod_rebound_deg: float = 3.0
    nod_window_sec: float = 1.5
    nod_cooldown_sec: float = 0.8
    nod_latch_sec: float = 0.3
    '''
    droop_threshold_deg: float = -10.0
    droop_duration_sec: float = 1.5
    
    pitch_baseline_alpha: float = 0.01
    baseline_freeze_offset_deg: float = 4.0
    baseline_freeze_velocity_dps: float = 12.0
    '''
    variance_window_sec: float = 30.0
    droop_threshold_deg: float = -12.0
    droop_duration_sec: float = 1.5
    baseline_freeze_offset_deg: float = 8.0    # freeze baseline when pitch moves >8° from it
    baseline_freeze_velocity_dps: float = 6.0  # freeze at lower velocity
    pitch_baseline_alpha: float = 0.003        # much slower baseline drift
    

    last_smoothed: Optional[np.ndarray] = None
    pitch_baseline: Optional[float] = None
    prev_pitch_for_baseline: Optional[float] = None
    prev_pitch_time: Optional[float] = None
    pose_history: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    nod_event_timestamps: List[float] = field(default_factory=list)
    last_nod_event_time: float = -1e9
    nod_state: str = "NEUTRAL"
    nod_start_time: Optional[float] = None
    nod_extreme_pitch: float = 0.0
    nod_rearm_neutral_band_deg: float = 4.0
    nod_block_until_time: float = -1e9
    nod_latch_until_time: float = -1e9
    droop_start_time: Optional[float] = None
    last_csv_log_time: float = 0.0


class HeadPoseFeatureExtractor:
    def __init__(self, log_path: str = "data/head_pose_log.csv", nod_log_path: str = "data/nod_events.log") -> None:
        self.state = HeadPoseState()
        self.log_path = Path(log_path)
        self.nod_log_path = Path(nod_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.nod_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if not self.log_path.exists():
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "pitch",
                    "yaw",
                    "roll",
                    "head_var",
                    "nod_detected",
                    "head_droop",
                ]
            ).to_csv(self.log_path, index=False)

    @staticmethod
    def _extract_xy(landmark: object, frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
        if landmark is None:
            return None

        # MediaPipe landmark object path.
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            x = float(getattr(landmark, "x")) * frame_w
            y = float(getattr(landmark, "y")) * frame_h
            if np.isnan(x) or np.isnan(y):
                return None
            return x, y

        # Generic sequence path: [x, y] normalized or pixel coordinates.
        if isinstance(landmark, (tuple, list, np.ndarray)) and len(landmark) >= 2:
            x = float(landmark[0])
            y = float(landmark[1])
            if np.isnan(x) or np.isnan(y):
                return None

            # If coordinates are normalized, project to pixels.
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x *= frame_w
                y *= frame_h
            return x, y

        return None

    def _collect_image_points(self, landmarks: Sequence[object], frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        frame_h, frame_w = frame_shape[:2]
        image_points: List[Tuple[float, float]] = []

        for idx in ANCHOR_INDEX_MAP.values():
            if idx >= len(landmarks):
                return None
            xy = self._extract_xy(landmarks[idx], frame_w=frame_w, frame_h=frame_h)
            if xy is None:
                return None
            image_points.append(xy)

        if len(image_points) < 6:
            return None

        return np.array(image_points, dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_euler_degrees(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        # Use arcsin on the off-diagonal element for pitch — avoids the
        # ±180° flip that arctan2 produces when the face is near-frontal.
        pitch = float(np.degrees(np.arcsin(-np.clip(rotation_matrix[2, 1], -1.0, 1.0))))
        yaw   = float(np.degrees(np.arctan2(rotation_matrix[2, 0], rotation_matrix[2, 2])))
        roll  = float(np.degrees(np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1])))

        # Clamp to physically meaningful webcam ranges
        pitch = float(np.clip(pitch, -90.0, 90.0))
        yaw   = float(np.clip(yaw,   -90.0, 90.0))
        roll  = float(np.clip(roll,  -90.0, 90.0))

        return pitch, yaw, roll

    def _smooth_angles(self, angles: np.ndarray) -> np.ndarray:
        if self.state.last_smoothed is None:
            self.state.last_smoothed = angles
            return angles

        smoothed = self.state.alpha * angles + (1.0 - self.state.alpha) * self.state.last_smoothed
        self.state.last_smoothed = smoothed
        return smoothed

    def _center_pitch(self, timestamp: float, pitch_deg: float) -> float:
        # Remove slow drift / camera bias but freeze baseline during active movement.
        if self.state.pitch_baseline is None:
            self.state.pitch_baseline = pitch_deg
            self.state.prev_pitch_for_baseline = pitch_deg
            self.state.prev_pitch_time = timestamp
            return 0.0
        
        prev_pitch = self.state.prev_pitch_for_baseline
        prev_time = self.state.prev_pitch_time
        velocity_dps = 0.0
        if prev_pitch is not None and prev_time is not None:
            dt = max(timestamp - prev_time, 1e-6)
            velocity_dps = abs((pitch_deg - prev_pitch) / dt)

        baseline = float(self.state.pitch_baseline)
        offset = abs(pitch_deg - baseline)
        in_active_motion = (
            offset >= self.state.baseline_freeze_offset_deg
            or velocity_dps >= self.state.baseline_freeze_velocity_dps
        )

        if not in_active_motion:
            self.state.pitch_baseline = baseline + self.state.pitch_baseline_alpha * (pitch_deg - baseline)
        else:
            self.state.pitch_baseline = baseline

        self.state.prev_pitch_for_baseline = pitch_deg
        self.state.prev_pitch_time = timestamp

        #return pitch_deg - float(self.state.pitch_baseline)
        return pitch_deg

    def _detect_nod(self, timestamp: float, pitch_deg: float) -> bool:
        if timestamp < self.state.nod_block_until_time:
            return False

        cooldown_ok = (timestamp - self.state.last_nod_event_time) > self.state.nod_cooldown_sec

        if self.state.nod_state == "WAIT_NEUTRAL":
            if abs(pitch_deg) <= self.state.nod_rearm_neutral_band_deg:
                self.state.nod_state = "NEUTRAL"
            return False

        # Bidirectional FSM:
        # NEUTRAL -> GOING_DOWN -> REBOUNDING_UP -> COMPLETE
        # NEUTRAL -> GOING_UP -> REBOUNDING_DOWN -> COMPLETE
        if self.state.nod_state == "NEUTRAL":
            if pitch_deg <= -self.state.nod_threshold_deg:
                self.state.nod_state = "GOING_DOWN"
                self.state.nod_start_time = timestamp
                self.state.nod_extreme_pitch = pitch_deg
            elif pitch_deg >= self.state.nod_threshold_deg:
                self.state.nod_state = "GOING_UP"
                self.state.nod_start_time = timestamp
                self.state.nod_extreme_pitch = pitch_deg
            return False

        if self.state.nod_state == "GOING_DOWN":
            self.state.nod_extreme_pitch = min(self.state.nod_extreme_pitch, pitch_deg)

            if self.state.nod_start_time is not None and (timestamp - self.state.nod_start_time) > self.state.nod_window_sec:
                self.state.nod_state = "NEUTRAL"
                self.state.nod_start_time = None
                return False

            rebound_from_extreme = pitch_deg - self.state.nod_extreme_pitch
            if rebound_from_extreme >= self.state.nod_rebound_deg:
                self.state.nod_state = "REBOUNDING_UP"
            return False

        if self.state.nod_state == "GOING_UP":
            self.state.nod_extreme_pitch = max(self.state.nod_extreme_pitch, pitch_deg)

            if self.state.nod_start_time is not None and (timestamp - self.state.nod_start_time) > self.state.nod_window_sec:
                self.state.nod_state = "NEUTRAL"
                self.state.nod_start_time = None
                return False

            rebound_from_extreme = self.state.nod_extreme_pitch - pitch_deg
            if rebound_from_extreme >= self.state.nod_rebound_deg:
                self.state.nod_state = "REBOUNDING_DOWN"
            return False

        if self.state.nod_state == "REBOUNDING_UP":
            if self.state.nod_start_time is not None and (timestamp - self.state.nod_start_time) > self.state.nod_window_sec:
                self.state.nod_state = "NEUTRAL"
                self.state.nod_start_time = None
                return False

            rebound_from_extreme = pitch_deg - self.state.nod_extreme_pitch
            nod_completed = rebound_from_extreme >= self.state.nod_min_rebound_total_deg

            if nod_completed and cooldown_ok:
                self.state.nod_state = "WAIT_NEUTRAL"
                self.state.nod_start_time = None
                self.state.last_nod_event_time = timestamp
                self.state.nod_block_until_time = timestamp + self.state.nod_cooldown_sec
                self.state.nod_event_timestamps.append(timestamp)
                with self.nod_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{timestamp:.3f},nod\n")
                return True

            # If rebound fails and dips lower than prior trough, restart with hysteresis.
            if pitch_deg < self.state.nod_extreme_pitch - 1.5:
                self.state.nod_state = "GOING_DOWN"
                self.state.nod_extreme_pitch = pitch_deg
            return False

        if self.state.nod_state == "REBOUNDING_DOWN":
            if self.state.nod_start_time is not None and (timestamp - self.state.nod_start_time) > self.state.nod_window_sec:
                self.state.nod_state = "NEUTRAL"
                self.state.nod_start_time = None
                return False

            rebound_from_extreme = self.state.nod_extreme_pitch - pitch_deg
            nod_completed = rebound_from_extreme >= self.state.nod_min_rebound_total_deg

            if nod_completed and cooldown_ok:
                self.state.nod_state = "WAIT_NEUTRAL"
                self.state.nod_start_time = None
                self.state.last_nod_event_time = timestamp
                self.state.nod_block_until_time = timestamp + self.state.nod_cooldown_sec
                self.state.nod_event_timestamps.append(timestamp)
                with self.nod_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{timestamp:.3f},nod\n")
                return True

            # If rebound fails and rises higher than prior crest, restart with hysteresis.
            if pitch_deg > self.state.nod_extreme_pitch + 1.5:
                self.state.nod_state = "GOING_UP"
                self.state.nod_extreme_pitch = pitch_deg

        return False

    def _detect_head_droop(self, timestamp: float, pitch_deg: float) -> bool:
        if pitch_deg < self.state.droop_threshold_deg:
            if self.state.droop_start_time is None:
                self.state.droop_start_time = timestamp
            if (timestamp - self.state.droop_start_time) >= self.state.droop_duration_sec:
                return True
            return False

        self.state.droop_start_time = None
        return False

    def _compute_head_variance(self, timestamp: float, pitch: float, yaw: float, roll: float) -> float:
        self.state.pose_history.append((timestamp, pitch, yaw, roll))

        while self.state.pose_history and (timestamp - self.state.pose_history[0][0]) > self.state.variance_window_sec:
            self.state.pose_history.popleft()

        if len(self.state.pose_history) < 2:
            return 0.0

        arr = np.array([[p, y, r] for _, p, y, r in self.state.pose_history], dtype=np.float64)
        per_axis_var = np.var(arr, axis=0)
        return float(np.mean(per_axis_var))

    @staticmethod
    def _draw_pose_axes(
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        origin_px: Tuple[int, int],
    ) -> None:
        axis_len = 40.0
        axis_3d = np.float32(
            [
                [axis_len, 0.0, 0.0],
                [0.0, -axis_len, 0.0],
                [0.0, 0.0, axis_len],
            ]
        )

        projected, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2).astype(int)
        ox, oy = origin_px

        # X-axis red, Y-axis green, Z-axis blue.
        cv2.line(frame, (ox, oy), tuple(projected[0]), (0, 0, 255), 2)
        cv2.line(frame, (ox, oy), tuple(projected[1]), (0, 255, 0), 2)
        cv2.line(frame, (ox, oy), tuple(projected[2]), (255, 0, 0), 2)

    def _append_csv_row(
        self,
        timestamp: float,
        pitch: float,
        yaw: float,
        roll: float,
        head_var: float,
        nod_detected: bool,
        head_droop: bool,
        force: bool = False,
    ) -> None:
        if not force and (timestamp - self.state.last_csv_log_time) < 1.0:
            return

        self.state.last_csv_log_time = timestamp
        row = pd.DataFrame(
            [
                {
                    "timestamp": round(timestamp, 3),
                    "pitch": round(float(pitch), 3),
                    "yaw": round(float(yaw), 3),
                    "roll": round(float(roll), 3),
                    "head_var": round(float(head_var), 6),
                    "nod_detected": bool(nod_detected),
                    "head_droop": bool(head_droop),
                }
            ]
        )
        row.to_csv(self.log_path, mode="a", index=False, header=False)

    def get_head_pose_features(
        self,
        landmarks: Optional[Sequence[object]],
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Optional[Dict[str, float | bool]]:
        """
        Returns per-frame head pose features.

        Output keys:
          pitch, yaw, roll, head_var, nod_detected, head_droop
        """
        if landmarks is None or len(landmarks) == 0:
            return None

        image_points = self._collect_image_points(landmarks, frame.shape)
        if image_points is None or image_points.shape[0] < 6:
            return None

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rot_matrix, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler_degrees(rot_matrix)

        smoothed = self._smooth_angles(np.array([pitch, yaw, roll], dtype=np.float64))
        pitch_s, yaw_s, roll_s = smoothed.tolist()

        timestamp = time.time()
        pitch_centered = self._center_pitch(timestamp, pitch_s)

        nod_event = self._detect_nod(timestamp, pitch_centered)
        if nod_event:
            self.state.nod_latch_until_time = timestamp + self.state.nod_latch_sec

        nod_detected = timestamp <= self.state.nod_latch_until_time
        head_droop = self._detect_head_droop(timestamp, pitch_centered)
        head_var = self._compute_head_variance(timestamp, pitch_s, yaw_s, roll_s)

        nose_tip = tuple(image_points[0].astype(int))
        self._draw_pose_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, origin_px=nose_tip)

        self._append_csv_row(
            timestamp=timestamp,
            pitch=pitch_s,
            yaw=yaw_s,
            roll=roll_s,
            head_var=head_var,
            nod_detected=nod_detected,
            head_droop=head_droop,
            force=nod_event,
        )

        return {
            "pitch": float(pitch_s),
            "yaw": float(yaw_s),
            "roll": float(roll_s),
            "head_var": float(head_var),
            "nod_detected": bool(nod_detected),
            "head_droop": bool(head_droop),
        }


_DEFAULT_EXTRACTOR: Optional[HeadPoseFeatureExtractor] = None


def get_head_pose_features(
    landmarks: Optional[Sequence[object]],
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[Dict[str, float | bool]]:
    """
    Public module-level API required by the pipeline.
    """
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = HeadPoseFeatureExtractor()

    return _DEFAULT_EXTRACTOR.get_head_pose_features(
        landmarks=landmarks,
        frame=frame,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )


def _build_default_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    # Approximate webcam intrinsics; 0.9 * max dimension is usually closer than width alone.
    focal_length = float(max(frame_width, frame_height) * 0.9)
    center = (frame_width / 2.0, frame_height / 2.0)
    return np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _draw_feature_overlay(frame: np.ndarray, features: Optional[Dict[str, float | bool]]) -> None:
    if features is None:
        cv2.putText(
            frame,
            "No face / insufficient landmarks",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    lines = [
        f"Pitch: {features['pitch']:.2f}",
        f"Yaw: {features['yaw']:.2f}",
        f"Roll: {features['roll']:.2f}",
        f"Head Var: {features['head_var']:.5f}",
        f"Nod: {features['nod_detected']}",
        f"Droop: {features['head_droop']}",
    ]

    y = 30
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        y += 24


def _draw_landmark_overlay(frame: np.ndarray, landmarks: Optional[Sequence[object]]) -> None:
    if landmarks is None or len(landmarks) == 0:
        return

    h, w = frame.shape[:2]
    for lm in landmarks:
        if not hasattr(lm, "x") or not hasattr(lm, "y"):
            continue
        x = int(float(lm.x) * w)
        y = int(float(lm.y) * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)


def _is_low_light(frame: np.ndarray, brightness_threshold: float = 45.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < brightness_threshold


def _run_standalone_webcam() -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required for standalone mode. Install with: pip install mediapipe"
        ) from exc

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Installed mediapipe build does not expose Face Mesh solutions API. "
            "Use a mediapipe build with FaceMesh support for standalone mode, "
            "or call get_head_pose_features from your existing pipeline landmarks."
        )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            camera_matrix = _build_default_camera_matrix(w, h)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            face_landmarks = None
            if result.multi_face_landmarks:
                face_landmarks = result.multi_face_landmarks[0].landmark

            low_light = _is_low_light(frame)
            _draw_landmark_overlay(frame, face_landmarks)

            features = get_head_pose_features(
                landmarks=face_landmarks,
                frame=frame,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
            _draw_feature_overlay(frame, features)

            if low_light:
                cv2.putText(
                    frame,
                    "Low light detected: increase illumination",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Head Pose Standalone Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_standalone_webcam()
