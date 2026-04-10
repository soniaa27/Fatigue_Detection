import numpy as np
from scipy.spatial import distance as dist
import cv2

from head_pose import get_head_pose_features, _build_default_camera_matrix

# ─── Landmark indices ─────────────────────────────────────────────────────────

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308, 82,  312]

# Neutral fallback values returned when landmarks are missing/corrupt.
# Chosen to be "unremarkable" so they don't trigger false fatigue events.
_EAR_NEUTRAL = 0.25
_MAR_NEUTRAL = 0.30


# ─── EAR ──────────────────────────────────────────────────────────────────────

def compute_ear(landmarks, eye_indices, img_w, img_h):
    """
    Eye Aspect Ratio.  Returns _EAR_NEUTRAL on any failure.
    Normal open-eye range: ~0.25–0.40
    Blink threshold:       ~0.20
    """
    try:
        pts = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in eye_indices
        ])
        A   = dist.euclidean(pts[1], pts[5])
        B   = dist.euclidean(pts[2], pts[4])
        C   = dist.euclidean(pts[0], pts[3])
        if C < 1e-6:                          # degenerate geometry
            return _EAR_NEUTRAL
        ear = (A + B) / (2.0 * C)
        if not (0.0 < ear < 1.0):             # sanity-check range
            return _EAR_NEUTRAL
        return round(ear, 4)
    except (IndexError, AttributeError, ZeroDivisionError):
        return _EAR_NEUTRAL


# ─── MAR ──────────────────────────────────────────────────────────────────────

def compute_mar(landmarks, img_w, img_h):
    """
    Mouth Aspect Ratio.  Returns _MAR_NEUTRAL on any failure.
    Normal closed-mouth range: ~0.20–0.45
    Yawn threshold:            ~0.55–0.60
    """
    try:
        pts = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in MOUTH
        ])
        A   = dist.euclidean(pts[0], pts[1])   # vertical
        B   = dist.euclidean(pts[2], pts[3])   # horizontal
        if B < 1e-6:
            return _MAR_NEUTRAL
        mar = A / B
        if not (0.0 <= mar < 2.0):             # sanity-check range
            return _MAR_NEUTRAL
        return round(mar, 4)
    except (IndexError, AttributeError):
        return _MAR_NEUTRAL


# ─── Per-frame feature extraction ─────────────────────────────────────────────

def extract_features(landmarks, frame, img_w, img_h, camera_matrix, dist_coeffs):
    """
    Extract all per-frame features.

    Args:
        landmarks:     MediaPipe face landmark list for one face
        frame:         BGR numpy array (needed by head_pose for axis drawing)
        img_w/img_h:   frame dimensions in pixels
        camera_matrix: precomputed 3×3 camera intrinsics (compute once per session)
        dist_coeffs:   precomputed distortion coefficients (zeros for webcam)

    Returns:
        dict with 10 feature keys, or None if head pose fails completely.
        EAR/MAR always return neutral fallbacks rather than None.
    """
    ear_left  = compute_ear(landmarks, LEFT_EYE,  img_w, img_h)
    ear_right = compute_ear(landmarks, RIGHT_EYE, img_w, img_h)
    ear_avg   = round((ear_left + ear_right) / 2.0, 4)
    mar       = compute_mar(landmarks, img_w, img_h)

    head_feats = get_head_pose_features(landmarks, frame, camera_matrix, dist_coeffs)

    # head_pose returns None if solvePnP fails — use safe defaults
    pitch     = head_feats["pitch"]        if head_feats else 0.0
    yaw       = head_feats["yaw"]          if head_feats else 0.0
    roll      = head_feats["roll"]         if head_feats else 0.0
    head_var  = head_feats["head_var"]     if head_feats else 0.0
    nod       = head_feats["nod_detected"] if head_feats else False
    droop     = head_feats["head_droop"]   if head_feats else False

    return {
        "ear_left":     ear_left,
        "ear_right":    ear_right,
        "ear_avg":      ear_avg,
        "mar":          mar,
        "pitch":        pitch,
        "yaw":          yaw,
        "roll":         roll,
        "head_var":     head_var,
        "nod_detected": int(nod),
        "head_droop":   int(droop),
    }


# ─── Per-second aggregator ────────────────────────────────────────────────────

class FrameAggregator:
    """
    Collects per-frame feature dicts and produces one training row per second.

    Output columns (14 total):
      ear_mean, ear_min, perclos, blink_count,
      mar_mean, mar_max, yawn_flag,
      pitch_mean, pitch_var, yaw_mean, roll_mean,
      nod_detected, head_droop, head_var
    """

    BLINK_THRESHOLD = 0.20   # EAR below this = eye closed
    YAWN_THRESHOLD  = 0.55   # MAR above this = yawning

    def __init__(self):
        self.buffer      = []
        self.prev_ear    = None
        self.blink_count = 0

    def add_frame(self, features: dict):
        """Add one frame's feature dict. Silently ignores None."""
        if features is None:
            return
        ear = features.get("ear_avg", _EAR_NEUTRAL)
        # Detect blink: EAR crosses below threshold from above
        if self.prev_ear is not None:
            if self.prev_ear >= self.BLINK_THRESHOLD > ear:
                self.blink_count += 1
        self.prev_ear = ear
        self.buffer.append(features)

    def get_second_features(self) -> dict | None:
        """
        Aggregate buffer into one row. Resets buffer and blink counter.
        Returns None if buffer is empty.
        """
        # Drop any None entries that slipped through
        self.buffer = [f for f in self.buffer if f is not None]
        if not self.buffer:
            return None

        ears      = np.array([f["ear_avg"]       for f in self.buffer])
        mars      = np.array([f["mar"]            for f in self.buffer])
        pitches   = np.array([f["pitch"]          for f in self.buffer])
        yaws      = np.array([f["yaw"]            for f in self.buffer])
        rolls     = np.array([f["roll"]           for f in self.buffer])
        nods      = np.array([f["nod_detected"]   for f in self.buffer])
        droops    = np.array([f["head_droop"]     for f in self.buffer])
        head_vars = np.array([f["head_var"]       for f in self.buffer])

        result = {
            # Eye features
            "ear_mean":     round(float(np.mean(ears)),    4),
            "ear_min":      round(float(np.min(ears)),     4),
            "perclos":      round(float(np.mean(ears < self.BLINK_THRESHOLD)), 4),
            "blink_count":  self.blink_count,
            # Mouth features
            "mar_mean":     round(float(np.mean(mars)),    4),
            "mar_max":      round(float(np.max(mars)),     4),
            "yawn_flag":    int(np.max(mars) > self.YAWN_THRESHOLD),
            # Head pose — continuous
            "pitch_mean":   round(float(np.mean(pitches)), 2),
            "pitch_var":    round(float(np.var(pitches)),  2),
            "yaw_mean":     round(float(np.mean(yaws)),    2),
            "roll_mean":    round(float(np.mean(rolls)),   2),
            # Head pose — events (use any(), not mean)
            "nod_detected": int(np.any(nods)),
            "head_droop":   int(np.any(droops)),
            "head_var":     round(float(np.mean(head_vars)), 6),
        }

        # Reset for next second
        self.buffer      = []
        self.blink_count = 0
        return result