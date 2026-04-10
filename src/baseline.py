"""
baseline.py — Personalised alertness baseline with three-layer protection.

Layer 1: Absolute floor thresholds (population-level, always enforced)
Layer 2: Calibration quality validation (rejects drowsy calibration sessions)
Layer 3: Population prior blending (prevents extreme personal drift)

Usage:
    from baseline import BaselineMonitor
    monitor = BaselineMonitor(user_id="sonia")

    # During calibration (first 5 min):
    monitor.update(second_features_dict)

    # After calibration:
    result = monitor.score(second_features_dict)
    print(result["deviation_score"])   # 0.0–1.0
    print(result["alerts"])            # list of triggered thresholds
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ─── Population-level priors (from drowsiness literature) ────────────────────
# These represent a typical alert person.
# Used for blending to prevent personal baseline from drifting too far.

POPULATION_PRIOR: Dict[str, float] = {
    "ear_mean":    0.30,
    "ear_min":     0.22,
    "perclos":     0.02,
    "blink_count": 0.25,   # per second
    "mar_mean":    0.32,
    "mar_max":     0.40,
    "pitch_mean":  0.0,
    "pitch_var":   5.0,
    "yaw_mean":    0.0,
    "roll_mean":   0.0,
    "head_var":    0.0005,
}

# Weight given to population prior vs personal baseline (0.0 = fully personal)
PRIOR_WEIGHT = 0.25

# ─── Absolute thresholds (always fire regardless of personal baseline) ────────
# If ANY of these trigger, fatigue is flagged immediately.
# These cannot be overridden by a drowsy baseline.

ABSOLUTE_THRESHOLDS: Dict[str, tuple] = {
    # (feature, operator, threshold)
    "perclos":     (">=", 0.20),   # >20% eye closure in any second
    "ear_min":     ("<=", 0.08),   # eye stayed this closed (not just a blink)
    "ear_mean":    ("<=", 0.18),   # average EAR dangerously low
}

# ─── Calibration validation thresholds ───────────────────────────────────────
# If calibration data looks like this, reject it as possibly drowsy.

CALIBRATION_LIMITS: Dict[str, tuple] = {
    "ear_mean":    ("<=", 0.22),   # too low = already drowsy
    "perclos":     (">=", 0.10),   # too much eye closure
    "blink_count": ("<=", 0.2),    # too few blinks per second
}

# Features used for z-score deviation (continuous features only)
DEVIATION_FEATURES = [
    "ear_mean", "ear_min", "perclos", "blink_count",
    "mar_mean", "mar_max", "pitch_mean", "pitch_var",
    "yaw_mean", "roll_mean", "head_var",
]

# Feature weights for composite deviation score
# Higher = more important for fatigue detection
FEATURE_WEIGHTS: Dict[str, float] = {
    "ear_mean":    3.0,
    "ear_min":     2.5,
    "perclos":     3.5,
    "blink_count": 1.5,
    "mar_mean":    1.0,
    "mar_max":     1.5,
    "pitch_mean":  1.0,
    "pitch_var":   1.0,
    "yaw_mean":    0.5,
    "roll_mean":   0.5,
    "head_var":    1.0,
}


@dataclass
class BaselineProfile:
    """Stores per-user baseline statistics."""
    user_id:    str
    mean:       Dict[str, float] = field(default_factory=dict)
    std:        Dict[str, float] = field(default_factory=dict)
    n_samples:  int = 0
    valid:      bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class BaselineMonitor:
    """
    Personalised fatigue monitor.

    Typical flow:
        monitor = BaselineMonitor(user_id="sonia")

        # Phase 1 — calibration (feed ~300 seconds of alert data)
        for row in alert_session:
            monitor.update(row)
        ok = monitor.finalise_calibration()

        # Phase 2 — runtime scoring
        result = monitor.score(current_row)
    """

    EMA_ALPHA          = 0.15    # smoothing for runtime deviation score
    MIN_CALIBRATION_N  = 60      # minimum seconds needed for valid baseline
    ONLINE_UPDATE_RATE = 0.001   # how fast baseline drifts toward new data

    def __init__(self, user_id: str = "default", profile_dir: str = "../models"):
        self.user_id     = user_id
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self._calibration_buffer: List[Dict] = []
        self._ema_score: float = 0.0
        self._profile: Optional[BaselineProfile] = None

        # Try loading existing profile
        self._profile = self._load_profile()
        if self._profile and self._profile.valid:
            print(f"[BASELINE] Loaded existing profile for '{user_id}' "
                  f"({self._profile.n_samples} samples)")
        else:
            print(f"[BASELINE] No valid profile found for '{user_id}' — "
                  f"calibration required.")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, features: Dict) -> None:
        """
        Add one second of features to the calibration buffer.
        Call this during the calibration phase only.
        """
        self._calibration_buffer.append(features)

    def finalise_calibration(self) -> bool:
        """
        Compute baseline statistics from the calibration buffer.
        Returns True if calibration is valid, False if it was rejected.
        """
        n = len(self._calibration_buffer)
        if n < self.MIN_CALIBRATION_N:
            print(f"[BASELINE] ✗ Only {n} seconds collected — "
                  f"need at least {self.MIN_CALIBRATION_N}. Keep recording.")
            return False

        # Convert to numpy arrays per feature
        data: Dict[str, np.ndarray] = {}
        for feat in DEVIATION_FEATURES:
            vals = [row[feat] for row in self._calibration_buffer if feat in row]
            if vals:
                data[feat] = np.array(vals, dtype=np.float64)

        # ── Layer 2: calibration quality check ────────────────────────────
        issues = self._validate_calibration(data)
        if issues:
            print("[BASELINE] ⚠ Calibration may be unreliable:")
            for issue in issues:
                print(f"  → {issue}")
            print("  Recommendation: rest for 10 minutes and recalibrate.\n")
            # We still save it but mark as suspect — user is warned
            valid = False
        else:
            valid = True
            print(f"[BASELINE] ✓ Calibration valid ({n} seconds).")

        # ── Layer 3: blend with population prior ──────────────────────────
        mean: Dict[str, float] = {}
        std:  Dict[str, float] = {}

        for feat, arr in data.items():
            personal_mean = float(np.mean(arr))
            personal_std  = float(np.std(arr)) + 1e-6   # avoid zero std

            prior_mean = POPULATION_PRIOR.get(feat, personal_mean)

            # Blend: w * population + (1-w) * personal
            blended_mean = (PRIOR_WEIGHT * prior_mean +
                            (1 - PRIOR_WEIGHT) * personal_mean)

            mean[feat] = round(blended_mean, 6)
            std[feat]  = round(personal_std,  6)

        self._profile = BaselineProfile(
            user_id=self.user_id,
            mean=mean,
            std=std,
            n_samples=n,
            valid=valid,
        )
        self._save_profile()
        self._calibration_buffer = []
        return valid

    def score(self, features: Dict) -> Dict:
        """
        Score one second of features against the personal baseline.

        Returns:
            {
                "deviation_score": float 0–1,   # smoothed composite score
                "raw_score":       float 0–1,   # unsmoothed this-second score
                "alerts":          list[str],   # absolute threshold triggers
                "z_scores":        dict,        # per-feature z-scores
                "fatigue_flag":    bool,        # True if action needed
            }
        """
        alerts = self._check_absolute_thresholds(features)

        z_scores: Dict[str, float] = {}
        if self._profile and self._profile.mean:
            for feat in DEVIATION_FEATURES:
                if feat not in features:
                    continue
                val  = features[feat]
                mu   = self._profile.mean.get(feat, POPULATION_PRIOR.get(feat, 0.0))
                sig  = self._profile.std.get(feat, 1.0)
                z    = (val - mu) / (sig + 1e-6)
                z_scores[feat] = round(float(z), 3)
        else:
            # No profile yet — fall back to population prior
            for feat in DEVIATION_FEATURES:
                if feat not in features:
                    continue
                val = features[feat]
                mu  = POPULATION_PRIOR.get(feat, 0.0)
                z   = (val - mu) / 1.0
                z_scores[feat] = round(float(z), 3)

        # Weighted composite z-score → sigmoid → 0–1
        raw_score = self._composite_score(z_scores)

        # Exponential moving average smoothing
        self._ema_score = (self.EMA_ALPHA * raw_score +
                           (1 - self.EMA_ALPHA) * self._ema_score)

        # Online baseline update (very slow drift toward current data)
        self._online_update(features)

        fatigue_flag = bool(alerts) or self._ema_score > 0.65

        return {
            "deviation_score": round(self._ema_score, 4),
            "raw_score":       round(raw_score, 4),
            "alerts":          alerts,
            "z_scores":        z_scores,
            "fatigue_flag":    fatigue_flag,
        }

    @property
    def is_calibrated(self) -> bool:
        return self._profile is not None and bool(self._profile.mean)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_calibration(self, data: Dict[str, np.ndarray]) -> List[str]:
        issues = []
        checks = {
            "ear_mean":    ("Avg EAR too low — eyes may already be heavy",    "<=", 0.22),
            "perclos":     ("PERCLOS too high — too much eye closure",         ">=", 0.10),
            "blink_count": ("Blink rate very low — possible early drowsiness", "<=", 0.20),
        }
        for feat, (msg, op, threshold) in checks.items():
            if feat not in data:
                continue
            val = float(np.mean(data[feat]))
            if op == "<=" and val <= threshold:
                issues.append(f"{msg} (mean={val:.3f}, limit={threshold})")
            elif op == ">=" and val >= threshold:
                issues.append(f"{msg} (mean={val:.3f}, limit={threshold})")
        return issues

    def _check_absolute_thresholds(self, features: Dict) -> List[str]:
        """Layer 1: always-on population-level safety net."""
        triggered = []
        descriptions = {
            "perclos": "High PERCLOS — sustained eye closure",
            "ear_min": "Very low EAR minimum — prolonged eye closure detected",
            "ear_mean": "Very low average EAR — eyes barely open",
        }
        for feat, (op, threshold) in ABSOLUTE_THRESHOLDS.items():
            val = features.get(feat)
            if val is None:
                continue
            if op == ">=" and val >= threshold:
                triggered.append(descriptions.get(feat, feat))
            elif op == "<=" and val <= threshold:
                triggered.append(descriptions.get(feat, feat))
        return triggered

    def _composite_score(self, z_scores: Dict[str, float]) -> float:
        """
        Weighted average of absolute z-scores → sigmoid → 0–1.
        Features that indicate fatigue when they INCREASE get positive z.
        Features that indicate fatigue when they DECREASE get negated z.
        """
        # Features where DECREASE = fatigue (negate z so higher = more fatigued)
        decrease_features = {"ear_mean", "ear_min", "blink_count"}

        total_weight = 0.0
        weighted_sum = 0.0

        for feat, z in z_scores.items():
            w = FEATURE_WEIGHTS.get(feat, 1.0)
            # For decrease features, negative z = fatigue = we want positive contribution
            if feat in decrease_features:
                z = -z
            weighted_sum += w * z
            total_weight += w

        if total_weight == 0:
            return 0.0

        avg_z = weighted_sum / total_weight

        # Sigmoid centred at 0, scaled so z=2 → ~0.88, z=3 → ~0.95
        score = float(1.0 / (1.0 + np.exp(-0.8 * avg_z)))

        # Clip to [0, 1]
        return float(np.clip(score, 0.0, 1.0))

    def _online_update(self, features: Dict) -> None:
        """Very slowly drift baseline mean toward new observations."""
        if not self._profile or not self._profile.mean:
            return
        alpha = self.ONLINE_UPDATE_RATE
        for feat in DEVIATION_FEATURES:
            if feat not in features or feat not in self._profile.mean:
                continue
            current = self._profile.mean[feat]
            new_val = features[feat]
            self._profile.mean[feat] = round(
                current + alpha * (new_val - current), 6
            )

    # ── Profile persistence ───────────────────────────────────────────────────

    def _profile_path(self) -> Path:
        return self.profile_dir / f"baseline_{self.user_id}.json"

    def _save_profile(self) -> None:
        if not self._profile:
            return
        with open(self._profile_path(), "w") as f:
            json.dump(asdict(self._profile), f, indent=2)

    def _load_profile(self) -> Optional[BaselineProfile]:
        path = self._profile_path()
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return BaselineProfile(**data)
        except Exception as e:
            print(f"[BASELINE] Could not load profile: {e}")
            return None


# ─── Convenience: build baseline from existing CSV sessions ──────────────────

def build_baseline_from_csv(
    alert_csv_paths: List[str],
    user_id: str = "default",
    profile_dir: str = "../models",
) -> BaselineMonitor:
    """
    Build and save a baseline profile from one or more alert session CSVs.

    Args:
        alert_csv_paths: List of paths to alert session CSV files
        user_id:         Username for the profile
        profile_dir:     Where to save the JSON profile

    Returns:
        Calibrated BaselineMonitor ready for scoring

    Example:
        monitor = build_baseline_from_csv(
            ["../data/sessions/alert_1.csv",
             "../data/sessions/alert_2.csv"],
            user_id="sonia"
        )
    """
    import pandas as pd

    monitor = BaselineMonitor(user_id=user_id, profile_dir=profile_dir)

    all_rows = []
    for path in alert_csv_paths:
        if not os.path.exists(path):
            print(f"[BASELINE] Warning: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        all_rows.append(df)
        print(f"[BASELINE] Loaded {len(df)} rows from {path}")

    if not all_rows:
        raise FileNotFoundError("No alert CSV files found.")

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"[BASELINE] Total calibration rows: {len(combined)}")

    for _, row in combined.iterrows():
        monitor.update(row.to_dict())

    success = monitor.finalise_calibration()
    if success:
        print(f"[BASELINE] Profile saved → {monitor._profile_path()}")
    return monitor


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    alert_csvs = sorted(glob.glob("../data/sessions/alert*.csv"))
    if not alert_csvs:
        print("No alert CSVs found in ../data/sessions/")
        print("Run: python capture.py --session alert_1")
    else:
        print(f"Found alert sessions: {alert_csvs}")
        monitor = build_baseline_from_csv(alert_csvs, user_id="sonia")

        if monitor.is_calibrated:
            print("\n── Baseline profile ──")
            for feat in ["ear_mean", "perclos", "blink_count", "pitch_mean"]:
                mu  = monitor._profile.mean.get(feat, "N/A")
                sig = monitor._profile.std.get(feat, "N/A")
                print(f"  {feat:15s}  mean={mu:.4f}  std={sig:.4f}")

            # Simulate scoring a tired-looking row
            print("\n── Scoring a simulated tired row ──")
            tired_row = {
                "ear_mean": 0.15, "ear_min": 0.05, "perclos": 0.80,
                "blink_count": 0, "mar_mean": 0.35, "mar_max": 0.45,
                "pitch_mean": -18.0, "pitch_var": 25.0,
                "yaw_mean": 0.0, "roll_mean": 0.0, "head_var": 0.002,
            }
            result = monitor.score(tired_row)
            print(f"  deviation_score : {result['deviation_score']}")
            print(f"  fatigue_flag    : {result['fatigue_flag']}")
            print(f"  alerts          : {result['alerts']}")

            print("\n── Scoring a simulated alert row ──")
            alert_row = {
                "ear_mean": 0.31, "ear_min": 0.20, "perclos": 0.02,
                "blink_count": 1, "mar_mean": 0.32, "mar_max": 0.38,
                "pitch_mean": 2.0, "pitch_var": 4.0,
                "yaw_mean": 0.0, "roll_mean": 0.0, "head_var": 0.0003,
            }
            result = monitor.score(alert_row)
            print(f"  deviation_score : {result['deviation_score']}")
            print(f"  fatigue_flag    : {result['fatigue_flag']}")
            print(f"  alerts          : {result['alerts']}")