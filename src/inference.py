"""
inference.py — Real-time GRU fatigue inference engine.

Loads the trained model and scaler, maintains a rolling 30-second
feature buffer, and outputs a fatigue probability every 5 seconds.

Usage:
    from inference import FatigueInferenceEngine
    engine = FatigueInferenceEngine()
    prob   = engine.update(feature_dict)   # call every second
"""

import json
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

print("STARTED")
import sys
print("ARGS:", sys.argv)
# ─── Model definition (must match training exactly) ───────────────────────────

class FatigueGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last    = out[:, -1, :]
        return self.head(last).squeeze(1)


# ─── Inference engine ─────────────────────────────────────────────────────────

class FatigueInferenceEngine:
    """
    Maintains a rolling window of feature vectors and runs the GRU
    every INFERENCE_INTERVAL seconds.

    Args:
        model_dir:          Directory containing model files
        decision_threshold: Override the saved threshold (None = use saved)
        inference_interval: How often to run the model in seconds
    """

    INFERENCE_INTERVAL = 5     # run GRU every 5 seconds
    SLOPE_WINDOW       = 5     # must match training

    def __init__(
        self,
        model_dir: str = "../models",
        decision_threshold: Optional[float] = None,
        inference_interval: int = INFERENCE_INTERVAL,
    ):
        self.model_dir          = Path(model_dir)
        self.inference_interval = inference_interval
        self._seconds_since_inference = 0
        self._last_probability  = 0.0

        # Load config
        config_path = self.model_dir / "gru_config.json"
        scaler_path = self.model_dir / "scaler_params.json"
        model_path  = self.model_dir / "gru_fatigue_best.pt"

        for p in [config_path, scaler_path, model_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    "Download from Colab Cell 17 and place in models/"
                )

        with open(config_path) as f:
            self.config = json.load(f)

        with open(scaler_path) as f:
            scaler_data = json.load(f)

        self.feature_cols   = scaler_data['feature_cols']
        self.scaler_mean    = np.array(scaler_data['mean'],  dtype=np.float32)
        self.scaler_std     = np.array(scaler_data['std'],   dtype=np.float32)
        self.window_size    = self.config['window_size']

        # Use saved threshold but override 0.001 artifacts — floor at 0.40
        saved_threshold = self.config.get('decision_threshold', 0.5)
        if decision_threshold is not None:
            self.threshold = decision_threshold
        elif saved_threshold < 0.05:
            # Artifact from overfitting — use sensible default
            self.threshold = 0.50
            print(f"[INFERENCE] Saved threshold ({saved_threshold:.3f}) too low "
                  f"— using 0.50 instead")
        else:
            self.threshold = saved_threshold

        print(f"[INFERENCE] Decision threshold: {self.threshold:.3f}")

        # Load model
        self.device = torch.device('cpu')   # inference on CPU
        self.model  = FatigueGRU(
            input_size  = self.config['input_size'],
            hidden_size = self.config['hidden_size'],
            num_layers  = self.config['num_layers'],
            dropout     = self.config['dropout'],
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"[INFERENCE] Model loaded — {sum(p.numel() for p in self.model.parameters()):,} params")

        # Rolling feature buffer
        self._buffer: deque = deque(maxlen=self.window_size)

        # Slope feature history (for temporal features added during training)
        self._slope_history: dict = {
            feat: deque(maxlen=self.SLOPE_WINDOW)
            for feat in ['ear_mean', 'perclos', 'pitch_mean']
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, features: dict) -> Optional[float]:
        """
        Feed one second of aggregated features.
        Returns fatigue probability (0–1) every INFERENCE_INTERVAL seconds,
        or None if the buffer isn't full yet or it's not time to infer.

        Args:
            features: dict from FrameAggregator.get_second_features()

        Returns:
            float probability or None
        """
        # Update slope history
        for feat in self._slope_history:
            if feat in features:
                self._slope_history[feat].append(features[feat])

        # Build slope features
        slope_feats = {}
        for feat in self._slope_history:
            hist = list(self._slope_history[feat])
            if len(hist) >= 2:
                slope = np.polyfit(range(len(hist)), hist, 1)[0]
            else:
                slope = 0.0
            slope_feats[f'{feat}_slope'] = float(slope)

        # Merge into full feature vector
        full_features = {**features, **slope_feats}

        # Build ordered array matching training feature_cols
        try:
            row = np.array(
                [full_features.get(col, 0.0) for col in self.feature_cols],
                dtype=np.float32
            )
        except Exception:
            return None

        # Normalise using saved scaler params
        row = (row - self.scaler_mean) / (self.scaler_std + 1e-8)

        self._buffer.append(row)
        self._seconds_since_inference += 1

        # Only infer when buffer is full AND interval has elapsed
        if (len(self._buffer) < self.window_size or
                self._seconds_since_inference < self.inference_interval):
            return None

        self._seconds_since_inference = 0
        prob = self._run_inference()
        self._last_probability = prob
        return prob

    @property
    def fatigue_probability(self) -> float:
        """Most recent fatigue probability (0–1)."""
        return self._last_probability

    @property
    def is_fatigued(self) -> bool:
        """True if latest probability exceeds decision threshold."""
        return self._last_probability > self.threshold

    @property
    def buffer_fill(self) -> float:
        """How full the window buffer is (0–1). Inference needs 1.0."""
        return len(self._buffer) / self.window_size

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_inference(self) -> float:
        window = np.stack(list(self._buffer), axis=0)   # (30, n_features)
        x      = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(x).item()
        return float(prob)


# ─── Offline test: replay a CSV through the model ────────────────────────────

def test_on_csv(csv_path: str, label: int, model_dir: str = "../models"):
    """
    Replay a recorded session through the model second-by-second.
    Prints probability at each inference point and final accuracy.
    """
    import pandas as pd

    engine = FatigueInferenceEngine(model_dir=model_dir)
    df     = pd.read_csv(csv_path)

    print(f"\nReplaying: {csv_path}  (true label={'TIRED' if label else 'ALERT'})")
    print(f"Rows: {len(df)}  |  Threshold: {engine.threshold:.3f}\n")

    correct = 0
    total   = 0

    for i, row in df.iterrows():
        prob = engine.update(row.to_dict())
        if prob is not None:
            predicted = 'TIRED' if prob > engine.threshold else 'ALERT'
            actual    = 'TIRED' if label else 'ALERT'
            match     = '✓' if predicted == actual else '✗'
            print(f"  t={i:>3d}s  prob={prob:.3f}  → {predicted}  {match}")
            correct += int(predicted == actual)
            total   += 1

    if total > 0:
        print(f"\nAccuracy on this session: {correct/total*100:.1f}%  ({correct}/{total})")
    else:
        print("\nNot enough data to run inference (need 30+ seconds)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',       action='store_true',
                        help='Run offline test on your session CSVs')
    parser.add_argument('--csv',        type=str, default=None,
                        help='Path to specific CSV to test')
    parser.add_argument('--label',      type=int, default=1,
                        help='True label: 0=alert, 1=tired')
    parser.add_argument('--threshold',  type=float, default=None,
                        help='Override decision threshold')
    parser.add_argument('--model-dir',  type=str, default='../models')
    args = parser.parse_args()

    if args.test:
        import glob
        tired_csvs = glob.glob('../data/sessions/tired*.csv')
        alert_csvs = glob.glob('../data/sessions/alert*.csv')

        print("=== Testing on TIRED sessions ===")
        for csv in tired_csvs[:2]:
            test_on_csv(csv, label=1, model_dir=args.model_dir)

        print("\n=== Testing on ALERT sessions ===")
        for csv in alert_csvs[:2]:
            test_on_csv(csv, label=0, model_dir=args.model_dir)

    elif args.csv:
        test_on_csv(args.csv, label=args.label, model_dir=args.model_dir)

    else:
        # Quick sanity check
        engine = FatigueInferenceEngine(
            model_dir=args.model_dir,
            decision_threshold=args.threshold
        )
        print(f"\nEngine ready. Threshold: {engine.threshold:.3f}")
        print("Run with --test to replay your session CSVs")