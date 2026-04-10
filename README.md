# 🧠 Personalized Fatigue Detection System

A real-time fatigue prediction system that learns your personal alertness baseline and detects cognitive decline **before micro-sleep occurs** — using just a webcam.

---

## 🚀 Key Features

- 👁️ Eye tracking using EAR (Eye Aspect Ratio)
- 😮 Yawn detection using MAR (Mouth Aspect Ratio)
- 🧍 Head pose tracking (pitch, yaw, roll)
- 🧠 Personalized baseline calibration
- ⚡ GRU-based temporal fatigue prediction
- 📊 Real-time monitoring dashboard (Streamlit)
- 🚨 Early fatigue alerts (before micro-sleep)

---

## 🧠 How It Works

1. **Capture** → Webcam extracts facial landmarks (MediaPipe)
2. **Feature Extraction** → EAR, MAR, PERCLOS, head pose
3. **Baseline Learning** → Learns your normal alert state
4. **ML Model (GRU)** → Detects temporal fatigue patterns
5. **Dashboard** → Displays real-time fatigue score

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- PyTorch
- Streamlit
- NumPy / Pandas

---

