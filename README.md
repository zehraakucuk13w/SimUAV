# SimUAV — Air-to-Air UAV Detection & Tracking

> Autonomous air-to-air UAV detection and tracking system built with **Unreal Engine 4.27**, **AirSim**, and **YOLOv8**.

<img width="1575" height="409" alt="image" src="https://github.com/user-attachments/assets/bbc3f1aa-321e-4d52-ab46-bf296a7841c7" />

---

## About The Project

SimUAV is a simulation-based system that detects and pursues enemy UAVs in real time. A hunter drone equipped with a forward-facing camera uses a custom-trained YOLOv8 model to detect a target drone, then autonomously follows it using PID-based velocity control. The entire pipeline — from synthetic data generation to model training and real-time inference — runs within an Unreal Engine + AirSim environment.

### How It Works

```
Unreal Engine (AirSim)  →  Drone Camera Feed  →  YOLOv8 Detection  →  PID Pursuit Controller
        ↑                                                                        |
        └────────────────── Velocity Commands ──────────────────────────────────┘
```

1. **Two drones** spawn in the simulation: a hunter (`Drone1`) and an enemy (`Drone2`).
2. The enemy drone follows a predefined waypoint path autonomously.
3. The hunter drone captures frames from its front camera.
4. **YOLOv8n** runs inference on each frame to detect the enemy UAV.
5. A **PID controller** converts bounding-box error into yaw, pitch, and throttle commands.
6. If no target is detected, the hunter enters a rotating search pattern.

---

## Features

- **YOLOv8-based real-time UAV detection** — custom-trained nano model for fast inference
- **PID-controlled pursuit system** — smooth tracking via proportional yaw/pitch/throttle control
- **Multi-drone AirSim simulation** — two fully autonomous drones, no manual input required
- **Synthetic dataset generation** — 1000 auto-labeled images via segmentation masks (see below)

---

## Synthetic Dataset Generation

Training data was generated entirely within the simulation using AirSim's segmentation API:

1. **`synthetic_data_generator.py`** teleports both drones to randomized positions and orientations.
2. For each frame, it captures an **RGB image** and a **segmentation mask** from the hunter's camera.
3. The segmentation mask isolates `Drone2` (enemy), and **OpenCV contour detection** extracts bounding boxes.
4. Bounding boxes are converted to **YOLO format** (`class cx cy w h`) and saved as `.txt` label files.
5. Frames where the target is too small, too large, or not visible are automatically discarded.

This produces a clean, auto-annotated dataset of **1000 valid image–label pairs** — no manual labeling required.

**`prepare_dataset.py`** then splits this into train/val/test sets (80/15/5) and generates the `data.yaml` config for YOLOv8 training.

---

## Model Weights

Download the trained YOLOv8 model:

**[Download best.pt](https://drive.google.com/file/d/1vGZjd9YYJ09w-7NGtMdplejxxcuochL7/view?usp=sharing)**

Place `best.pt` in the project root directory.

---

## Project Structure

| File | Description |
|------|-------------|
| `airsim_tracker.py` | Real-time detection and autonomous pursuit |
| `record_flight.py` | Record video from the hunter drone's camera |
| `detect_on_video.py` | Run YOLOv8 detection on a recorded video |
| `synthetic_data_generator.py` | Generate auto-labeled training data via AirSim |
| `prepare_dataset.py` | Split dataset into train/val/test + create `data.yaml` |
| `train_yolo.ipynb` | Google Colab notebook for YOLOv8 training |

---

## Setup

### Requirements

```
pip install airsim ultralytics opencv-python numpy
```

### AirSim Settings

Copy the following to `C:\Users\<username>\Documents\AirSim\settings.json`

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "FlyWithMe",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": -2
    },
    "Drone2": {
      "VehicleType": "SimpleFlight",
      "X": 4, "Y": 0, "Z": -2
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 640,
        "Height": 640,
        "FOV_Degrees": 90
      },
      {
        "ImageType": 5,
        "Width": 640,
        "Height": 640,
        "FOV_Degrees": 90
      }
    ]
  }
}
```

---

## Usage

### Real-time Tracking
1. Open the Unreal Engine project and press **Play**
2. Run: `python airsim_tracker.py`
3. Press **Q** to quit

### Video Demo
1. Record flight footage: `python record_flight.py`
2. Run detection on the video: `python detect_on_video.py`

---

## Training

1. Generate data: `python synthetic_data_generator.py`
2. Prepare dataset: `python prepare_dataset.py`
3. Zip `yolo_dataset/` and upload to Google Drive
4. Open `train_yolo.ipynb` in Google Colab
5. Run all cells
6. Download `best.pt` and place it in the project root

---

## Results

| Metric | Score |
|--------|-------|
| mAP50 | 0.995 |
| Precision | 1.000 |
| Recall | 0.998 |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Simulation | Unreal Engine 4.27 + AirSim |
| Detection | YOLOv8n (Ultralytics) |
| Control | PID velocity controller |
| Training | Google Colab (GPU) |
| Language | Python 3.11 |
