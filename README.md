<img width="1575" height="409" alt="image" src="https://github.com/user-attachments/assets/86684d0e-53d6-4bfb-abe7-e6a0b4710371" /># SimUAV - Air-to-Air UAV Detection & Tracking

Unreal Engine 4.27 + AirSim + YOLOv8 Simulation Project

## Project Overview
Autonomous air-to-air UAV detection and tracking system using computer vision and PID control.

## Features
- YOLOv8-based real-time UAV detection
- PID-controlled pursuit system
- Multi-drone AirSim simulation
- Custom synthetic dataset (1000 images)

## Model Weights
Download the trained YOLOv8 model:
**[Download best.pt](https://drive.google.com/file/d/1vGZjd9YYJ09w-7NGtMdplejxxcuochL7/view?usp=sharing)**

Place `best.pt` in the project root directory.

## Files
| File | Description |
|------|-------------|
| `airsim_tracker.py` | Real-time detection and pursuit |
| `record_flight.py` | Record video from drone camera |
| `detect_on_video.py` | Run detection on recorded video |
| `prepare_dataset.py` | Dataset preparation script |
| `synthetic_data_generator.py` | Synthetic data generation |
| `train_yolo.ipynb` | Google Colab training notebook |

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

## Usage

### Real-time Tracking
1. Open Unreal Engine project and press Play
2. Run: `python airsim_tracker.py`

<img width="1575" height="409" alt="image" src="https://github.com/user-attachments/assets/9509bdc9-0d96-41fc-9f85-de8c4a34909a" />

### Video Demo
1. Record: `python record_flight.py`
2. Detect: `python detect_on_video.py`

## Training
1. Upload `yolo_dataset.zip` to Google Drive
2. Open `train_yolo.ipynb` in Google Colab
3. Run all cells
4. Download `best.pt`

## Results
- mAP50: 0.995
- Precision: 1.0
- Recall: 0.998
