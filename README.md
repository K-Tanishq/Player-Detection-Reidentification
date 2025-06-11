# Player Detection and Re-Identification System

This project implements a real-time object detection and tracking system using YOLO (You Only Look Once) for detection and DeepSORT for tracking. It's particularly optimized for tracking players and referees in sports videos.

## Features

- Real-time object detection using YOLO
- Object tracking and Re-Identification using DeepSORT
- Configurable confidence thresholds
- Support for multiple object classes (players, referees)
- Real-time visualization with bounding boxes and tracking IDs
- FPS monitoring

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- DeepSORT
- CUDA (optional, for GPU acceleration)

## How to run

1. Clone the repository:
```bash
git clone https://github.com/K-Tanishq/Player-Detection-Reidentification.git
cd Player-Detection-Reidentification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
If using conda
```bash
conda create -p venv
conda activate venv
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Download the YOLO model from the following link
```
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
```

5. Give the correct model and video paths in the main.py

6. To run the full detection and tracking system:
```bash
python main.py
```

## Project Structure

```
Player-Detection-Reidentification/
├── detector.py      # YOLO-based object detector
├── tracker.py       # DeepSORT-based object tracker
├── main.py         # Main application with tracking
├── best.pt         # YOLO model weights
├── requirements.txt # Project dependencies
└── README.md       # This file
```

### Configuration

You can modify the following parameters in the code:

- Detection confidence threshold (default: 0.8)
- IOU threshold for tracking (default: 0.3)
- Valid object classes
- Tracking parameters (max_age, n_init, etc.)

## How It Works

1. **Detection**: The system uses YOLO to detect objects in each frame
2. **Filtering**: Detections are filtered based on confidence threshold
3. **Tracking**: DeepSORT assigns and maintains tracking IDs for detected objects
4. **Visualization**: Bounding boxes and tracking IDs are drawn on the frame

## Performance

The system's performance depends on:
- Input video resolution
- Hardware capabilities (CPU/GPU)
- Number of objects being tracked
- Confidence thresholds
