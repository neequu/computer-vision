# Object Detection and Segmentation Evaluation

This project evaluates three state-of-the-art object detection models (YOLOv8, Faster R-CNN, and Mask R-CNN) on video processing tasks, comparing their accuracy and inference speed.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- torchvision
- tabulate

## Setup

1. Clone the repository
2. Install dependencies: `pip install ultralytics torch torchvision opencv-python tabulate`
3. Place your test video file named `test.mp4` in the root directory

## Usage

Run the main evaluation script:

```bash
python main.py
```