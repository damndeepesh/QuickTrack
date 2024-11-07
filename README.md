# Real-Time Object Detection with YOLOv8

A Streamlit-based web application for real-time object detection in videos using YOLOv8. This application supports multiple YOLO models, real-time detection, object tracking, and video processing with annotated output.

## Demo

[image](https://github.com/user-attachments/assets/2bae0f1e-c98f-45c5-bd81-6a8c1de01d1b)
[image](https://github.com/user-attachments/assets/e9241364-0d1f-45c1-a8d3-2eea501357e7)

## Features

- Multiple YOLOv8 model support (Nano to XLarge)
- Real-time object detection and tracking
- Support for video files and live streams
- Unique ID tracking for detected objects
- Customizable detection confidence
- Color-coded object categories
- Downloadable processed videos
- Interactive web interface

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for better performance)

### Step 1: Clone the Repository 
- git clone <repository-url>
- cd <repository-name>

### Step 2: Create a Virtual Environment (Recommended)

## Windows
- python -m venv venv
- venv\Scripts\activate

## Linux/Mac
- python3 -m venv venv
- source venv/bin/activate

### Step 3: Install Dependencies
- pip install -r requirements.txt

## Usage

### Starting the Application
- streamlit run app.py


### Step-by-Step Guide

1. **Select a Model**:
   - Choose from available YOLOv8 models in the sidebar
   - Models range from Nano (fastest) to XLarge (most accurate)
   - Review model details in the expandable section
   - Click "Load Selected Model" to download and initialize

2. **Configure Settings**:
   - Adjust detection confidence using the slider
   - Lower values detect more objects but may increase false positives
   - Higher values are more selective but might miss some objects

3. **Input Selection**:
   - Choose between "Video File" or "Live Stream URL"
   - For video files: Upload MP4 or AVI format
   - For streams: Enter a valid stream URL

4. **Start Detection**:
   - Click "Start Detection" in the sidebar
   - Watch real-time detection with bounding boxes
   - Each object gets a unique tracking ID

5. **Download Results**:
   - Stop detection when finished
   - Download button appears automatically
   - Processed video includes all annotations

## About YOLO Models

### Available Models

1. **YOLOv8n (Nano)**:
   - Size: 6.7 MB
   - Best for: Real-time applications on CPU
   - Speed: ⚡⚡⚡⚡⚡
   - Accuracy: ⭐⭐

2. **YOLOv8s (Small)**:
   - Size: 22.4 MB
   - Best for: Balanced performance
   - Speed: ⚡⚡⚡⚡
   - Accuracy: ⭐⭐⭐

3. **YOLOv8m (Medium)**:
   - Size: 52.2 MB
   - Best for: Standard detection tasks
   - Speed: ⚡⚡⚡
   - Accuracy: ⭐⭐⭐⭐

4. **YOLOv8l (Large)**:
   - Size: 87.7 MB
   - Best for: High accuracy needs
   - Speed: ⚡⚡
   - Accuracy: ⭐⭐⭐⭐⭐

5. **YOLOv8x (XLarge)**:
   - Size: 131.7 MB
   - Best for: Maximum accuracy
   - Speed: ⚡
   - Accuracy: ⭐⭐⭐⭐⭐⭐

### Model Selection Guide

- **CPU Only**: Use Nano or Small models
- **GPU Available**: Medium to XLarge models recommended
- **Real-time Needs**: Nano or Small models
- **Accuracy Priority**: Large or XLarge models
- **Balanced**: Medium model

## Technical Details

- Built with Streamlit and OpenCV
- Uses Ultralytics YOLOv8 implementation
- Supports multiple video codecs
- Real-time frame processing and buffering
- Unique object tracking with IoU
- Color-coded object categories
- Frame buffer for smooth video writing

## Troubleshooting

1. **Video Not Loading**:
   - Check file format (MP4/AVI supported)
   - Ensure file isn't corrupted
   - Try a different video codec

2. **Slow Performance**:
   - Use a smaller YOLO model
   - Reduce input video resolution
   - Check GPU availability

3. **Detection Issues**:
   - Adjust confidence threshold
   - Try a larger YOLO model
   - Ensure good lighting in video

4. **Download Issues**:
   - Wait for processing to complete
   - Check available disk space
   - Try a different browser

## Requirements

- streamlit>=1.24.0
- opencv-python-headless>=4.7.0
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- ultralytics>=8.0.0
- python-dateutil>=2.8.2

## Acknowledgments

- YOLOv8 by Ultralytics - https://docs.ultralytics.com/models/yolov8/
- Streamlit Framework - https://streamlit.io
- OpenCV Project - https://docs.opencv.org/4.x/index.html
