# Cross-Camera Player Tracking System

## Overview
A computer vision system that detects and tracks players across multiple camera views (tacticam and broadcast) using YOLOv8 for player detection and Local Binary Patterns (LBP) for cross-camera player matching.

## Features
- Dual-camera processing (tacticam + broadcast views)
- Real-time player detection using YOLOv8
- Player feature extraction using Local Binary Patterns (LBP)
- Cross-camera player matching algorithm
- Visualization with global player IDs
- Data logging for analysis

## System Requirements
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- FFmpeg (for video processing)
- 8GB+ RAM (16GB recommended for HD video processing)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Cross-Camera-Player-Mapping.git
cd Cross-Camera-Player-Mapping
```

### 2. create a venv
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4.Prepare videos
```bash
mkdir -p data/inputs models
```

## Place your video files in data/inputs/:
### tacticam.mp4
### broadcast.mp4

### 4.Prepare model 
```bash
mkdir -p models
```
### Place your YOLO model (best.pt) in models/

### Edit the config/config.yaml as per structure


### run the main script
```bash
python src/main.py
```
