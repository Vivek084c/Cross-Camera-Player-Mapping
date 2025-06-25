import cv2
import os

# Input MJPEG video path
input_path = "data/cam2/camera2_20080409T164100+02.mjpeg"

# Output MP4 video path
output_path = "output_converted.mp4"

# Open input video
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("❌ Failed to open input video.")
    exit()

# Get original properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'h264' if available
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read and write frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)  # Write frame to output video

# Release resources
cap.release()
out.release()

print(f"✅ Converted successfully: {output_path}")