from ultralytics import YOLO
import cv2
import os

# Load your custom YOLO model
model = YOLO('model/best.pt')

# Open the video
video_path = 'data/cam1/camera1_20080409T164100+02.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
max_frames = 5

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(source=frame, conf=0.25, save=False, verbose=False)
    
    # Get annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Save frame to disk
    save_path = os.path.join(output_dir, f"frame_{frame_count + 1}.jpg")
    cv2.imwrite(save_path, annotated_frame)

    print(f"✅ Saved: {save_path}")
    frame_count += 1

cap.release()