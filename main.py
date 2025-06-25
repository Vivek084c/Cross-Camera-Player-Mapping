from ultralytics import YOLO
import cv2
import os

# Load your custom YOLO model
model = YOLO('model/best.pt')

# Open the video
video_path = 'data/cam1/camera1_20080409T164100+02.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video.")
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
    result = results[0]
    annotated_frame = result.plot()

    # Save full annotated frame
    frame_id = frame_count + 1
    frame_dir = os.path.join(output_dir, f"frame_{frame_id}")
    os.makedirs(frame_dir, exist_ok=True)

    frame_path = os.path.join(frame_dir, f"frame_{frame_id}.jpg")
    cv2.imwrite(frame_path, annotated_frame)
    print(f"Saved annotated frame: {frame_path}")

    # Save each player crop from the current frame
    boxes = result.boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]

            if class_name.lower() == "player":  # change or remove filter if needed
                crop = frame[y1:y2, x1:x2]
                player_path = os.path.join(frame_dir, f"player_{i+1}.jpg")
                cv2.imwrite(player_path, crop)
                print(f"Saved player crop: {player_path}")

    frame_count += 1

cap.release()