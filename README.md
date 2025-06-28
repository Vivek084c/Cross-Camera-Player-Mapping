### Project Structure

```
video_processing_pipeline/
│
├── config/
│   └── config.yaml
│
├── logs/
│   └── video_processing.log
│
├── src/
│   ├── __init__.py
│   ├── video_processor.py
│   ├── camera.py
│   └── main.py
│
└── requirements.txt
```

### Step 1: Configuration File

Create a configuration file `config.yaml` in the `config` directory to store parameters like input video paths, output video path, and logging level.

```yaml
# config/config.yaml
input:
  tacticam: "path/to/tacticam_video.mp4"
  broadcast: "path/to/broadcast_video.mp4"

output:
  video: "path/to/output_video.mp4"

logging:
  level: "DEBUG"
```

### Step 2: Requirements File

Create a `requirements.txt` file to list the necessary libraries.

```plaintext
# requirements.txt
opencv-python
opencv-python-headless
PyYAML
```

### Step 3: Camera Class

Create a `camera.py` file to handle video input from the cameras.

```python
# src/camera.py
import cv2

class Camera:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)

    def read_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.capture.release()
```

### Step 4: Video Processor Class

Create a `video_processor.py` file to handle the video processing logic.

```python
# src/video_processor.py
import cv2
import numpy as np
import logging

class VideoProcessor:
    def __init__(self, tacticam, broadcast):
        self.tacticam = tacticam
        self.broadcast = broadcast

    def process_videos(self, output_path):
        while True:
            frame_tacticam = self.tacticam.read_frame()
            frame_broadcast = self.broadcast.read_frame()

            if frame_tacticam is None or frame_broadcast is None:
                break

            # Resize frames to the same size
            frame_tacticam = cv2.resize(frame_tacticam, (640, 480))
            frame_broadcast = cv2.resize(frame_broadcast, (640, 480))

            # Create a combined frame
            combined_frame = np.hstack((frame_tacticam, frame_broadcast))

            # Add labels
            cv2.putText(combined_frame, "Tacticam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, "Broadcast", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the combined frame
            cv2.imshow('Combined Video', combined_frame)

            # Write the frame to output video
            out.write(combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.tacticam.release()
        self.broadcast.release()
        cv2.destroyAllWindows()
```

### Step 5: Main Script

Create a `main.py` file to tie everything together.

```python
# src/main.py
import cv2
import yaml
import logging
from camera import Camera
from video_processor import VideoProcessor

def setup_logging(level):
    logging.basicConfig(filename='logs/video_processing.log', level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config()
    setup_logging(getattr(logging, config['logging']['level'].upper()))

    logging.info("Starting video processing pipeline.")

    tacticam = Camera(config['input']['tacticam'])
    broadcast = Camera(config['input']['broadcast'])

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(config['output']['video'], fourcc, 30.0, (1280, 480))

    processor = VideoProcessor(tacticam, broadcast)
    processor.process_videos(out)

    out.release()
    logging.info("Video processing completed.")

if __name__ == "__main__":
    main()
```

### Step 6: Running the Project

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python src/main.py
   ```

### Conclusion

This project structure and implementation provide a solid foundation for a video processing pipeline. It uses classes for better organization, a configuration file for easy parameter management, and logging for tracking the application's behavior. You can expand this project by adding features like error handling, more complex video processing, or a GUI for user interaction.