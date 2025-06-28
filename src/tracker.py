from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
from skimage.feature import local_binary_pattern
import cv2

@dataclass
class PlayerDetection:
    player_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    crop: np.ndarray
    lbp_histogram: np.ndarray
    class_name: str

class PlayerTracker:
    def __init__(self, model_path: str, confidence_threshold: float, target_class: str):
        self.model = YOLO(model_path)
        self.conf_thresh = confidence_threshold
        self.target_class = target_class.lower()
    
    def detect_players(self, frame: np.ndarray, radius: int = 2, method: str = 'uniform') -> list[PlayerDetection]:
        results = self.model.predict(
            source=frame, 
            conf=self.conf_thresh, 
            save=False, 
            verbose=False
        )[0]
        
        detections = []
        n_points = 8 * radius
        
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0].item())
                class_name = self.model.names[cls_id]
                
                if class_name.lower() != self.target_class:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                lbp_hist = self._compute_lbp(crop, n_points, radius, method)
                
                detections.append(PlayerDetection(
                    player_id=i,
                    bbox=(x1, y1, x2, y2),
                    crop=crop,
                    lbp_histogram=lbp_hist,
                    class_name=class_name
                ))
        
        return detections
    
    def _compute_lbp(self, crop: np.ndarray, n_points: int, radius: int, method: str) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method)
        
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_points + 3),
            range=(0, n_points + 2)
        )
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist