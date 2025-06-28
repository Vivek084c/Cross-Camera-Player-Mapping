from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class PlayerDetection:
    player_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    crop: np.ndarray
    lbp_histogram: np.ndarray
    class_name: str
    camera_source: str  # 'tacticam' or 'broadcast'

@dataclass
class FrameData:
    frame_number: int
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[PlayerDetection]
    frame_dir: str
    camera_source: str