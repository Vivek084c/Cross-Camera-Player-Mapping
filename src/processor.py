import os
import cv2
import numpy as np
from typing import List, Optional, Dict
from models import PlayerDetection, FrameData

class FrameProcessor:
    def __init__(self, output_dir: str, save_crops: bool, save_lbp: bool):
        self.output_dir = output_dir
        self.save_crops = save_crops
        self.save_lbp = save_lbp

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[PlayerDetection],
        frame_number: int,
        camera_source: str,  # Add this parameter
        vis_config: Optional[Dict] = None
    ) -> FrameData:
        """Process a single frame from a specific camera source"""
        frame_dir = os.path.join(self.output_dir, f"{camera_source}_frame_{frame_number}")
        os.makedirs(frame_dir, exist_ok=True)

        # Set default visualization parameters if not provided
        default_vis = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'font_scale': 0.6,
            'color': (255, 0, 0),
            'thickness': 2
        }
        vis_config = {**default_vis, **(vis_config or {})}

        annotated_frame = self._annotate_frame(
            frame.copy(),
            detections,
            **vis_config
        )

        if self.save_crops or self.save_lbp:
            self._save_player_data(detections, frame_dir)

        return FrameData(
            frame_number=frame_number,
            original_frame=frame,
            annotated_frame=annotated_frame,
            detections=detections,
            frame_dir=frame_dir,
            camera_source=camera_source  # Include camera source in FrameData
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[PlayerDetection],
        font: int,
        font_scale: float,
        color: tuple,
        thickness: int
    ) -> np.ndarray:
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                f"Player {detection.player_id}",
                (x1, y1 - 10),
                font,
                font_scale,
                color,
                thickness
            )
        return frame

    def _save_player_data(self, detections: List[PlayerDetection], frame_dir: str):
        for detection in detections:
            if self.save_crops:
                crop_path = os.path.join(frame_dir, f"player_{detection.player_id}.jpg")
                cv2.imwrite(crop_path, detection.crop)
            
            if self.save_lbp:
                lbp_path = os.path.join(frame_dir, f"player_{detection.player_id}_lbp.npy")
                np.save(lbp_path, detection.lbp_histogram)