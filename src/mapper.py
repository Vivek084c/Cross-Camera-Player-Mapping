import json
import os
import numpy as np
from typing import Dict, List, Optional
from models import FrameData, PlayerDetection  # Add this import

class CrossCameraMapper:
    def __init__(self, output_dir: str, threshold: float):
        self.output_dir = output_dir
        self.threshold = threshold
        self.global_mappings: Dict[int, Dict] = {}
        self.next_global_id = 1
        self.camera_mappings = {
            'tacticam': {},
            'broadcast': {}
        }

    def update_mappings(self, frame_data_list: List[FrameData]):  # Now FrameData is recognized
        """Update mappings across multiple camera feeds"""
        all_detections = []
        for frame_data in frame_data_list:
            for detection in frame_data.detections:
                all_detections.append((detection, frame_data.camera_source))

    def _match_or_create_player(self, detection: PlayerDetection, camera_source: str):
        best_match = None
        best_similarity = 0

        for global_id, player_data in self.global_mappings.items():
            # Skip if this player was already mapped in this camera
            if global_id in self.camera_mappings[camera_source].values():
                continue

            # Compare with all appearances from other cameras
            for other_hist in player_data['lbp_histograms']:
                similarity = self._cosine_similarity(
                    detection.lbp_histogram,
                    np.array(other_hist)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = global_id

        # If good match found and above threshold
        if best_match and best_similarity > self.threshold:
            self._update_player_mapping(best_match, detection, camera_source)
        else:
            self._create_new_mapping(detection, camera_source)

    def _update_player_mapping(self, global_id: int, detection: PlayerDetection, camera_source: str):
        self.global_mappings[global_id]['lbp_histograms'].append(detection.lbp_histogram.tolist())
        self.global_mappings[global_id]['frames_seen'] += 1
        self.camera_mappings[camera_source][detection.player_id] = global_id

    def _create_new_mapping(self, detection: PlayerDetection, camera_source: str):
        self.global_mappings[self.next_global_id] = {
            'lbp_histograms': [detection.lbp_histogram.tolist()],
            'frames_seen': 1,
            'first_frame': detection.player_id
        }
        self.camera_mappings[camera_source][detection.player_id] = self.next_global_id
        self.next_global_id += 1

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_global_id(self, local_id: int, camera_source: str) -> Optional[int]:
        return self.camera_mappings[camera_source].get(local_id)

    def save_mappings(self):
        path = os.path.join(self.output_dir, "global_mappings.json")
        with open(path, 'w') as f:
            json.dump({
                'global_mappings': self.global_mappings,
                'camera_mappings': self.camera_mappings
            }, f, indent=2)