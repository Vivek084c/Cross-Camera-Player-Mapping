import os
import cv2
import yaml
from typing import List
from tracker import PlayerTracker
from processor import FrameProcessor
from mapper import CrossCameraMapper
from models import FrameData  # Import from models.py
import numpy as np
class CrossCameraApp:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        os.makedirs(self.config['output']['directory'], exist_ok=True)
        
        self.tracker = PlayerTracker(
            model_path=self.config['model']['path'],
            confidence_threshold=self.config['model']['confidence_threshold'],
            target_class=self.config['model']['target_class']
        )
        
        self.processor = FrameProcessor(
            output_dir=self.config['output']['directory'],
            save_crops=self.config['output']['save_crops'],
            save_lbp=self.config['output']['save_lbp']
        )
        
        self.mapper = CrossCameraMapper(
            output_dir=self.config['output']['directory'],
            threshold=self.config['lbp']['matching_threshold']
        )
        
        self.vis_config = {
            'tacticam': {
                'color': self.config['visualization']['tacticam_color'],
                'font': getattr(cv2, self.config['visualization']['font']),
                'font_scale': self.config['visualization']['font_scale'],
                'thickness': self.config['visualization']['box_thickness']
            },
            'broadcast': {
                'color': self.config['visualization']['broadcast_color'],
                'font': getattr(cv2, self.config['visualization']['font']),
                'font_scale': self.config['visualization']['font_scale'],
                'thickness': self.config['visualization']['box_thickness']
            }
        }

    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _process_single_frame(self, frame: np.ndarray, frame_number: int, camera_source: str) -> FrameData:
        detections = self.tracker.detect_players(
            frame,
            radius=self.config['lbp']['radius'],
            method=self.config['lbp']['method']
        )
        
        # Add camera source to each detection
        for detection in detections:
            detection.camera_source = camera_source
        
        return self.processor.process_frame(
            frame=frame,
            detections=detections,
            frame_number=frame_number,
            camera_source=camera_source,  # This is now accepted
            vis_config=self.vis_config[camera_source]
        )

    def run(self):
        # Initialize video captures
        cap1 = cv2.VideoCapture(self.config['videos']['tacticam'])
        cap2 = cv2.VideoCapture(self.config['videos']['broadcast'])
        
        frame_count = 0
        while frame_count < self.config.get('max_frames', 1000):
            # Read frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("End of video(s) reached.")
                break
            
            # Process both frames
            tacticam_data = self._process_single_frame(frame1, frame_count, 'tacticam')
            broadcast_data = self._process_single_frame(frame2, frame_count, 'broadcast')
            
            # Update mappings with both frames
            self.mapper.update_mappings([tacticam_data, broadcast_data])
            
            # Visualize results
            self._visualize_results(tacticam_data, broadcast_data)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        self.mapper.save_mappings()
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

    def _visualize_results(self, tacticam_data: FrameData, broadcast_data: FrameData):
        """Create side-by-side visualization with mapping info"""
        # Resize frames to match the smallest dimensions
        h1, w1 = tacticam_data.original_frame.shape[:2]
        h2, w2 = broadcast_data.original_frame.shape[:2]
        
        # Use the smallest height and width
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        # Resize all frames to match target dimensions
        def resize_frame(frame, target_h, target_w):
            return cv2.resize(frame, (target_w, target_h))
        
        # Resize original frames
        tacticam_orig = resize_frame(tacticam_data.original_frame, target_h, target_w)
        broadcast_orig = resize_frame(broadcast_data.original_frame, target_h, target_w)
        
        # Resize annotated frames
        tacticam_annot = resize_frame(tacticam_data.annotated_frame, target_h, target_w)
        broadcast_annot = resize_frame(broadcast_data.annotated_frame, target_h, target_w)
        
        # Combine original frames
        combined_frames = np.hstack([tacticam_orig, broadcast_orig])
        
        # Combine annotated frames
        combined_annotated = np.hstack([tacticam_annot, broadcast_annot])
        
        # Create info panel with matching width
        info_width = combined_frames.shape[1]
        info_panel = np.zeros((200, info_width, 3), dtype=np.uint8)  # Fixed height for info panel
        
        # Add mapping info
        y = 30
        for global_id, player_data in self.mapper.global_mappings.items():
            text = f"Global ID {global_id}: Seen {player_data['frames_seen']} frames"
            cv2.putText(info_panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
            y += 30
        
        # Final combined visualization
        top_row = np.vstack([combined_frames, combined_annotated])
        final_visualization = np.vstack([top_row, info_panel])
        
        cv2.imshow("Cross-Camera Player Mapping", final_visualization)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app = CrossCameraApp()
    app.run()