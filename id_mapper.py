import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Add this at the very top of your script

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor

def get_embedding(image_path, extractor):
    """Get image embedding with error handling"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return extractor(img)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def compare_images(img1_path, img2_path, extractor):
    """Safe image comparison with error handling"""
    try:
        emb1 = get_embedding(img1_path, extractor)
        emb2 = get_embedding(img2_path, extractor)
        
        if emb1 is None or emb2 is None:
            return 0.0
            
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        return torch.mm(emb1, emb2.t()).item()
    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return 0.0

if __name__ == "__main__":
    # Initialize with error handling
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='osnet_x1_0_market1501.pth',
            device=device
        )
        print(f"✅ Model loaded on {device.upper()}")

        img1 = "saved_frames_broadcast/frame_1/broadcast_player_23.jpg"
        img2 = "saved_frames_tacticam/frame_1/tacticam_player_5.jpg"  # replace with another player's image
        
        if os.path.exists(img1) and os.path.exists(img2):
            similarity = compare_images(img1, img2, extractor)
            print(f"🔍 Similarity: {similarity:.4f}")
        else:
            print("❌ One or both images not found")
            
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")



