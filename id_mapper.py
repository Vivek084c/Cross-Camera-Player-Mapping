import cv2
import torch
from torchreid.utils import FeatureExtractor
import numpy as np
import torch.nn.functional as F


def get_embedding(image_path):
    """
    Loads an image, converts to RGB, and returns the embedding vector.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    emb = extractor(img)  # shape: (1, 512)
    return emb

def compare_images(img1_path, img2_path):
    """
    Computes cosine similarity between embeddings of two images.
    Returns similarity score between 0 (completely different) and 1 (identical).
    """
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    # Normalize embeddings
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    # Cosine similarity
    similarity = torch.mm(emb1, emb2.t()).item()  # dot product
    return similarity
if __name__ == "__main__":

    # Initialize the Torchreid feature extractor
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='osnet_x1_0_market1501.pth',
        device='cpu'
    )

    # Example usage
    img1 = "saved_frames/frame_2/player_11.jpg"
    img2 = "saved_frames/frame_5/player_4.jpg"  # replace with another player's image

    similarity_score = compare_images(img1, img2)
    print(f"🔍 Similarity between '{img1}' and '{img2}': {similarity_score:.4f}")