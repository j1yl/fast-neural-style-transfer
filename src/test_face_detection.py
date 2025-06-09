import torch
import cv2
import numpy as np
from face_utils import create_face_mask, calculate_adaptive_weights
from image_io import load_image
import matplotlib.pyplot as plt

def test_face_detection(image_path):
    # Load image
    print(f"Loading image from {image_path}...")
    image = load_image(image_path, size=1080, force_size=True)
    
    # Create face mask
    print("Creating face mask...")
    face_mask = create_face_mask(image)
    
    # Calculate adaptive weights
    print("Calculating adaptive weights...")
    style_weights = calculate_adaptive_weights(face_mask)
    
    # Convert tensors to numpy arrays for visualization
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Face mask
    plt.subplot(132)
    plt.imshow(face_mask, cmap='gray')
    plt.title('Face Mask')
    plt.axis('off')
    
    # Style weights
    plt.subplot(133)
    plt.imshow(style_weights, cmap='jet')
    plt.title('Style Weights')
    plt.axis('off')
    
    # Save visualization
    plt.savefig('face_detection_test.png')
    print("Visualization saved as 'face_detection_test.png'")
    
    # Print some statistics
    print(f"\nFace detection statistics:")
    print(f"Image shape: {image_np.shape}")
    print(f"Face mask shape: {face_mask.shape}")
    print(f"Number of face pixels: {np.sum(face_mask > 0)}")
    print(f"Style weights range: [{style_weights.min():.2f}, {style_weights.max():.2f}]")

if __name__ == "__main__":
    # Test with the content image from experiment.py
    test_face_detection("data/headshot3.jpg") 