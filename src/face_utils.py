import cv2
import dlib
import numpy as np
import torch
from typing import Tuple, Optional


def create_face_mask(
    image: torch.Tensor, predictor_path: str = "shape_predictor_68_face_landmarks.dat"
) -> np.ndarray:
    """
    Create a face mask using dlib's facial landmark detector.

    Args:
        image: Input image tensor (B, C, H, W) or (C, H, W)
        predictor_path: Path to dlib's facial landmark predictor file

    Returns:
        Binary mask where face regions are marked as 255
    """
    # Convert image to grayscale for face detection
    if isinstance(image, torch.Tensor):
        # Handle batched input
        if len(image.shape) == 4:  # (B, C, H, W)
            image = image[0]  # Take first image from batch
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return np.ones_like(gray)  # Return full weight if no face detected

    # Create mask
    mask = np.zeros_like(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask


def calculate_adaptive_weights(
    face_mask: np.ndarray, base_style_weight: float = 1.0
) -> np.ndarray:
    """
    Calculate adaptive style weights based on distance from face regions.
    Creates an extreme case where faces receive all the styling while the rest of the image remains largely unchanged.

    Args:
        face_mask: Binary mask where face regions are marked as 255
        base_style_weight: Base weight for style transfer

    Returns:
        Weight map where values are much higher in face regions and very low elsewhere
    """
    # Create distance transform from face mask
    distance_map = cv2.distanceTransform(face_mask, cv2.DIST_L2, 3)
    # Normalize distance map
    distance_map = distance_map / distance_map.max()
    
    # Create extreme contrast by using a higher power
    style_weights = np.power(distance_map, 4)  # Higher power for more extreme contrast
    
    # Scale to create very high weights for faces and very low weights elsewhere
    style_weights = 0.01 + 0.99 * style_weights  # Most of the image gets very low weight
    
    # Create a sharp transition around face regions
    face_region = (face_mask > 0).astype(np.float32)
    style_weights = np.where(face_region > 0, 1.0, style_weights)  # Set face regions to maximum weight
    
    # Normalize to have mean of 1.0
    style_weights = style_weights / style_weights.mean()
    return style_weights


def compute_style_loss(
    features_y: list,
    gram_style: list,
    style_weights: torch.Tensor,
    mse_loss: torch.nn.Module,
) -> torch.Tensor:
    """
    Compute style loss with adaptive weights.

    Args:
        features_y: List of feature maps from the generated image
        gram_style: List of gram matrices from the style image
        style_weights: Weight map for style transfer
        mse_loss: MSE loss function

    Returns:
        Weighted style loss
    """
    style_loss = 0.0
    for ft_y, gm_s in zip(features_y, gram_style):
        # Get feature dimensions
        b, c, h, w = ft_y.shape
        
        # Ensure style_weights is on the same device as features
        style_weights = style_weights.to(ft_y.device)
        
        # Reshape style weights to match batch size and normalize
        style_weights = style_weights.view(b, -1).mean(dim=1)  # Average across spatial dimensions
        style_weights = style_weights / style_weights.mean()  # Ensure mean is 1.0
        
        # Compute gram matrix for generated image
        gm_y = gram_matrix(ft_y)
        
        # Apply style weights to the generated gram matrix
        weighted_gm_y = gm_y * style_weights.view(b, 1, 1)
        
        # Compute loss between weighted generated gram matrix and style gram matrix
        style_loss += mse_loss(weighted_gm_y, gm_s.expand(b, -1, -1))
    return style_loss


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of a tensor.

    Args:
        tensor: Input tensor of shape (B, C, H, W)

    Returns:
        Gram matrix of shape (B, C, C)
    """
    b, c, h, w = tensor.size()
    tensor = tensor.view(b, c, h * w)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    return gram.div(c * h * w)
