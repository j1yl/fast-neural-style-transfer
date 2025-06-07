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
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
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

    Args:
        face_mask: Binary mask where face regions are marked as 255
        base_style_weight: Base weight for style transfer

    Returns:
        Weight map where values are higher away from face regions
    """
    # Create distance transform from face mask
    distance_map = cv2.distanceTransform(face_mask, cv2.DIST_L2, 3)
    # Normalize distance map
    distance_map = distance_map / distance_map.max()
    # Invert to get higher weights away from face
    style_weights = 1 - distance_map
    # Scale to desired range
    style_weights = base_style_weight * (0.2 + 0.8 * style_weights)
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
        gm_y = gram_matrix(ft_y)
        # Apply style weights to the gram matrix
        weighted_gm_y = gm_y * style_weights.view(-1, 1, 1)
        weighted_gm_s = gm_s.expand(gm_y.size(0), -1, -1) * style_weights.view(-1, 1, 1)
        style_loss += mse_loss(weighted_gm_y, weighted_gm_s)
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
