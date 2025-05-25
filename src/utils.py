import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Dict

def load_image(image_path: str, size: Optional[int] = None, force_size: bool = False) -> torch.Tensor:
    """
    Load and preprocess an image for the neural style transfer model.
    
    Args:
        image_path: Path to the image file
        size: Optional target size for the image. If None, uses original dimensions.
              If specified, maintains aspect ratio and resizes the smaller dimension to this size.
        force_size: If True, forces the image to be exactly size x size. If False, maintains aspect ratio.
    
    Returns:
        Preprocessed image tensor of shape (1, 3, H, W)
    """
    image = Image.open(image_path).convert('RGB')
    
    if size is not None:
        if force_size:
            # Force exact size
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
        else:
            # Calculate new dimensions while maintaining aspect ratio
            w, h = image.size
            if w > h:
                new_w = int(w * size / h)
                new_h = size
            else:
                new_w = size
                new_h = int(h * size / w)
            
            transform = transforms.Compose([
                transforms.Resize((new_h, new_w)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    
    image = transform(image)
    
    # Ensure the image has the correct shape (C, H, W)
    if len(image.shape) != 3:
        raise ValueError(f"Expected image to have 3 dimensions (C, H, W), got {len(image.shape)}")
    if image.shape[0] != 3:
        raise ValueError(f"Expected image to have 3 channels, got {image.shape[0]}")
    
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

def save_image(tensor: torch.Tensor, output_path: str) -> None:
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor of shape (1, 3, H, W) or (3, H, W)
        output_path: Path where the image should be saved
    """
    # Move tensor to CPU and convert to numpy
    img = tensor.clone().detach().cpu()
    
    # Ensure we have the right shape
    if len(img.shape) == 4:  # (1, 3, H, W)
        img = img.squeeze(0)  # Remove batch dimension
    elif len(img.shape) != 3:  # Should be (3, H, W)
        raise ValueError(f"Expected tensor of shape (1, 3, H, W) or (3, H, W), got {tensor.shape}")
    
    # Convert to numpy and save
    img = img.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(output_path)

def show_images(images: List[torch.Tensor], titles: Optional[List[str]] = None) -> None:
    """
    Display a list of images side by side.
    
    Args:
        images: List of image tensors
        titles: Optional list of titles for each image
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        # Convert to numpy and display
        img = img.clone().detach().cpu()
        img = img.squeeze(0)
        img = img.clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        ax.imshow(img)
        ax.axis('off')
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display images: {e}")
        print("Images will be saved to disk instead.")

def normalize_batch(batch):
    """Normalize batch using ImageNet mean and std with value clamping."""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    # Clamp values to prevent extreme normalization
    batch = torch.clamp(batch, 0.0, 1.0)
    return (batch - mean) / std

def gram_matrix(tensor):
    """
    Calculate the Gram matrix of a tensor.
    
    Args:
        tensor: A tensor of shape (B, C, H, W)
    
    Returns:
        A tensor of shape (B, C, C) containing the Gram matrices
    """
    b, c, h, w = tensor.size()
    # Reshape to (B, C, H*W)
    tensor = tensor.view(b, c, h * w)
    # Compute Gram matrix: (B, C, C)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    # Normalize by the number of elements
    return gram.div(c * h * w)

def get_device() -> torch.device:
    """
    Get the appropriate device (CPU/GPU) for training.
    
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def create_output_dir(output_dir: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def setup_experiment_dir(base_dir: str = "experiments") -> str:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for all experiments
    
    Returns:
        Path to the new experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    
    # Create directory structure
    dirs = {
        'checkpoints': 'model checkpoints',
        'outputs': 'stylized outputs',
        'plots': 'training plots',
        'logs': 'training logs'
    }
    
    for dir_name, _ in dirs.items():
        create_output_dir(os.path.join(exp_dir, dir_name))
    
    return exp_dir

def save_training_config(args: Dict, exp_dir: str) -> None:
    """
    Save training configuration to a JSON file.
    
    Args:
        args: Training arguments
        exp_dir: Experiment directory
    """
    config = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'style_image': args.style_image,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'content_weight': args.content_weight,
        'style_weight': args.style_weight,
        'checkpoint_interval': args.checkpoint_interval
    }
    
    config_path = os.path.join(exp_dir, 'logs', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def save_loss_plot(losses: Dict[str, List[float]], exp_dir: str) -> None:
    """
    Save loss plots to file.
    
    Args:
        losses: Dictionary of loss values over time
        exp_dir: Experiment directory
    """
    plt.figure(figsize=(12, 6))
    
    # Plot individual losses
    plt.subplot(1, 2, 1)
    for name, values in losses.items():
        if name != 'total':
            plt.plot(values, label=name)
    plt.title('Content and Style Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot total loss
    plt.subplot(1, 2, 2)
    plt.plot(losses['total'], label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'plots', 'losses.png'))
    plt.close()

def save_training_summary(losses: Dict[str, List[float]], exp_dir: str) -> None:
    """
    Save training summary to a text file.
    
    Args:
        losses: Dictionary of loss values over time
        exp_dir: Experiment directory
    """
    summary = {
        'final_losses': {
            name: values[-1] for name, values in losses.items()
        },
        'min_losses': {
            name: min(values) for name, values in losses.items()
        },
        'max_losses': {
            name: max(values) for name, values in losses.items()
        },
        'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(exp_dir, 'logs', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

def save_stylized_results(content_img: torch.Tensor, 
                         style_img: torch.Tensor, 
                         output_img: torch.Tensor, 
                         exp_dir: str,
                         test_name: str = "test") -> None:
    """
    Save stylized results with comparison.
    
    Args:
        content_img: Content image tensor
        style_img: Style image tensor
        output_img: Generated image tensor
        exp_dir: Experiment directory
        test_name: Name for this test result
    """
    # Save individual images
    save_image(content_img, os.path.join(exp_dir, 'outputs', f'{test_name}_content.jpg'))
    save_image(style_img, os.path.join(exp_dir, 'outputs', f'{test_name}_style.jpg'))
    save_image(output_img, os.path.join(exp_dir, 'outputs', f'{test_name}_output.jpg'))
    
    # Create and save comparison plot
    plt.figure(figsize=(15, 5))
    
    images = [content_img, style_img, output_img]
    titles = ['Content', 'Style', 'Generated']
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, idx + 1)
        img = img.clone().detach().cpu()
        img = img.squeeze(0)
        img = img.clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'plots', f'{test_name}_comparison.png'))
    plt.close() 