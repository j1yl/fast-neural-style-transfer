import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Dict
from scipy.stats import wasserstein_distance

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
    # Open image and handle transparency
    image = Image.open(image_path)
    
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a white background
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    
    # Composite the image onto the white background
    image = Image.alpha_composite(background, image)
    
    # Convert to RGB after compositing
    image = image.convert('RGB')
    
    if size is not None:
        if force_size:
            # Force exact size
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()  # This will normalize to [0,1]
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
                transforms.ToTensor()  # This will normalize to [0,1]
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()  # This will normalize to [0,1]
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
        tensor: Image tensor of shape (1, 3, H, W) or (3, H, W) in [0,1] range
        output_path: Path where the image should be saved
    """
    # Move tensor to CPU and convert to numpy
    img = tensor.clone().detach().cpu()
    
    # Ensure we have the right shape
    if len(img.shape) == 4:  # (1, 3, H, W)
        img = img.squeeze(0)  # Remove batch dimension
    elif len(img.shape) != 3:  # Should be (3, H, W)
        raise ValueError(f"Expected tensor of shape (1, 3, H, W) or (3, H, W), got {tensor.shape}")
    
    # Ensure values are in [0,1] range
    img = torch.clamp(img, 0, 1)
    
    # Convert to [0,255] range
    img = (img * 255).clamp(0, 255)
    
    # Convert to numpy and save
    img = img.numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    
    # Enhance saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)  # Increase saturation by 20%
    
    img.save(output_path)

def debug_image_values(tensor: torch.Tensor, name: str) -> None:
    """
    Print debug information about image tensor values.
    
    Args:
        tensor: Image tensor
        name: Name to identify the tensor in debug output
    """
    tensor = tensor.clone().detach().cpu()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    print(f"\n{name} Statistics:")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min().item():.3f}")
    print(f"Max: {tensor.max().item():.3f}")
    print(f"Mean: {tensor.mean().item():.3f}")
    print(f"Std: {tensor.std().item():.3f}")
    
    # Print channel-wise statistics
    for i, channel in enumerate(['R', 'G', 'B']):
        print(f"{channel} channel - Min: {tensor[i].min().item():.3f}, Max: {tensor[i].max().item():.3f}, Mean: {tensor[i].mean().item():.3f}")

def show_images(images: List[torch.Tensor], titles: Optional[List[str]] = None) -> None:
    """
    Display a list of images side by side.
    
    Args:
        images: List of image tensors in [0,1] range
        titles: Optional list of titles for each image
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        # Debug image values
        debug_image_values(img, f"Image {idx + 1}")
        
        # Convert to numpy and display
        img = img.clone().detach().cpu()
        img = img.squeeze(0)
        
        # Ensure values are in [0,1] range
        img = torch.clamp(img, 0, 1)
        
        # Convert to [0,255] range
        img = (img * 255).clamp(0, 255)
        
        img = img.numpy()
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
    """Normalize batch using ImageNet mean and std.
    Expects input in [0,1] range from transformer."""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    # Ensure input is in [0,1] range
    batch = torch.clamp(batch, 0, 1)
    
    # Apply mean/std normalization
    normalized = (batch - mean) / std
    
    # Use a conservative clamping range
    normalized = torch.clamp(normalized, -2.5, 2.5)
    
    return normalized

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
        content_img: Content image tensor in [0,1] range
        style_img: Style image tensor in [0,1] range
        output_img: Generated image tensor in [0,1] range
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
        
        # Ensure values are in [0,1] range
        img = torch.clamp(img, 0, 1)
        
        # Convert to [0,255] range
        img = (img * 255).clamp(0, 255)
        
        img = img.numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'plots', f'{test_name}_comparison.png'))
    plt.close()

def compute_color_statistics(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute color statistics (mean, std) for each channel.
    
    Args:
        tensor: Image tensor of shape (B, C, H, W) or (C, H, W)
    
    Returns:
        Dictionary containing mean and std for each channel
    """
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    # Compute mean and std for each channel
    mean = tensor.mean(dim=(2, 3))  # (B, C)
    std = tensor.std(dim=(2, 3))    # (B, C)
    
    # Compute overall brightness
    brightness = tensor.mean()  # Scalar
    
    return {
        'mean': mean,
        'std': std,
        'brightness': brightness
    }

def compute_color_histogram(tensor: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Compute color histogram for each channel.
    
    Args:
        tensor: Image tensor of shape (B, C, H, W) or (C, H, W)
        bins: Number of histogram bins
    
    Returns:
        Histogram tensor of shape (B, C, bins)
    """
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    B, C, H, W = tensor.shape
    histograms = []
    
    for b in range(B):
        channel_hists = []
        for c in range(C):
            # Flatten channel and compute histogram
            channel_data = tensor[b, c].detach().cpu().numpy()
            # Ensure data is in valid range
            channel_data = np.clip(channel_data, 0, 255)
            # Compute histogram with proper normalization
            hist, _ = np.histogram(channel_data, bins=bins, range=(0, 255), density=True)
            # Handle zero sum case
            if np.sum(hist) == 0:
                hist = np.ones_like(hist) / bins
            else:
                hist = hist / np.sum(hist)
            channel_hists.append(hist)
        histograms.append(channel_hists)
    
    # Convert to numpy array first to avoid the slow tensor creation warning
    histograms = np.array(histograms)
    return torch.from_numpy(histograms).to(tensor.device)

def color_histogram_loss(source: torch.Tensor, target: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    Compute color histogram matching loss using Wasserstein distance.
    
    Args:
        source: Source image tensor
        target: Target image tensor
        bins: Number of histogram bins
    
    Returns:
        Histogram matching loss
    """
    try:
        # Detach tensors before computing histograms
        source = source.detach()
        target = target.detach()
        
        # Ensure target has the same batch size as source
        if len(target.shape) == 3:
            target = target.unsqueeze(0)
        if target.shape[0] == 1 and source.shape[0] > 1:
            target = target.expand(source.shape[0], -1, -1, -1)
        
        source_hist = compute_color_histogram(source, bins)
        target_hist = compute_color_histogram(target, bins)
        
        loss = 0
        for b in range(source_hist.shape[0]):
            for c in range(source_hist.shape[1]):
                # Compute Wasserstein distance between histograms
                dist = wasserstein_distance(
                    source_hist[b, c].cpu().numpy(),
                    target_hist[b, c].cpu().numpy()
                )
                loss += dist
        
        return torch.tensor(loss, device=source.device)
    except Exception as e:
        print(f"Warning: Error in color histogram loss computation: {e}")
        return torch.tensor(0.0, device=source.device)

def color_statistics_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute color statistics matching loss with emphasis on brightness preservation.
    
    Args:
        source: Source image tensor
        target: Target image tensor
    
    Returns:
        Color statistics matching loss
    """
    try:
        # Detach tensors before computing statistics
        source = source.detach()
        target = target.detach()
        
        # Ensure target has the same batch size as source
        if len(target.shape) == 3:
            target = target.unsqueeze(0)
        if target.shape[0] == 1 and source.shape[0] > 1:
            target = target.expand(source.shape[0], -1, -1, -1)
        
        source_stats = compute_color_statistics(source)
        target_stats = compute_color_statistics(target)
        
        # Mean matching loss
        mean_loss = torch.mean((source_stats['mean'] - target_stats['mean'])**2)
        
        # Std matching loss
        std_loss = torch.mean((source_stats['std'] - target_stats['std'])**2)
        
        # Brightness preservation loss (weighted more heavily)
        brightness_loss = 2.0 * ((source_stats['brightness'] - target_stats['brightness'])**2)
        
        return mean_loss + std_loss + brightness_loss
    except Exception as e:
        print(f"Warning: Error in color statistics loss computation: {e}")
        return torch.tensor(0.0, device=source.device)

def preserve_color_characteristics(content: torch.Tensor, 
                                 style: torch.Tensor, 
                                 output: torch.Tensor,
                                 content_weight: float = 0.5,
                                 style_weight: float = 0.5) -> torch.Tensor:
    """
    Compute color preservation loss that balances content and style color characteristics.
    
    Args:
        content: Content image tensor
        style: Style image tensor
        output: Generated image tensor
        content_weight: Weight for content color preservation
        style_weight: Weight for style color preservation
    
    Returns:
        Color preservation loss
    """
    try:
        # Detach tensors before computing losses
        content = content.detach()
        style = style.detach()
        output = output.detach()
        
        # Ensure style has the same batch size as output
        if len(style.shape) == 3:
            style = style.unsqueeze(0)
        if style.shape[0] == 1 and output.shape[0] > 1:
            style = style.expand(output.shape[0], -1, -1, -1)
        
        # Histogram matching loss
        content_hist_loss = color_histogram_loss(output, content)
        style_hist_loss = color_histogram_loss(output, style)
        
        # Color statistics loss
        content_stats_loss = color_statistics_loss(output, content)
        style_stats_loss = color_statistics_loss(output, style)
        
        # Combine losses with weights
        total_loss = (
            content_weight * (content_hist_loss + content_stats_loss) +
            style_weight * (style_hist_loss + style_stats_loss)
        )
        
        return total_loss
    except Exception as e:
        print(f"Warning: Error in color preservation loss computation: {e}")
        return torch.tensor(0.0, device=output.device)

def debug_color_values(tensor: torch.Tensor, stage: str, exp_dir: str) -> None:
    """
    Debug function to save color statistics at different stages of processing to a file.
    
    Args:
        tensor: Image tensor to analyze
        stage: String indicating the processing stage
        exp_dir: Experiment directory to save debug output
    """
    # Move tensor to CPU and ensure it's in the right range
    tensor = tensor.detach().cpu()
    
    # Create debug output
    debug_output = []
    debug_output.append(f"\nColor Debug - {stage}:")
    debug_output.append(f"Original shape: {tensor.shape}")
    
    # Ensure we have a 4D tensor (B, C, H, W)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    debug_output.append(f"Shape after batch dim: {tensor.shape}")
    
    # Get min, max, mean values for each channel
    # Reduce over spatial dimensions (last two dimensions)
    min_vals = tensor.min(dim=1)[0].min(dim=1)[0]  # (B, C)
    max_vals = tensor.max(dim=1)[0].max(dim=1)[0]  # (B, C)
    mean_vals = tensor.mean(dim=1)[0].mean(dim=1)[0]  # (B, C)
    
    # Calculate averages across batch dimension
    avg_min = min_vals.mean(dim=0) if min_vals.dim() > 0 else min_vals
    avg_max = max_vals.mean(dim=0) if max_vals.dim() > 0 else max_vals
    avg_mean = mean_vals.mean(dim=0) if mean_vals.dim() > 0 else mean_vals
    
    # Convert to list and handle 0-dim tensors
    min_list = avg_min.tolist() if avg_min.dim() > 0 else [avg_min.item()]
    max_list = avg_max.tolist() if avg_max.dim() > 0 else [avg_max.item()]
    mean_list = avg_mean.tolist() if avg_mean.dim() > 0 else [avg_mean.item()]
    
    debug_output.append(f"Average Min values (RGB): {min_list}")
    debug_output.append(f"Average Max values (RGB): {max_list}")
    debug_output.append(f"Average Mean values (RGB): {mean_list}")
    debug_output.append(f"Overall mean: {tensor.mean().item():.4f}")
    debug_output.append(f"Overall std: {tensor.std().item():.4f}")
    
    # Save to file
    debug_file = os.path.join(exp_dir, 'logs', 'color_debug.txt')
    with open(debug_file, 'a') as f:
        f.write('\n'.join(debug_output) + '\n')

def debug_normalize_batch(batch, exp_dir=None):
    """Debug version of normalize_batch that logs statistics."""
    # Log input statistics
    print(f"\nInput batch stats:")
    print(f"Min: {batch.min().item():.3f}")
    print(f"Max: {batch.max().item():.3f}")
    print(f"Mean: {batch.mean().item():.3f}")
    print(f"Std: {batch.std().item():.3f}")
    
    # Normalize
    normalized = normalize_batch(batch)
    
    # Log output statistics
    print(f"\nNormalized batch stats:")
    print(f"Min: {normalized.min().item():.3f}")
    print(f"Max: {normalized.max().item():.3f}")
    print(f"Mean: {normalized.mean().item():.3f}")
    print(f"Std: {normalized.std().item():.3f}")
    
    return normalized 