import torch
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict
import glob
import imageio


def show_images(images: List[torch.Tensor], titles: Optional[List[str]] = None) -> None:
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    if n_images == 1:
        axes = [axes]
    for idx, (ax, img) in enumerate(zip(axes, images)):
        img = img.clone().detach().cpu()
        img = img.squeeze(0)
        img = torch.clamp(img, 0, 1)
        img = (img * 255).clamp(0, 255)
        img = img.numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        ax.imshow(img)
        ax.axis("off")
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display images: {e}")
        print("Images will be saved to disk instead.")


def save_loss_plot(losses: Dict[str, List[float]], exp_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(losses["content"], label="Content Loss", color="blue")
    plt.plot(losses["style"], label="Style Loss", color="red")
    plt.plot(
        losses["total"], label="Total Loss", color="black", linestyle="--", alpha=0.7
    )
    plt.title("Training Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", "losses.png"))
    plt.close()


def save_stylized_results(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    output_img: torch.Tensor,
    exp_dir: str,
    test_name: str = "test",
) -> None:
    from image_io import save_image

    save_image(
        content_img, os.path.join(exp_dir, "outputs", f"{test_name}_content.jpg")
    )
    save_image(style_img, os.path.join(exp_dir, "outputs", f"{test_name}_style.jpg"))
    save_image(output_img, os.path.join(exp_dir, "outputs", f"{test_name}_output.jpg"))
    plt.figure(figsize=(15, 5))
    images = [content_img, style_img, output_img]
    titles = ["Content", "Style", "Generated"]
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, idx + 1)
        img = img.clone().detach().cpu()
        img = img.squeeze(0)
        img = torch.clamp(img, 0, 1)
        img = (img * 255).clamp(0, 255)
        img = img.numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "plots", f"{test_name}_comparison.png"))
    plt.close()


def plot_gram_matrices(
    gram_style, gram_generated, exp_dir, layer_names=None, epoch=None
):
    os.makedirs(os.path.join(exp_dir, "plots", "gram_matrices"), exist_ok=True)
    num_layers = len(gram_style)
    for i in range(num_layers):
        style_gram = gram_style[i][0].detach().cpu().numpy()
        gen_gram = gram_generated[i][0].detach().cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(style_gram, cmap="viridis")
        plt.title(f"Style Gram {layer_names[i] if layer_names else i}")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(gen_gram, cmap="viridis")
        plt.title(f"Generated Gram {layer_names[i] if layer_names else i}")
        plt.colorbar()
        plt.tight_layout()
        fname = f"gram_{layer_names[i] if layer_names else i}"
        if epoch is not None:
            fname += f"_epoch{epoch}"
        plt.savefig(os.path.join(exp_dir, "plots", "gram_matrices", f"{fname}.png"))
        plt.close()


def create_training_gif(exp_dir: str, output_name: str = "training_progress.gif"):
    """Create a GIF from the training progress images.
    
    Args:
        exp_dir (str): Directory containing the progress images
        output_name (str): Name of the output GIF file
    """
    # Get all checkpoint images
    image_files = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "progress_*.jpg")))
    
    if not image_files:
        print("No progress images found to create GIF")
        return
    
    # Read images
    images = []
    for filename in image_files:
        images.append(imageio.imread(filename))
    
    # Save as GIF
    output_path = os.path.join(exp_dir, output_name)
    imageio.mimsave(output_path, images, duration=0.3)  # 0.3 seconds per frame
    print(f"Training progress GIF saved to {output_path}")
