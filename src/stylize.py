import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from typing import Optional
from transformer import TransformerNet
from vgg import Vgg16
from image_io import (
    load_image,
    save_image,
    create_output_dir,
    setup_experiment_dir,
    get_device,
    save_training_config,
    save_training_summary,
)
from plotting import (
    show_images,
    save_loss_plot,
    save_stylized_results,
)
from color_utils import preserve_color_characteristics
from losses import normalize_batch, gram_matrix
from debug_utils import debug_image_values
from config import (
    CONTENT_WEIGHT,
    STYLE_WEIGHT,
    STYLE_LAYER_WEIGHTS,
    COLOR_WEIGHT,
    BRIGHTNESS_WEIGHT,
    CONTRAST_WEIGHT,
    COLOR_MATCHING_WEIGHT,
)


class ImageDataset(Dataset):
    def __init__(
        self,
        content_dir: str,
        size: Optional[int] = None,
        force_size: bool = False,
        transform=None,
    ):
        self.content_dir = Path(content_dir)
        self.image_files = list(self.content_dir.glob("*.jpg")) + list(
            self.content_dir.glob("*.png")
        )
        self.transform = transform
        self.size = size
        self.force_size = force_size
        print(f"Found {len(self.image_files)} images in {content_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = load_image(str(img_path), size=self.size, force_size=self.force_size)
        image = image.squeeze(0)
        return image, 0


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    exp_dir = setup_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    save_training_config(args, exp_dir)

    style = load_image(args.style_image, size=224, force_size=True).to(device)

    transformer = TransformerNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    # Original dataset loading code (commented out for comparison)
    """
    train_dataset = ImageDataset(args.dataset, size=224, force_size=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    """
    
    # Load FFHQ dataset from local directory
    print("Loading FFHQ dataset from local directory...")
    train_dataset = ImageDataset(
        content_dir="data/ffhq",  # Path to your local FFHQ images
        size=224,
        force_size=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    optimizer = optim.Adam(transformer.parameters(), args.lr)

    mse_loss = nn.MSELoss()

    style = normalize_batch(style)
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style]

    losses = {"content": [], "style": [], "color": [], "total": []}

    for epoch in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        agg_color_loss = 0.0
        count = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_id, batch in enumerate(pbar):
                # Get images from batch tuple (image, label)
                x, _ = batch
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                x = x.to(device)
                y = transformer(x)

                # debug_color_values(x, "Input content image", exp_dir)
                # debug_color_values(y, "Transformer output", exp_dir)

                y = normalize_batch(y)
                x = normalize_batch(x)

                features_y = vgg(y)
                features_x = vgg(x)

                # Content loss
                content_loss = args.content_weight * mse_loss(
                    features_y.relu2_2, features_x.relu2_2
                )

                # Style loss
                style_loss = 0.0
                for i, (ft_y, gm_s) in enumerate(zip(features_y, gram_style)):
                    gm_y = gram_matrix(ft_y)
                    layer_weight = (
                        STYLE_LAYER_WEIGHTS[i] if i < len(STYLE_LAYER_WEIGHTS) else 1.0
                    )
                    style_loss += layer_weight * mse_loss(
                        gm_y, gm_s.expand(n_batch, -1, -1)
                    )
                style_loss *= args.style_weight

                # Color loss
                color_loss = (
                    preserve_color_characteristics(
                        x,
                        style,
                        y,
                        content_weight=0.7,
                        style_weight=0.3,
                    )
                    * COLOR_WEIGHT
                )

                # Brightness loss
                brightness_loss = (
                    torch.mean((y.mean() - x.mean()) ** 2) * BRIGHTNESS_WEIGHT
                )

                # Contrast loss
                contrast_loss = torch.mean((y.std() - x.std()) ** 2) * CONTRAST_WEIGHT

                # Color matching loss
                color_matching_loss = torch.mean((y - x) ** 2) * COLOR_MATCHING_WEIGHT

                # Total loss
                total_loss = (
                    content_loss
                    + style_loss
                    + color_loss
                    + brightness_loss
                    + contrast_loss
                    + color_matching_loss
                )
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

                optimizer.step()

                losses["content"].append(content_loss.item())
                losses["style"].append(style_loss.item())
                losses["color"].append(color_loss.item())
                losses["total"].append(total_loss.item())

                # Update aggregate losses
                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()
                agg_color_loss += color_loss.item()

                pbar.set_postfix(
                    {
                        "content_loss": f"{agg_content_loss / (batch_id + 1):.4f}",
                        "style_loss": f"{agg_style_loss / (batch_id + 1):.4f}",
                        "color_loss": f"{agg_color_loss / (batch_id + 1):.4f}",
                        "total_loss": f"{(agg_content_loss + agg_style_loss + agg_color_loss) / (batch_id + 1):.4f}",
                    }
                )

                if (batch_id + 1) % args.checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = f"ckpt_epoch_{epoch}_batch_{batch_id + 1}.pth"
                    ckpt_model_path = os.path.join(
                        exp_dir, "checkpoints", ckpt_model_filename
                    )
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()

        # After each epoch, plot Gram matrices for style and a sample generated image
        # with torch.no_grad():
        #     sample_content = x[0:1].to(device)
        #     sample_generated = transformer(sample_content)
        #     sample_generated_norm = normalize_batch(sample_generated)
        #     features_sample = vgg(sample_generated_norm)
        #     gram_generated = [gram_matrix(y) for y in features_sample]
        #     plot_gram_matrices(gram_style, gram_generated, exp_dir, epoch=epoch + 1)
        #     log_gram_matrix_stats(gram_style, gram_generated, exp_dir, epoch=epoch + 1)

    # Save final model
    transformer.eval().cpu()
    save_model_filename = f"final_model.pth"
    save_model_path = os.path.join(exp_dir, "checkpoints", save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    # Save training results
    save_loss_plot(losses, exp_dir)
    save_training_summary(losses, exp_dir)

    print(f"\nTraining completed. Results saved in {exp_dir}")
    return exp_dir


def test_pipeline(args):
    """Test the pipeline with a single image to verify everything works."""
    device = get_device()
    print(f"Using device: {device}")

    # Load models
    transformer = TransformerNet().to(device)

    # Load test image
    test_image = load_image(args.test_image).to(device)
    style_image = load_image(args.style_image).to(device)

    # Debug input images
    debug_image_values(test_image, "Input test image")
    debug_image_values(style_image, "Input style image")

    # Forward pass
    with torch.no_grad():
        output = transformer(test_image)

    # Debug output image
    debug_image_values(output, "Transformer output")

    # Display results
    show_images(
        [test_image, style_image, output], titles=["Content", "Style", "Generated"]
    )

    # Save output
    create_output_dir("outputs")
    save_image(output, "outputs/test_output.jpg")

    print("Pipeline test completed. Check outputs/test_output.jpg")


def stylize(args):
    """Stylize a single image using a trained model."""
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(args.model))
    transformer.eval()

    # Load and process image
    image = load_image(args.image, size=args.output_size, force_size=True).to(device)
    # Warn if not 1080x1080
    if image.shape[-2:] != (1080, 1080):
        print(
            f"[WARNING] Content image is {image.shape[-2:]} but expected 1080x1080 for output. Output will be resized."
        )

    # Generate stylized image
    with torch.no_grad():
        output = transformer(image)

    # Save output
    save_image(output, args.output_image)
    print(f"Stylized image saved to {args.output_image}")


def test_trained_model(args, exp_dir: str):
    """Test the trained model on a sample image."""
    device = get_device()
    print(f"Using device: {device}")

    # Load the trained model
    transformer = TransformerNet().to(device)
    model_path = os.path.join(exp_dir, "checkpoints", "final_model.pth")
    transformer.load_state_dict(torch.load(model_path))
    transformer.eval()

    # Load test image for VGG as 224x224, but also load original for output
    style_image = load_image(args.style_image, size=224, force_size=True).to(device)
    # For output, reload test image at user-specified output size
    test_image_out = load_image(
        args.test_image, size=args.output_size, force_size=True
    ).to(device)
    if test_image_out.shape[-2:] != (1080, 1080):
        print(
            f"[WARNING] Test content image is {test_image_out.shape[-2:]} but expected 1080x1080 for output. Output will be resized."
        )

    # Generate stylized image
    with torch.no_grad():
        output = transformer(test_image_out)

    # Save results
    save_stylized_results(
        test_image_out,
        style_image,
        output,
        exp_dir,
        test_name=os.path.splitext(os.path.basename(args.test_image))[0],
    )
    print(f"Test completed. Results saved in {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument(
        "--style-image", type=str, required=True, help="path to style image"
    )

    # Training arguments
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument(
        "--dataset", type=str, default="data/content", help="path to content images"
    )
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="saved_models",
        help="path to folder where trained model will be saved",
    )
    parser.add_argument(
        "--checkpoint-model-dir",
        type=str,
        default="checkpoints",
        help="path to folder where checkpoints of trained models will be saved",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--content-weight",
        type=float,
        default=CONTENT_WEIGHT,
        help="weight for content-loss",
    )
    parser.add_argument(
        "--style-weight", type=float, default=STYLE_WEIGHT, help="weight for style-loss"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="number of batches after which a checkpoint of the trained model will be created",
    )

    # Testing arguments
    parser.add_argument("--test", action="store_true", help="test the pipeline")
    parser.add_argument("--test-image", type=str, help="path to test image")

    # Stylization arguments
    parser.add_argument("--stylize", action="store_true", help="stylize an image")
    parser.add_argument("--model", type=str, help="path to trained model")
    parser.add_argument("--image", type=str, help="path to image to stylize")
    parser.add_argument("--output-image", type=str, help="path to save stylized image")
    parser.add_argument(
        "--output-size",
        type=int,
        default=1080,
        help="output size for final stylized image (default: 1080; VGG features always use 224x224)",
    )

    args = parser.parse_args()

    if args.train:
        exp_dir = train(args)
        if args.test_image:
            test_trained_model(args, exp_dir)
    elif args.test:
        if not args.test_image:
            parser.error("--test requires --test-image")
        test_pipeline(args)
    elif args.stylize:
        if not all([args.model, args.image, args.output_image]):
            parser.error("--stylize requires --model, --image, and --output-image")
        stylize(args)
    else:
        parser.error("Please specify either --train, --test, or --stylize")
