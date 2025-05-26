import torch
from PIL import Image, ImageEnhance
import os
from torchvision import transforms
from datetime import datetime
from typing import Optional, Dict
import json

_already_reported_resize = set()


def load_image(
    image_path: str,
    size: Optional[int] = 224,
    force_size: bool = True,
    output_size: Optional[int] = None,
) -> torch.Tensor:
    global _already_reported_resize
    image = Image.open(image_path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    orig_size = image.size
    if size is not None:
        if force_size:
            if (
                image.size != (size, size)
                and image_path not in _already_reported_resize
            ):
                print(
                    f"[INFO] Resizing {image_path} from {orig_size} to {size}x{size} for VGG16 compatibility."
                )
                _already_reported_resize.add(image_path)
            transform = transforms.Compose(
                [transforms.Resize((size, size)), transforms.ToTensor()]
            )
        else:
            w, h = image.size
            if w > h:
                new_w = int(w * size / h)
                new_h = size
            else:
                new_w = size
                new_h = int(h * size / w)
            if (
                new_w,
                new_h,
            ) != image.size and image_path not in _already_reported_resize:
                print(
                    f"[INFO] Resizing {image_path} from {orig_size} to {new_w}x{new_h} for VGG16 compatibility."
                )
                _already_reported_resize.add(image_path)
            transform = transforms.Compose(
                [transforms.Resize((new_h, new_w)), transforms.ToTensor()]
            )
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected image to have 3 dimensions (C, H, W), got {len(image.shape)}"
        )
    if image.shape[0] != 3:
        raise ValueError(f"Expected image to have 3 channels, got {image.shape[0]}")
    image = image.unsqueeze(0)
    if output_size is not None:
        image = torch.nn.functional.interpolate(
            image, size=(output_size, output_size), mode="bilinear", align_corners=False
        )
        print(f"[INFO] Resized image to {output_size}x{output_size} for output.")
    return image


def save_image(tensor: torch.Tensor, output_path: str) -> None:
    img = tensor.clone().detach().cpu()
    if len(img.shape) == 4:
        img = img.squeeze(0)
    elif len(img.shape) != 3:
        raise ValueError(
            f"Expected tensor of shape (1, 3, H, W) or (3, H, W), got {tensor.shape}"
        )
    img = torch.clamp(img, 0, 1)
    img = (img * 255).clamp(0, 255)
    img = img.numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    img.save(output_path)


def create_output_dir(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def setup_experiment_dir(base_dir: str = "experiments") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    dirs = {
        "checkpoints": "model checkpoints",
        "outputs": "stylized outputs",
        "plots": "training plots",
        "logs": "training logs",
    }
    for dir_name, _ in dirs.items():
        create_output_dir(os.path.join(exp_dir, dir_name))
    return exp_dir


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_training_config(args: Dict, exp_dir: str) -> None:
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "style_image": args.style_image,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "content_weight": args.content_weight,
        "style_weight": args.style_weight,
        "checkpoint_interval": args.checkpoint_interval,
    }
    config_path = os.path.join(exp_dir, "logs", "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def save_training_summary(losses: Dict[str, list], exp_dir: str) -> None:
    summary = {
        "final_losses": {name: values[-1] for name, values in losses.items()},
        "min_losses": {name: min(values) for name, values in losses.items()},
        "max_losses": {name: max(values) for name, values in losses.items()},
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = os.path.join(exp_dir, "logs", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
