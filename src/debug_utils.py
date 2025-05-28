import torch
import os
import numpy as np
import json


def debug_normalize_batch(batch, exp_dir=None):
    normalized = None
    from losses import normalize_batch

    normalized = normalize_batch(batch)
    return normalized


def debug_image_values(tensor: torch.Tensor, name: str) -> None:
    tensor = tensor.clone().detach().cpu()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    print(f"\n{name} Statistics:")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min().item():.3f}")
    print(f"Max: {tensor.max().item():.3f}")
    print(f"Mean: {tensor.mean().item():.3f}")
    print(f"Std: {tensor.std().item():.3f}")
    for i, channel in enumerate(["R", "G", "B"]):
        print(
            f"{channel} channel - Min: {tensor[i].min().item():.3f}, Max: {tensor[i].max().item():.3f}, Mean: {tensor[i].mean().item():.3f}"
        )


def log_gram_matrix_stats(
    gram_style, gram_generated, exp_dir, layer_names=None, epoch=None
):
    stats = {"epoch": epoch, "layers": []}
    num_layers = len(gram_style)
    for i in range(num_layers):
        style_gram = gram_style[i][0].detach().cpu().numpy()
        gen_gram = gram_generated[i][0].detach().cpu().numpy()
        mse = float(np.mean((style_gram - gen_gram) ** 2))
        diag_style = np.diag(style_gram)
        diag_gen = np.diag(gen_gram)
        offdiag_style = style_gram[~np.eye(style_gram.shape[0], dtype=bool)]
        offdiag_gen = gen_gram[~np.eye(gen_gram.shape[0], dtype=bool)]
        layer_stat = {
            "layer": layer_names[i] if layer_names else str(i),
            "mse": mse,
            "diag_style_mean": float(np.mean(diag_style)),
            "diag_style_std": float(np.std(diag_style)),
            "diag_gen_mean": float(np.mean(diag_gen)),
            "diag_gen_std": float(np.std(diag_gen)),
            "offdiag_style_mean": float(np.mean(offdiag_style)),
            "offdiag_style_std": float(np.std(offdiag_style)),
            "offdiag_gen_mean": float(np.mean(offdiag_gen)),
            "offdiag_gen_std": float(np.std(offdiag_gen)),
        }
        stats["layers"].append(layer_stat)
    log_path = os.path.join(exp_dir, "logs", "gram_stats.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(stats)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)
