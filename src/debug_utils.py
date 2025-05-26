import torch
import os
import numpy as np
import json


def debug_color_values(tensor: torch.Tensor, stage: str, exp_dir: str) -> None:
    tensor = tensor.detach().cpu()
    debug_output = []
    debug_output.append(f"\nColor Debug - {stage}:")
    debug_output.append(f"Original shape: {tensor.shape}")
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    debug_output.append(f"Shape after batch dim: {tensor.shape}")
    min_vals = tensor.min(dim=1)[0].min(dim=1)[0]
    max_vals = tensor.max(dim=1)[0].max(dim=1)[0]
    mean_vals = tensor.mean(dim=1)[0].mean(dim=1)[0]
    avg_min = min_vals.mean(dim=0) if min_vals.dim() > 0 else min_vals
    avg_max = max_vals.mean(dim=0) if max_vals.dim() > 0 else max_vals
    avg_mean = mean_vals.mean(dim=0) if mean_vals.dim() > 0 else mean_vals
    min_list = avg_min.tolist() if avg_min.dim() > 0 else [avg_min.item()]
    max_list = avg_max.tolist() if avg_max.dim() > 0 else [avg_max.item()]
    mean_list = avg_mean.tolist() if avg_mean.dim() > 0 else [avg_mean.item()]
    debug_output.append(f"Average Min values (RGB): {min_list}")
    debug_output.append(f"Average Max values (RGB): {max_list}")
    debug_output.append(f"Average Mean values (RGB): {mean_list}")
    debug_output.append(f"Overall mean: {tensor.mean().item():.4f}")
    debug_output.append(f"Overall std: {tensor.std().item():.4f}")
    debug_file = os.path.join(exp_dir, "logs", "color_debug.txt")
    with open(debug_file, "a") as f:
        f.write("\n".join(debug_output) + "\n")


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
