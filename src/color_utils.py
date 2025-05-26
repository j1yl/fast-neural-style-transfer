import torch
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict


def compute_color_statistics(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    mean = tensor.mean(dim=(2, 3))
    std = tensor.std(dim=(2, 3))
    brightness = tensor.mean()
    return {"mean": mean, "std": std, "brightness": brightness}


def compute_color_histogram(tensor: torch.Tensor, bins: int = 256) -> torch.Tensor:
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    B, C, H, W = tensor.shape
    histograms = []
    for b in range(B):
        channel_hists = []
        for c in range(C):
            channel_data = tensor[b, c].detach().cpu().numpy()
            channel_data = np.clip(channel_data, 0, 255)
            hist, _ = np.histogram(
                channel_data, bins=bins, range=(0, 255), density=True
            )
            if np.sum(hist) == 0:
                hist = np.ones_like(hist) / bins
            else:
                hist = hist / np.sum(hist)
            channel_hists.append(hist)
        histograms.append(channel_hists)
    histograms = np.array(histograms)
    return torch.from_numpy(histograms).to(tensor.device)


def color_histogram_loss(
    source: torch.Tensor, target: torch.Tensor, bins: int = 256
) -> torch.Tensor:
    try:
        source = source.detach()
        target = target.detach()
        if len(target.shape) == 3:
            target = target.unsqueeze(0)
        if target.shape[0] == 1 and source.shape[0] > 1:
            target = target.expand(source.shape[0], -1, -1, -1)
        source_hist = compute_color_histogram(source, bins)
        target_hist = compute_color_histogram(target, bins)
        loss = 0
        for b in range(source_hist.shape[0]):
            for c in range(source_hist.shape[1]):
                dist = wasserstein_distance(
                    source_hist[b, c].cpu().numpy(), target_hist[b, c].cpu().numpy()
                )
                loss += dist
        return torch.tensor(loss, device=source.device)
    except Exception as e:
        print(f"Warning: Error in color histogram loss computation: {e}")
        return torch.tensor(0.0, device=source.device)


def color_statistics_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    try:
        source = source.detach()
        target = target.detach()
        if len(target.shape) == 3:
            target = target.unsqueeze(0)
        if target.shape[0] == 1 and source.shape[0] > 1:
            target = target.expand(source.shape[0], -1, -1, -1)
        source_stats = compute_color_statistics(source)
        target_stats = compute_color_statistics(target)
        mean_loss = torch.mean((source_stats["mean"] - target_stats["mean"]) ** 2)
        std_loss = torch.mean((source_stats["std"] - target_stats["std"]) ** 2)
        brightness_loss = 2.0 * (
            (source_stats["brightness"] - target_stats["brightness"]) ** 2
        )
        return mean_loss + std_loss + brightness_loss
    except Exception as e:
        print(f"Warning: Error in color statistics loss computation: {e}")
        return torch.tensor(0.0, device=source.device)


def preserve_color_characteristics(
    content: torch.Tensor,
    style: torch.Tensor,
    output: torch.Tensor,
    content_weight: float = 0.5,
    style_weight: float = 0.5,
) -> torch.Tensor:
    try:
        content = content.detach()
        style = style.detach()
        output = output.detach()
        if len(style.shape) == 3:
            style = style.unsqueeze(0)
        if style.shape[0] == 1 and output.shape[0] > 1:
            style = style.expand(output.shape[0], -1, -1, -1)
        content_hist_loss = color_histogram_loss(output, content)
        style_hist_loss = color_histogram_loss(output, style)
        content_stats_loss = color_statistics_loss(output, content)
        style_stats_loss = color_statistics_loss(output, style)
        total_loss = content_weight * (
            content_hist_loss + content_stats_loss
        ) + style_weight * (style_hist_loss + style_stats_loss)
        return total_loss
    except Exception as e:
        print(f"Warning: Error in color preservation loss computation: {e}")
        return torch.tensor(0.0, device=output.device)
