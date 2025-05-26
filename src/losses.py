import torch


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = torch.clamp(batch, 0, 1)
    normalized = (batch - mean) / std
    normalized = torch.clamp(normalized, -2.5, 2.5)
    return normalized


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b, c, h * w)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    return gram.div(c * h * w)
