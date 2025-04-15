import torch


def fill_diagonal(tensor, value):
    mask = torch.eye(tensor.shape[0], device=tensor.device, dtype=torch.bool)
    tensor = tensor.masked_fill(mask, value) 
    return tensor