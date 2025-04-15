import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class KernelComputations:
    """Separate class for kernel-related computations."""
    
    @staticmethod
    def compute_kernel_cross_entropy(target: torch.Tensor, 
                                   learned: torch.Tensor, 
                                   eps: float = 1e-7, 
                                   log: bool = True) -> torch.Tensor:
        """Compute cross entropy between target and learned kernel matrices."""
        if target.shape != learned.shape:
            raise ValueError(f"Shape mismatch: target {target.shape} != learned {learned.shape}")
            
        target_flat = target.flatten()
        learned_flat = learned.flatten()
        non_zero_mask = target_flat > eps

        target_filtered = target_flat[non_zero_mask]
        learned_filtered = learned_flat[non_zero_mask]
        
        if torch.isnan(learned_filtered).any():
            raise ValueError("NaN values detected in learned kernel")
            
        log_q = learned_filtered if log else torch.log(learned_filtered.clamp(min=eps))
        cross_entropy_loss = -torch.sum(target_filtered * log_q)
        
        return cross_entropy_loss / target.shape[0]

class KernelLoss:
    """Enhanced loss function factory with input validation."""
    
    @staticmethod
    def get_loss_fn(loss_type: str):
        loss_functions = {
            'kl': lambda x, y, log: F.kl_div(y.clamp(min=1e-10).log(), x, reduction='batchmean'),
            'ce': KernelComputations.compute_kernel_cross_entropy,
            'l2': lambda x, y, log: F.mse_loss(x, y),
            'tv': lambda x, y, log: 0.5 * torch.abs(x - y).sum()/x.shape[0],
            'hellinger': lambda x, y, log: (torch.sqrt(x.clamp(min=1e-10)) - 
                                     torch.sqrt(y.clamp(min=1e-10))).pow(2).sum()/x.shape[0],
            'orthogonality': lambda x, y: -(x * y).mean(),
            'jsd': lambda x, y, log: 0.5 * (
                F.kl_div(y.clamp(min=1e-10).log(), x, reduction='batchmean') +
                F.kl_div(x.clamp(min=1e-10).log(), y, reduction='batchmean')
            ),
            'none': lambda x, y: torch.tensor(0.0, device=x.device)
        }
        if loss_type not in loss_functions:
            raise ValueError(f"Unsupported loss_type: {loss_type}. "
                           f"Available types: {list(loss_functions.keys())}")
        return loss_functions[loss_type]

def debiased_loss(out1, out2, temperature=0.5, tau_plus=0.1, debiased=True, cosine = True):
    # Concatenate outputs
    out = torch.cat([out1, out2], dim=0)   
    if cosine:
        out = F.normalize(out, dim=1)
        d_matrix = torch.mm(out, out.t().contiguous())
    else:
        d_matrix = -compute_squared_euclidean_distance_matrix(out)

    batch_size = out1.size(0)
    mask, pos_mask = get_negative_mask(batch_size, device=out.device)
    
    neg = torch.exp(d_matrix/ temperature)
    neg = neg.masked_select(mask).view(2 * batch_size, -1) 

    d_pos = d_matrix.masked_select(pos_mask)   
    pos = torch.exp(d_pos/ temperature)

    # Estimator g()
    if debiased:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim=-1)) / (1 - tau_plus)
        # Constrain Ng
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    # Contrastive loss
    return (-torch.log(pos / (pos + Ng))).mean()

def get_negative_mask(batch_size, device='cuda'):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    positive_mask = ((1-1*negative_mask)-torch.eye(2*batch_size)).bool()
    return negative_mask.to(device), positive_mask.to(device)

def compute_squared_euclidean_distance_matrix(out):
    squared_norm = (out ** 2).sum(dim=1, keepdim=True)
    dist_matrix = squared_norm + squared_norm.t() - 2 * torch.mm(out, out.t())
    dist_matrix = torch.clamp(dist_matrix, min=0.0)
    
    return dist_matrix/2
