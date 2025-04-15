# kernels/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from .base import Kernel
from .distance import DistanceKernel

class UniformKNN(Kernel):
    """Computes k-nearest neighbor graph as a kernel."""
    
    def __init__(self, k: int, metric = 'euclidean', **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance_kernel = DistanceKernel(metric=metric)
    
    def _compute(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Compute KNN graph from features."""
        # Compute pairwise distances
        distances = self.distance_kernel(features)
        
        # Handle self-connections
        if self.mask_diagonal:
            distances.fill_diagonal_(float('inf'))
        
        # Get KNN indices
        _, nn_idx = torch.topk(distances, k=min(self.k, distances.size(1)), 
                              dim=1, largest=False)
        
        # Create adjacency matrix
        n = distances.size(0)
        adj_matrix = torch.zeros_like(distances)
        row_idx = torch.arange(n, device=distances.device).view(-1, 1).expand(-1, self.k)
        adj_matrix[row_idx.flatten(), nn_idx.flatten()] = 1.0
        
        # Normalize if configured
        if self.normalize:
            deg = adj_matrix.sum(dim=1, keepdim=True)
            adj_matrix = adj_matrix / (deg + self.eps)
            
        return adj_matrix

class Label(Kernel):
    """Creates a kernel based on label similarity."""
    
    def __init__(self,pairwise: bool = True, num_classes: Optional[int] = None,  **kwargs):
        super().__init__( **kwargs)
        self.pairwise = pairwise
        self.num_classes = num_classes
    
    def _compute(self, labels: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Compute label-based similarity matrix."""
        if isinstance(labels, list):
            labels1 = labels[0]
            labels2 = labels[1]
        else:
            labels1 = labels2 = labels
        if not self.pairwise:
            #tensor of numbers from 0 to self.num_classes
            labels2 = torch.arange(self.num_classes, device=labels2.device)
        
        adj_matrix = (labels1.unsqueeze(1) == labels2.unsqueeze(0)).float()
        
        # Remove self-loops if configured
        if self.mask_diagonal and self.pairwise:
            adj_matrix.fill_diagonal_(0)
        
        # Normalize if configured
        if self.normalize:
            row_sums = adj_matrix.sum(dim=1, keepdim=True)
            adj_matrix = adj_matrix / (row_sums + self.eps)
                
        return adj_matrix

class Augmentation(Kernel):
    """Kernel for defining relationships between augmented views of data."""
    
    def __init__(
        self,
        block: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
        **kwargs):
        """Initialize augmentation kernel with block pattern."""
        super().__init__(**kwargs)
        self.label_kernel = Labell(**kwargs)
        self.label_kernel.normalize = False
        
        if block is not None:
            self.block = block
            self.block_size = block.shape[0]
        if block_size is not None:
            self.block = torch.ones(block_size, block_size)
            self.block_size = block_size
            if block_size < 2:
                self.mask_diagonal = False

    def _compute(
        self, idx: Union[torch.Tensor, List[torch.Tensor]],
        return_log: bool = False ) -> torch.Tensor:
        """Compute augmentation kernel matrix."""
        if isinstance(idx, list):
            batch_size = idx[0].shape[0]
            device = idx[0].device
        else:
            batch_size = idx.shape[0]
            device = idx.device
            
        kernel_matrix = torch.zeros(batch_size, batch_size, device=device)
        
        # If block is provided, create block diagonal matrix
        if self.block is not None:
            block = self.block.to(device=device)
            num_blocks = batch_size // self.block_size
            if num_blocks > 0:  # Only create if we have complete blocks
                kernel_matrix[:num_blocks*self.block_size, :num_blocks*self.block_size] = \
                    torch.block_diag(*[block] * num_blocks)
                
        # If indices provided, fill with ones based on indices
        if idx is not None:
            #use labels kernel to get kernel
            kernel_matrix = self.label_kernel(idx)
            
        if self.mask_diagonal:
            kernel_matrix = kernel_matrix.fill_diagonal_(0)
            
        if self.normalize:
            if return_log:
                kernel_matrix = torch.log(kernel_matrix.clamp(min=1e-8))
                kernel_matrix = kernel_matrix - torch.logsumexp(kernel_matrix, dim=1, keepdim=True)
            else:
                kernel_matrix = F.normalize(kernel_matrix, p=1, dim=1)
            
        return kernel_matrix