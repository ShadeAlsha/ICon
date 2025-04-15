# kernels/clustering.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
from .base import Kernel
from .utils import fill_diagonal

class UniformCluster(Kernel):
    """Creates a kernel based on cluster assignment probabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _compute(self, cluster_probs: Union[torch.Tensor, List[torch.Tensor]], return_log: bool = False) -> torch.Tensor:
        """Compute clustering-based similarity matrix.
        
        Args:
            cluster_probs: Cluster assignment probabilities or list of [probs1, probs2]
            labels: Optional labels (not used)
            idx: Optional indices (not used)
        
        Returns:
            Kernel matrix based on cluster assignments
        """
        # Handle input format
        if isinstance(cluster_probs, list) and len(cluster_probs) == 2:
            probs1, probs2 = cluster_probs
        else:
            probs1 = probs2 = cluster_probs
        
        # Compute cluster sizes
        cluster_sizes = probs1.sum(dim=0)
        
        # Compute normalized kernel
        kernel = (probs1 / (cluster_sizes + self.eps)) @ probs2.t()
        
        # Apply masking if configured
        if self.mask_diagonal:
            kernel = fill_diagonal(kernel, 0.0)
                
        if return_log:
            return kernel.log()
        return kernel
