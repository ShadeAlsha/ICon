# kernels/distance.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
from .base import Kernel
from .utils import fill_diagonal

class DistanceKernel(Kernel):
    """Enhanced distance computation kernel with optimizations."""
    VALID_METRICS = {'euclidean', 'cosine', 'dot', 'manhattan', 'minkowski'}
    
    def __init__(self, metric: str = 'euclidean', p=2):
        super().__init__()
        self.metric = metric
        self.p = p
            
    def _compute(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Compute pairwise distances with optimizations."""
        # Process input

        if isinstance(features, list):
            x1, x2 = features
        else:
            x1 = x2 = features

        # Validate: embeddings must be 2D (B, D) - flatten should happen upstream
        # in Model._compute_loss before calling learned_distribution
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError(
                f"DistanceKernel requires 2D tensors (B, D), got shapes {x1.shape} and {x2.shape}. "
                f"Ensure embeddings are flattened before computing distances."
            )
        
        # Compute distances based on metric
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'cosine':
            return self._cosine_distance(x1, x2)
        elif self.metric == 'dot':
            return self._dot_product(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        elif self.metric == 'hyperbolic':
            return self._hyperbolic_distance2(x1, x2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optimized Euclidean distance computation."""
        #print(x1.shape, x2.shape)
        x1_norm = (x1**2).sum(1).view(-1, 1)
        x2_norm = (x2**2).sum(1).view(1, -1)
        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.t())
        return torch.clamp(dist, 0.0, float('inf'))

    def _cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optimized cosine distance computation."""
        x1_normalized = F.normalize(x1, p=self.p, dim=1)
        x2_normalized = F.normalize(x2, p=self.p, dim=1)
        return 1 - torch.mm(x1_normalized, x2_normalized.t())

    @staticmethod
    def _dot_product(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute dot product distance."""
        return -torch.mm(x1, x2.t())

    def _manhattan_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute Manhattan (L1) distance."""
        return torch.cdist(x1, x2, p=1)

    def _minkowski_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute Minkowski distance."""
        return torch.cdist(x1, x2, p=self.p)
    
    def _hyperbolic_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        #use built-in function
        ball = PoincareBall(c=1.0)
        distance = ball.dist(x1, x2)
        return distance

    def _hyperbolic_distance2(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distance using the Poincaré ball model."""
        # Norms
        x1_norm_sq = torch.sum(x1**2, dim=1, keepdim=True).clamp(max=1 - 1e-5)
        x2_norm_sq = torch.sum(x2**2, dim=1, keepdim=True).clamp(max=1 - 1e-5)
        
        # Pairwise squared Euclidean distance
        diff_sq = torch.cdist(x1, x2, p=2)**2
        
        # Expand norms for broadcasting
        x1_norm_sq = x1_norm_sq.expand(-1, x2.shape[0])
        x2_norm_sq = x2_norm_sq.T.expand(x1.shape[0], -1)

        denominator = (1 - x1_norm_sq) * (1 - x2_norm_sq)
        fraction = 2 * diff_sq / denominator.clamp(min=1e-10)
        
        dist = torch.acosh(1 + fraction.clamp(min=1 + 1e-5))
        return dist


class Gaussian(Kernel):
    def __init__(self, 
                 metric: str = 'euclidean',
                 sigma: Optional[float] = None, 
                 perplexity: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.perplexity = perplexity
        self.distance_kernel = DistanceKernel(metric=metric)
        
    def _compute(self, 
                features: Union[torch.Tensor, List[torch.Tensor]], 
                return_log: bool = False) -> torch.Tensor:
        distances = self.distance_kernel(features)
        if self.mask_diagonal:
            distances = fill_diagonal(distances,float('inf'))
            
        logits = -distances / (2 * self.sigma ** 2)
        
        if return_log:
            affinities = F.log_softmax(logits, dim=1)
        else:
            affinities = F.softmax(logits, dim=1)
        return affinities
 
    def _compute_gaussian_probs(self, distances: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian probabilities efficiently."""
        logits = -distances / (2 * sigma.view(-1, 1) ** 2)
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - logits_max
        
        exp_logits = torch.exp(logits)
        exp_sum = torch.sum(exp_logits, dim=1, keepdim=True)
        
        return exp_logits / exp_sum

class StudentT(Kernel):
    """
    Implements the Student-t kernel with an option to make gamma learnable.
    """
    def __init__(
            self,
            gamma: float = 1.0,
            df: float = 1.0,  # degrees of freedom
            learnable_gamma: bool = False,  # Option to learn gamma
            metric: str = 'euclidean',
            **kwargs):
        super().__init__(**kwargs)
        
        self.learnable_gamma = learnable_gamma  # Store flag
        
        if self.learnable_gamma:
            # Register gamma as a learnable parameter
            self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))
        else:
            # Store gamma as a constant buffer (moved with model but not trained)
            self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))
        
        # Store df as a fixed value
        self.df = df  
        self.distance_kernel = DistanceKernel(metric=metric)
        
    def _compute(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        return_log: bool = False
    ) -> torch.Tensor:
        # Compute pairwise distances
        distances = self.distance_kernel(features)

        # Mask diagonal if configured
        if self.mask_diagonal:
            distances = fill_diagonal(distances, float('inf'))
            
        # Compute kernel values
        squared_distances = distances ** 2


        if return_log:
            log_term = (self.df / 2) * (torch.log(squared_distances+1e-8)- 2*torch.log(self.gamma))
            log_denominator = torch.log1p(torch.exp(log_term))
            affinities = -log_denominator
        else:
            affinities = 1 / (1 + (squared_distances / self.gamma**2 ) ** (self.df / 2))
            
        # Normalize if configured
        if self.normalize:
            if return_log:
                affinities = affinities - torch.logsumexp(affinities, dim=1, keepdim=True)
            else:
                affinities = F.normalize(affinities, p=1, dim=1)
            
        return affinities

    def extra_repr(self) -> str:
        """Return a string with kernel parameters for printing"""
        gamma_val = self.gamma.item() if self.learnable_gamma else self.gamma
        return f'gamma={gamma_val}, df={self.df}, learnable_gamma={self.learnable_gamma}'
