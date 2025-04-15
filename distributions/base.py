# kernels/base.py
import torch
import torch.nn as nn
from typing import Union, List, Optional, Dict, Tuple
from dataclasses import dataclass, fields


class Kernel(nn.Module):
    def __init__(
        self,
        mask_diagonal: bool = False,
        normalize: bool = False,
        eps: float = 0.0,
        input_key: Optional[str] = None,
        input_key2: Optional[str] = None,
    ):
        super().__init__()
        self.mask_diagonal = mask_diagonal
        self.eps = eps
        self.input_key = input_key
        self.input_key2 = input_key2

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle dictionary-based input automatically
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            if self.input_key2:
                x = [batch[self.input_key], batch[self.input_key2]]
            elif self.input_key:
                x = batch[self.input_key]
            else:
                raise ValueError("No input_key(s) specified for dict input.")
            return self._compute(x, **kwargs)
        else:
            return self._compute(*args, **kwargs)

    def _compute(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
        
    def __add__(self, other: Union['Kernel', float, int]) -> 'Kernel':
        if isinstance(other, Kernel):
            return CompositeKernel([self, other], operation='add')
        elif isinstance(other, (int, float)):
            return ConstantAddedKernel(self, other)
        raise ValueError(f"Unsupported addition with type {type(other)}")
    
    def __radd__(self, other: Union['Kernel', float]) -> 'Kernel':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Kernel', float, int]) -> 'Kernel':
        if isinstance(other, Kernel):
            return CompositeKernel([self, other], operation='sub')
        elif isinstance(other, (int, float)):
            return ConstantAddedKernel(self, -other)
        raise ValueError(f"Unsupported subtraction with type {type(other)}")
    
    def __mul__(self, scalar: float) -> 'Kernel':
        return ScaledKernel(self, scalar)
    
    def __rmul__(self, scalar: float) -> 'Kernel':
        return self.__mul__(scalar)
    
    def normalize(self, eps: float = 0.0) -> 'Kernel':
        return NormalizedKernel(self, eps)
    
    def binarize(self, threshold: float = 0.0) -> 'Kernel':
        return BinarizedKernel(self, threshold)
    
    def leak(self, alpha: float) -> 'Kernel':
        return LeakKernel(self, alpha)
    
class CompositeKernel(Kernel):
    """Combines multiple kernels with various operations."""
    
    VALID_OPERATIONS = {'add', 'sub', 'max', 'compose', 'mul'}
    
    def __init__(self, kernels: List[Kernel], operation: str = 'add'):
        super().__init__()
        if operation not in self.VALID_OPERATIONS:
            raise ValueError(f"Operation must be one of {self.VALID_OPERATIONS}")
        self.kernels = nn.ModuleList(kernels)
        self.operation = operation
    
    def _compute(self, *args, **kwargs) -> torch.Tensor:
        if self.operation == 'add':
            return sum(kernel(*args, **kwargs) for kernel in self.kernels)
        
        elif self.operation == 'sub':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = result - kernel(*args, **kwargs)
            return result
        
        elif self.operation == 'max':
            kernel_outputs = [kernel(*args, **kwargs) for kernel in self.kernels]
            return torch.max(torch.stack(kernel_outputs), dim=0)[0]
        
        elif self.operation == 'compose':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = kernel(result)
            return result
            
        elif self.operation == 'mul':
            result = self.kernels[0](*args, **kwargs)
            for kernel in self.kernels[1:]:
                result = result * kernel(*args, **kwargs)
            return result

class ScaledKernel(Kernel):
    """Scales a kernel by a constant factor."""
    
    def __init__(self, kernel: Kernel, scalar: float):
        super().__init__()
        self.kernel = kernel
        self.scalar = scalar
    
    def _compute(self, *args, **kwargs) -> torch.Tensor:
        return self.scalar * self.kernel(*args, **kwargs)

class ConstantAddedKernel(Kernel):
    """Adds a constant to a kernel."""
    
    def __init__(self, kernel: Kernel, constant: float):
        super().__init__()
        self.kernel = kernel
        self.constant = constant
    
    def _compute(self, *args, **kwargs) -> torch.Tensor:
        return self.kernel(*args, **kwargs) + self.constant

class NormalizedKernel(Kernel):
    """Row-normalizes a kernel matrix."""
    
    def __init__(self, kernel: Kernel, eps: float):
        super().__init__()
        self.kernel = kernel
    
    def _compute(self, *args, **kwargs) -> torch.Tensor:
        kernel_output = self.kernel(*args, **kwargs)
        if isinstance(kernel_output, torch.Tensor):
            row_sums = kernel_output.sum(dim=1, keepdim=True)
            return kernel_output / (row_sums + self.eps)
        else:
            raise ValueError("Kernel output must be a tensor for normalization")
        

class BinarizedKernel(Kernel):
    """Binarizes a kernel matrix."""
    
    def __init__(self, kernel: Kernel, threshold: float):
        super().__init__()
        self.kernel = kernel
        self.threshold = threshold
    
    def _compute(self, *args, **kwargs) -> torch.Tensor:
        kernel_output = self.kernel(*args, **kwargs)
        binary_output = (kernel_output > self.threshold).float()
        return binary_output

class LeakKernel(Kernel):
    def __init__(self, kernel: Kernel, alpha: float):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.kernel = kernel
        self.alpha = alpha
    
    def _compute(self, *args, **kwargs) -> torch.Tensor:
        # Compute base kernel
        kernel_output = self.kernel(*args, **kwargs)
        
        # Create uniform component
        n = kernel_output.size(1)
        uniform_kernel = torch.ones_like(kernel_output) / n
        
        # Combine with leakage
        leaked_kernel = (1 - self.alpha) * kernel_output + self.alpha * uniform_kernel
        return leaked_kernel
