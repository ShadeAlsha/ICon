# models/components/ema.py
import torch.nn as nn
from contextlib import contextmanager

class EMAUpdater:
    """Manages Exponential Moving Average updates."""
    
    def __init__(self, model: nn.Module, momentum: float):
        self.model = model
        self.momentum = momentum
        self.shadow = {}
        self.backup = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize EMA shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.shadow[name] * self.momentum +
                    param.data * (1 - self.momentum)
                )
                
    @contextmanager
    def average_parameters(self):
        """Context manager for using EMA parameters during evaluation."""
        self.backup = {name: param.data.clone() 
                      for name, param in self.model.named_parameters()
                      if param.requires_grad}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
                
        try:
            yield
        finally:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.backup[name]