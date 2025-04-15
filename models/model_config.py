# configs/model_config.py
from dataclasses import dataclass
from typing import Optional, Union, Sequence
import torch.nn as nn

@dataclass
class IConConfig:
    """Configuration class for KernelModel with validation."""
    mapper: Union[nn.Module, Sequence[nn.Module]]
    supervisory_distribution: nn.Module
    learned_distribution: nn.Module
    mapper2: Optional[nn.Module] = None
    num_classes: Optional[int] = None
    
    lr: float = 5e-4
    accuracy_mode: Optional[str] = None
    use_ema: bool = False
    ema_momentum: float = 0.999
    loss_type: str = 'ce'
    decay_factor: float = 0.9
    linear_probe: bool = False
    optimizer: str = 'adamw'
    weight_decay: float = 0.0
    gradient_clip_val: float = 10.0
    use_mixed_precision: bool = False
    log_icon_loss:bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0 <= self.ema_momentum <= 1:
            raise ValueError("ema_momentum must be between 0 and 1")
        if not 0 < self.lr <= 1:
            raise ValueError("Learning rate must be between 0 and 1")