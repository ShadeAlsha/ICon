# configs/model_config.py
from dataclasses import dataclass
from typing import Optional, Union, Sequence
import torch.nn as nn

@dataclass
class Config:
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
    divergence: str = 'kl'
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
    def to_loggable_dict(self):
        def describe(obj):
            if isinstance(obj, nn.Module):
                name = obj.__class__.__name__
                # Collect any non-nn.Module public attributes
                attrs = {}
                for attr_name in dir(obj):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        val = getattr(obj, attr_name)
                    except Exception:
                        continue
                    if isinstance(val, (int, float, str, bool)) or (
                        hasattr(val, 'item') and callable(val.item)  # torch.Tensor scalar
                    ):
                        try:
                            val = val.item() if hasattr(val, 'item') else val
                            attrs[attr_name] = val
                        except Exception:
                            continue
                if attrs:
                    attr_str = ", ".join(f"{k}={v}" for k, v in sorted(attrs.items()))
                    return f"{name}({attr_str})"
                else:
                    return name
            elif isinstance(obj, (list, tuple)):
                return "+".join(describe(x) for x in obj)
            return str(obj)

        return {
            'lr': self.lr,
            'optimizer': self.optimizer,
            'loss_type': self.loss_type,
            'divergence': self.divergence,
            'weight_decay': self.weight_decay,
            'ema_momentum': self.ema_momentum,
            'linear_probe': self.linear_probe,
            'num_classes': self.num_classes,
            'use_ema': self.use_ema,
            'use_mixed_precision': self.use_mixed_precision,
            'accuracy_mode': self.accuracy_mode,
            'log_icon_loss': self.log_icon_loss,
            'mapper': describe(self.mapper),
            'mapper2': describe(self.mapper2),
            'supervisory_distribution': describe(self.supervisory_distribution),
            'learned_distribution': describe(self.learned_distribution),
        }
