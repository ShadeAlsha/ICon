"""
Utility functions and classes for the plot system.
"""
import os
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Any, Type, Callable
from abc import ABC, abstractmethod
import matplotlib as mpl
from PIL import Image as PILImage


# A high contrast color palette for visualizations
high_contrast_colors = [
    "#7F3C8D", "#1CA878", "#3969AC", "#F2B701", "#D63F6C",
    "#A0C95A", "#E68310", "#008695", "#CF1C90", "#005082"
]


class PlotConfig:
    """Configuration settings for plots."""
    
    def __init__(
        self,
        backend: str = "matplotlib",  # "matplotlib" or "plotly"
        figure_size: tuple = (8, 6),
        dpi: int = 100,
        theme: str = "default"
    ):
        self.backend = backend
        self.figure_size = figure_size
        self.dpi = dpi
        self.theme = theme


class BasePlot(ABC):
    """Abstract base class for all plot types."""
    
    def __init__(self, config: PlotConfig, required_keys: Optional[List[str]] = None):
        self.config = config
        self.name = self.__class__.__name__.replace("Plot", "").lower()
        self.required_keys = required_keys or []
    
    def plot(self, data: Dict[str, Any]):
        """Create a plot based on the provided data using the configured backend."""
        self._validate_data(data)
        
        if self.config.backend == "plotly":
            return self.plot_plotly(data)
        elif self.config.backend == "matplotlib":
            return self.plot_matplotlib(data)
        else:
            raise ValueError(f"Unsupported backend {self.config.backend}")
    
    def _validate_data(self, data: Dict[str, Any]):
        """Validate that all required keys are present in the data."""
        missing_keys = [key for key in self.required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys for {self.name} plot: {missing_keys}")
    
    @abstractmethod
    def plot_plotly(self, data: Dict[str, Any]):
        """Create a plotly figure from the data."""
        pass
    
    @abstractmethod
    def plot_matplotlib(self, data: Dict[str, Any]):
        """Create a matplotlib figure from the data."""
        pass


class PlotRegistry:
    """Registry for available plot types."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a plot class in the registry."""
        def decorator(plot_class):
            key = name or plot_class.__name__
            cls._registry[key] = plot_class
            return plot_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BasePlot]:
        """Get a plot class by name."""
        if name not in cls._registry:
            raise ValueError(f"Plot type '{name}' not found in registry")
        return cls._registry[name]
    
    @classmethod
    def list_plots(cls) -> List[str]:
        """List all registered plot types."""
        return list(cls._registry.keys())


class PlotUtils:
    @staticmethod
    def set_style():
        mpl.rcParams.update({
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 200,
            "figure.titlesize": 14
        })
    @staticmethod
    def save_figure(fig, filepath: str, backend: str = "matplotlib"):
        """Save a figure to a file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        if backend == "plotly":
            fig.write_image(filepath, scale=2)
        elif backend == "matplotlib":
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    @staticmethod
    def fig_to_tensor(fig, backend: str = "matplotlib") -> torch.Tensor:
        """Convert a figure to a PyTorch tensor for logging."""
        if backend == "matplotlib":
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img = np.array(PILImage.open(buf).convert("RGB"))
            buf.close()
            plt.close(fig)

        elif backend == "plotly":
            img_bytes = fig.to_image(format="png", scale=2)
            buf = io.BytesIO(img_bytes)
            img = np.array(PILImage.open(buf).convert("RGB"))
            buf.close()

        else:
            raise ValueError(f"Unsupported backend {backend}")
            
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
