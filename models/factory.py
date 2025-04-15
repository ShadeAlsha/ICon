# utils/factory.py
import pytorch_lightning as pl
from typing import Dict, Any
import logging
from .kernel_model import KernelModel
from .model_config import KernelModelConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating KernelModel instances with validated configs."""
    
    @staticmethod
    def create_model(config_dict: Dict[str, Any]) -> KernelModel:
        """Create a KernelModel instance from a configuration dictionary."""
        try:
            # Validate required components
            required_keys = ['mapper', 'target_kernel', 'learned_kernel']
            if not all(key in config_dict for key in required_keys):
                raise ValueError(f"Missing required configuration keys: {required_keys}")
                
            # Create configuration
            config = KernelModelConfig(**config_dict)
            
            # Create and return model
            return KernelModel(config)
            
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
            
    @staticmethod
    def create_training_config(model: KernelModel) -> Dict[str, Any]:
        """Create PyTorch Lightning Trainer configuration."""
        return {
            'gradient_clip_val': model.config.gradient_clip_val,
            'precision': '16-mixed' if model.config.use_mixed_precision else 32,
            'callbacks': [
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=3,
                    filename='{epoch}-{val_loss:.2f}'
                ),
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min'
                ),
                pl.callbacks.LearningRateMonitor(logging_interval='step')
            ],
            'logger': pl.loggers.TensorBoardLogger('logs/', name='kernel_model')
        }