import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class EmbeddingLogger(Callback):
    def __init__(self, log_dir="embeddings"):
        super().__init__()
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Collect and log embeddings at the end of validation."""
        if not hasattr(pl_module, "aggregated_val_outputs"):
            raise AttributeError("pl_module must have `aggregated_val_outputs` containing embeddings and labels.")

        # Extract embeddings and labels from aggregated validation outputs
        embeddings, logits, labels, learned_kernel, target_kernel = pl_module.aggregated_val_outputs

        # Save embeddings for future analysis
        save_path = f"{self.log_dir}/embeddings_epoch_{trainer.current_epoch}.pt"
        torch.save({"embeddings": embeddings.cpu(), "labels": labels.cpu()}, save_path)

        print(f"✅ Embeddings logged and saved at {save_path}")
