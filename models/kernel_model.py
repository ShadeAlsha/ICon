import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union, Any
from .losses import KernelLoss
from .ema import EMAUpdater
from .model_config import IConConfig
from .metrics import UnsupervisedAccuracy, Accuracy

class IConModel(pl.LightningModule):
    def __init__(self, config: IConConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self._setup_model()
        self._setup_metrics()
        self._setup_monitoring()

    def _setup_model(self):
        self.mapper = self.config.mapper
        if self.mapper is not None and isinstance(self.mapper, list):
            self.mapper = ConcatModels(self.mapper)
            
        self.supervisory_distribution = self.config.supervisory_distribution
        self.learned_distribution = self.config.learned_distribution
        
        if self.config.use_ema:
            self.ema = EMAUpdater(self.mapper, self.config.ema_momentum)

        self.linear_probe = (
            nn.Linear(self.mapper.output_dim, self.config.num_classes)
            if self.config.linear_probe else nn.Identity()
        )

        self.kl_divergence = KernelLoss.get_loss_fn(self.config.loss_type)

    def _setup_metrics(self):
        if self.config.accuracy_mode == 'regular' and self.config.num_classes:
            self.train_acc = Accuracy(num_classes=self.config.num_classes, ignore_index=-1)
            self.val_acc = Accuracy(num_classes=self.config.num_classes, ignore_index=-1)
        elif self.config.accuracy_mode == 'unsupervised' and self.config.num_classes:
            self.train_acc = UnsupervisedAccuracy(n_classes=self.config.num_classes)
            self.val_acc = UnsupervisedAccuracy(n_classes=self.config.num_classes)
        else:
            self.train_acc = self.val_acc = None

        self.grad_norm = torch.tensor(0.0)

    def _setup_monitoring(self):
        self.automatic_optimization = not self.config.use_mixed_precision
        self.validation_outputs = []

    def _compute_loss(self, batch) -> Dict[str, Any]:
        supervisory_distribution = self.supervisory_distribution(batch)
        mapper_output = self.mapper(batch) #dictionary
        batch.update(mapper_output)
        learned_distribution = self.learned_distribution(batch, return_log=self.config.log_icon_loss)
        embeddings = batch.get('embeddings', None) 
        
        losses = {
            'icon_loss': self.kl_divergence(supervisory_distribution, learned_distribution, log=self.config.log_icon_loss),
            'linear_probe_loss': self._compute_linear_probe_loss(embeddings.detach(), batch.get('label', None))
        }

        if any(torch.isnan(loss) for loss in losses.values()):
            raise ValueError("NaN loss detected")

        return {
            'losses': losses,
            'metrics': {
                'embeddings': embeddings,
                'supervisory_distribution': supervisory_distribution,
                'learned_distribution': torch.exp(learned_distribution) if self.config.log_icon_loss else learned_distribution,
                'logits': self.linear_probe(embeddings),
                'labels': batch.get('label', None),
            }
        }

    def _compute_linear_probe_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.config.linear_probe:
            if embeddings is not None and labels is not None:
                logits = self.linear_probe(embeddings.detach())
                loss = F.cross_entropy(logits, labels, ignore_index=-1)
                if not torch.isnan(loss):
                    return loss
        return torch.tensor(0.0, device=labels.device)

    def configure_optimizers(self):
        param_groups = [
            {'params': self.mapper.parameters(), 'lr': self.config.lr},
        ]

        if self.config.linear_probe:
            param_groups.append({
                'params': self.linear_probe.parameters(),
                'lr': self.config.lr * 5
            })

        if hasattr(self.learned_distribution, 'learnable_gamma') and self.learned_distribution.learnable_gamma:
            param_groups.append({
                'params': [self.learned_distribution.gamma],
                'lr': 0.001 * self.config.lr
            })

        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_groups, weight_decay=self.config.weight_decay, momentum=0)
        else:
            raise ValueError(f"Optimizer {self.config.optimizer} not supported")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        results = self._compute_loss(batch)
        loss = sum(results['losses'].values())

        if self.config.use_mixed_precision:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)
            if self.config.gradient_clip_val > 0:
                self.grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.config.gradient_clip_val
                )
            optimizer.step()

        if self.config.use_ema:
            self.ema.update()

        self._log_metrics('train', results, loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict:
        if self.config.use_ema:
            with self.ema.average_parameters():
                results = self._compute_loss(batch)
        else:
            results = self._compute_loss(batch)

        loss = sum(results['losses'].values())
        self._log_metrics('val', results, loss)

        metrics = results['metrics']
        return {
            'embeddings': metrics['embeddings'].detach().cpu(),
            'logits': metrics['logits'].detach().cpu(),
            'labels': metrics['labels'].detach().cpu(),
            'learned_distribution': results['metrics']['learned_distribution'].clip(1e-10).detach().cpu(),
            'supervisory_distribution': results['metrics']['supervisory_distribution'].clip(1e-10).detach().cpu(),
        }

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_acc, UnsupervisedAccuracy):
            accuracy = self.train_acc.compute()
            self.log('train_accuracy', accuracy, prog_bar=True)
            self.train_acc.reset()


    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return

        outputs = {}
        keys = self.validation_outputs[0].keys()
        for key in keys:
            tensors = [batch[key] for batch in self.validation_outputs]
            outputs[key] = torch.cat(tensors, dim=0)

        self.aggregated_val_outputs = (
            outputs['embeddings'],
            outputs['logits'],
            outputs['labels'],
            outputs['learned_distribution'],
            outputs['supervisory_distribution']
        )

        if isinstance(self.val_acc, UnsupervisedAccuracy):
            accuracy = self.val_acc.compute()
            self.log('val_accuracy', accuracy, prog_bar=True)
            self.val_acc.reset()

        # Clear stored outputs for next epoch
        self.validation_outputs.clear()


    def _log_metrics(self, phase: str, results: Dict, loss: torch.Tensor) -> None:
        self.log(f'{phase}_loss', loss, prog_bar=True)
        for loss_name, loss_value in results['losses'].items():
            self.log(f'{phase}_{loss_name}', loss_value)

        if accuracy_metric := getattr(self, f'{phase}_acc'):
            logits = results['metrics']['logits']
            labels = results['metrics']['labels']
            if isinstance(accuracy_metric, UnsupervisedAccuracy):
                accuracy_metric.update(logits.argmax(dim=-1), labels)
            else:
                accuracy_metric(logits.argmax(dim=-1), labels)
                self.log(f'{phase}_accuracy', accuracy_metric, prog_bar=True)

        if phase == 'train':
            self.log('grad_norm', self.grad_norm)

        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        for i, opt in enumerate(opts):
            for j, group in enumerate(opt.param_groups):
                self.log(f'lr_group_{i}_{j}', group['lr'])


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.config.use_ema:
            checkpoint['ema_state'] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.config.use_ema and 'ema_state' in checkpoint:
            self.ema.shadow = checkpoint['ema_state']

    def get_progress_bar_dict(self) -> Dict[str, Union[int, float, str]]:
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        if hasattr(self, 'grad_norm'):
            items['grad'] = f'{self.grad_norm:.3f}'
        return items


class ConcatModels(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        out = {}
        for module in self.modules_list:
            out.update(module(x))  # assumes each returns a dict
        return out

class SeqModels(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        out = {}
        for module in self.modules_list:
            x = module(x) #x is a dictionary
            out.update(x)
        return out