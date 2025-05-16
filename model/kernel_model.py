import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Union, Any
from .losses import KernelLoss
from .ema import EMAUpdater
from .model_config import Config
from .metrics import UnsupervisedAccuracy, Accuracy

class Model(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(config.to_loggable_dict())
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

    def _compute_loss(self, batch) -> Dict[str, Any]:
        supervisory_distribution = self.supervisory_distribution(batch)
        mapper_output = self.mapper(batch) #dictionary
        batch.update(mapper_output)
        learned_distribution = self.learned_distribution(batch, return_log=self.config.log_icon_loss)
        embeddings = batch.get('embedding', None) 
        
        losses = {
            'icon_loss': self.kl_divergence(supervisory_distribution, learned_distribution, log=self.config.log_icon_loss),
            'linear_probe_loss': self._compute_linear_probe_loss(embeddings.detach(), batch.get('label', None))
        }

        if any(torch.isnan(loss) for loss in losses.values()):
            raise ValueError("NaN loss detected")

        return {
            'losses': losses,
            'metrics': {
                'embedding': embeddings,
                'supervisory_distribution': supervisory_distribution,
                'learned_distribution': torch.exp(learned_distribution) if self.config.log_icon_loss else learned_distribution,
                'logits': self.linear_probe(embeddings),
                'label': batch.get('label', None),
                'index': batch.get('index', None),
            }
        }

    def _compute_linear_probe_loss(self, embeddings: Union[torch.Tensor, None], labels: torch.Tensor) -> torch.Tensor:
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
        return optimizer
        #return {
        #    "optimizer": optimizer,
        #    "lr_scheduler": {
        #        "scheduler": scheduler,
        #        "interval": "epoch",
        #        "frequency": 1
        #    }
        #}

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        results = self._compute_loss(batch)
        loss = sum(results['losses'].values())

        if self.config.use_mixed_precision:
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(loss)

            from collections import defaultdict

            layer_grad_sums = defaultdict(list)
            for name, param in self.named_parameters():
                if param.grad is not None and 'linear_probe' not in name:
                    # Use full module path minus the param name (e.g., remove '.weight' or '.bias')
                    layer_name = ".".join(name.split('.')[:-1])
                    layer_grad_sums[layer_name].append(param.grad.norm(2))

            layer_grad_norms = {
                layer: torch.norm(torch.stack(norms)) for layer, norms in layer_grad_sums.items() if norms
            }

            for layer, norm in layer_grad_norms.items():
                self.log(f'grad_norm/{layer}', norm, prog_bar=False)

            if layer_grad_norms:
                total_norm = torch.norm(torch.stack(list(layer_grad_norms.values())))
            else:
                total_norm = torch.tensor(0.0, device=self.device)

            self.grad_norm = total_norm


            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_val)

            optimizer.step()

        if self.config.use_ema:
            self.ema.update()

        self._log_metrics('train', results, loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict:
        if self.config.use_ema:
            with self.ema.average_parameters():
                results = self._compute_loss(batch)
        else:
            results = self._compute_loss(batch)

        loss = sum(results['losses'].values())
        self._log_metrics('val', results, loss)

        metrics = results['metrics']
        
        summary = {
            'embedding': metrics['embedding'].detach().cpu(),
            'logits': metrics['logits'].detach().cpu(),
            'index': metrics['index'].detach().cpu(),
            'label': metrics['label'].detach().cpu(),
            'learned_distribution': results['metrics']['learned_distribution'].clip(1e-10).detach().cpu(),
            'supervisory_distribution': results['metrics']['supervisory_distribution'].clip(1e-10).detach().cpu(),
        }
        return summary

    def on_train_epoch_end(self) -> None:
        if isinstance(self.train_acc, UnsupervisedAccuracy):
            accuracy = self.train_acc.compute()
            self.log('train_accuracy', accuracy, prog_bar=True)
            self.train_acc.reset()


    def on_validation_epoch_end(self) -> None:
        if isinstance(self.val_acc, UnsupervisedAccuracy):
            accuracy = self.val_acc.compute()
            self.log('val_accuracy', accuracy, prog_bar=True)
            self.val_acc.reset()




    def _log_metrics(self, phase: str, results: Dict, loss: torch.Tensor) -> None:
        self.log(f'{phase}_loss', loss, prog_bar=True)
        for loss_name, loss_value in results['losses'].items():
            self.log(f'{phase}_{loss_name}', loss_value)

        if accuracy_metric := getattr(self, f'{phase}_acc'):
            logits = results['metrics']['logits']
            labels = results['metrics']['label']
            if isinstance(accuracy_metric, UnsupervisedAccuracy):
                accuracy_metric.update(logits.argmax(dim=-1), labels)
            else:
                accuracy_metric(logits.argmax(dim=-1), labels)
                self.log(f'{phase}_accuracy', accuracy_metric, prog_bar=True)

        if phase == 'train' and self.grad_norm is not None:
            self.log('grad_norm', self.grad_norm, prog_bar=True)

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