from typing import Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseMapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_key: Union[str, List[str]] = "image",
        output_key: Union[str, List[str]] = "embeddings"
    ):
        super().__init__()
        self.model = model
        self.input_key = input_key
        self.output_key = output_key

        if isinstance(input_key, list) != isinstance(output_key, list):
            raise ValueError("input_key and output_key must both be strings or both be lists")
        if isinstance(input_key, list):
            assert len(input_key) == len(output_key), "input_key and output_key must be the same length"

    def extract_input(self, batch, key: str):
        if isinstance(batch, dict):
            return batch[key]
        return batch

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if isinstance(self.input_key, list):
            return {
                ok: self.forward_single(self.extract_input(batch, ik))
                for ik, ok in zip(self.input_key, self.output_key)
            }
        else:
            return {self.output_key: self.forward_single(self.extract_input(batch, self.input_key))}

class WrappedMapper(BaseMapper):
    def __init__(
        self,
        model: nn.Module,
        input_key: Union[str, List[str]] = "image",
        output_key: Union[str, List[str]] = "embeddings",
        normalize: bool = False
    ):
        super().__init__(model, input_key, output_key)
        self.normalize = normalize

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if self.normalize:
            x = F.normalize(x, dim=1)
        return x

class SimpleCNN(WrappedMapper):
    def __init__(
        self,
        output_dim=2,
        softmax=False,
        normalize_feats=False,
        unit_sphere=False,
        poincare_ball=False,
        input_key="image",
        output_key="embeddings"
    ):
        super().__init__(nn.Identity(), input_key, output_key)
        self.output_dim = output_dim
        self.softmax = softmax
        self.normalize_feats = normalize_feats
        self.unit_sphere = unit_sphere
        self.poincare_ball = poincare_ball

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_dim)


    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x) + x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x) + x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x) + x)

        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.softmax:
            x = F.softmax(x, dim=1)
        if self.normalize_feats:
            mean, std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True) + 1e-5
            x = (x - mean)
            if std.min().item() > 1:
                x = x / std
        if self.unit_sphere:
            x = F.normalize(x, p=2, dim=1)
        if self.poincare_ball:
            x = 5 * x
            x = self.ball.projx(x)

        return x

from torchvision import models

class ResNet(WrappedMapper):
    def __init__(self,
        model_type="resnet50",
        small_image=False,
        input_key="image",
        output_key="embedding",
        feat_dim=256,    
    ):
        if model_type == "resnet18":
            model = models.resnet18(weights=None)
            output_dim = 512
        elif model_type == "resnet34":
            model = models.resnet34(weights=None)
            output_dim = 512
        elif model_type == "resnet50":
            model = models.resnet50(weights=None)
            output_dim = 2048
        else:
            raise ValueError("Unknown ResNet type")

        if small_image:
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Identity()
        mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, feat_dim)
        )
        model.fc = mlp

        super().__init__(model, input_key=input_key, output_key=output_key)
        self.output_dim = feat_dim
    def forward_single(self, x):
        return super().forward_single(x)

class OneHotEncoder(BaseMapper):
    def __init__(
        self,
        num_classes: int,
        input_key: Union[str, List[str]] = "label",
        output_key: Union[str, List[str]] = "onehot",
        fixed: bool = False
    ):
        model = nn.Identity()  # forward_single will override
        super().__init__(model, input_key, output_key)
        self.num_classes = num_classes
        self.fixed = fixed

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed:
            return torch.eye(self.num_classes, device=x.device)
        return F.one_hot(x, num_classes=self.num_classes).float()

class LookUpTable(BaseMapper):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        normalize: bool = False,
        input_key: Union[str, List[str]] = "index",
        output_key: Union[str, List[str]] = "embedding",
        init_weights: torch.Tensor = None,
        get_all: bool = False,
    ):
        model = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        super().__init__(model, input_key, output_key)
        self.normalize = normalize
        self._output_dim = embedding_dim
        self.get_all = get_all

        if init_weights is not None:
            with torch.no_grad():
                model.weight.copy_(init_weights)

    @property
    def output_dim(self):
        return self._output_dim

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(1)
        if self.get_all:
            emb = self.model.weight
        else:   
            emb = self.model(x)
        if self.normalize:
            emb = F.normalize(emb, dim=1)
        return emb

    def update(self, x: torch.Tensor) -> None:
        pass  # No-op for standard embeddings


class MLPMapper(WrappedMapper):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: List[int] = [512, 512, 2000],
        output_dim: int = 2,
        softmax: bool = False,
        normalize: bool = False,
        input_key: Union[str, List[str]] = "image",
        output_key: Union[str, List[str]] = "embeddings"
    ):
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        model = nn.Sequential(*layers)

        super().__init__(model, input_key=input_key, output_key=output_key, normalize=normalize)

        self._output_dim = output_dim
        self.softmax = softmax

    @property
    def output_dim(self):
        return self._output_dim

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.softmax:
            x = F.softmax(x, dim=1)
        if self.normalize:
            x = F.normalize(x, dim=1)

        return x

def gather_batch_tensors(
    dataloader: torch.utils.data.DataLoader,
    input_key: str = "image",
    index_key: str = "index",
    num_samples: int = None) -> torch.Tensor:
    output_shape = None

    # First pass to get shape if not known
    if num_samples is None:
        num_samples = len(dataloader.dataset)

    for batch in dataloader:
        sample = batch[input_key]
        output_shape = sample.shape[1:]  # exclude batch dim
        break

    full_tensor = torch.empty((num_samples, *output_shape), dtype=torch.float32)

    for batch in dataloader:
        values = batch[input_key].cpu()
        indices = batch[index_key].view(-1).cpu()  # Flatten in case shape is [B, 1]

        full_tensor[indices] = values

    return full_tensor
