from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised when torch is installed
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - soft dependency
    torch = None
    F = None
    nn = None

from .data import CNNLayerVectorConfig


TorchModuleBase = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or F is None or nn is None:
        raise ModuleNotFoundError("cnn_1d requires the optional 'torch' dependency, but torch is not installed")


def masked_mean_pool(hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
    mask = layer_mask.unsqueeze(1).to(dtype=hidden.dtype)
    valid_counts = torch.clamp(mask.sum(dim=2), min=1.0)
    return (hidden * mask).sum(dim=2) / valid_counts


def masked_max_pool(hidden: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
    mask = layer_mask.unsqueeze(1) > 0
    masked_hidden = hidden.masked_fill(~mask, float("-inf"))
    pooled = masked_hidden.max(dim=2).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class _LayerNorm1d(TorchModuleBase):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.norm = nn.LayerNorm(int(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _Conv1DBlock(TorchModuleBase):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        normalization: str,
        use_residual: bool,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        padding = max(0, (int(kernel_size) - 1) * int(dilation) // 2)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.use_residual = bool(use_residual)
        self.conv = nn.Conv1d(
            int(input_channels),
            int(output_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            dilation=int(dilation),
        )
        if str(normalization) == "layernorm":
            self.normalization = _LayerNorm1d(int(output_channels))
        elif str(normalization) == "batchnorm":
            self.normalization = nn.BatchNorm1d(int(output_channels))
        elif str(normalization) == "none":
            self.normalization = None
        else:
            raise ValueError("cnn_1d normalization must be one of {'layernorm', 'batchnorm', 'none'}")
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def _propagate_mask(self, layer_mask: torch.Tensor) -> torch.Tensor:
        assert F is not None
        pooled_mask = F.max_pool1d(
            layer_mask.unsqueeze(1),
            kernel_size=int(self.kernel_size),
            stride=int(self.stride),
            padding=int(self.padding),
            dilation=int(self.dilation),
        )
        return (pooled_mask.squeeze(1) > 0).to(dtype=layer_mask.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.conv(hidden)
        next_layer_mask = self._propagate_mask(layer_mask)
        if self.normalization is not None:
            next_hidden = self.normalization(next_hidden)
        next_hidden = self.activation(next_hidden)
        next_hidden = self.dropout(next_hidden)

        mask = next_layer_mask.unsqueeze(1).to(dtype=next_hidden.dtype)
        next_hidden = next_hidden * mask
        if (
            self.use_residual
            and hidden.shape == next_hidden.shape
            and tuple(layer_mask.shape) == tuple(next_layer_mask.shape)
        ):
            next_hidden = (next_hidden + hidden * mask) * mask
        return next_hidden, next_layer_mask


class Conv1DLayerAggregator(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        if str(config.pooling) not in {"mean", "max", "mean_max"}:
            raise ValueError("cnn_1d pooling must be one of {'mean', 'max', 'mean_max'}")

        blocks: list[_Conv1DBlock] = []
        in_channels = int(input_dim)
        for _ in range(int(config.num_conv_layers)):
            blocks.append(
                _Conv1DBlock(
                    input_channels=int(in_channels),
                    output_channels=int(config.conv_channels),
                    kernel_size=int(config.kernel_size),
                    stride=int(config.stride),
                    dilation=int(config.dilation),
                    dropout=float(config.dropout),
                    normalization=str(config.normalization),
                    use_residual=bool(config.use_residual),
                )
            )
            in_channels = int(config.conv_channels)
        self.blocks = nn.ModuleList(blocks)
        self.pooling = str(config.pooling)
        self.output_channels = int(config.conv_channels)
        self.embedding_dim = (
            int(config.conv_channels) * 2 if str(config.pooling) == "mean_max" else int(config.conv_channels)
        )

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        hidden = inputs.transpose(1, 2)
        current_mask = layer_mask
        for block in self.blocks:
            hidden, current_mask = block(hidden, current_mask)
        hidden = hidden * current_mask.unsqueeze(1).to(dtype=hidden.dtype)

        if self.pooling == "mean":
            return masked_mean_pool(hidden, current_mask)
        if self.pooling == "max":
            return masked_max_pool(hidden, current_mask)
        return torch.cat(
            [
                masked_mean_pool(hidden, current_mask),
                masked_max_pool(hidden, current_mask),
            ],
            dim=1,
        )


class _CNNLayerSequenceClassifier(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
        output_dim: int,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.aggregator = Conv1DLayerAggregator(
            input_dim=int(input_dim),
            config=config,
        )
        self.output_dim = int(output_dim)
        self.head = nn.Linear(int(self.aggregator.embedding_dim), int(self.output_dim))

    def extract_features(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        return self.aggregator(inputs, layer_mask)

    def forward(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_features(inputs, layer_mask)
        logits = self.head(embedding)
        if self.output_dim == 1:
            return logits.squeeze(1)
        return logits


class _GradientReverseFunction(torch.autograd.Function if torch is not None else object):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -float(ctx.lambda_value) * grad_output, None


def _gradient_reverse(inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
    _require_torch()
    return _GradientReverseFunction.apply(inputs, float(lambda_value))


class _CNNLayerSequenceDANNClassifier(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        config: CNNLayerVectorConfig,
        output_dim: int,
        domain_output_dim: int,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None
        self.aggregator = Conv1DLayerAggregator(
            input_dim=int(input_dim),
            config=config,
        )
        self.output_dim = int(output_dim)
        self.domain_output_dim = int(domain_output_dim)
        if self.domain_output_dim < 2:
            raise ValueError("cnn_1d_dann requires at least two rank-domain classes")
        embedding_dim = int(self.aggregator.embedding_dim)
        self.head = nn.Linear(embedding_dim, int(self.output_dim))
        self.domain_head = nn.Linear(embedding_dim, int(self.domain_output_dim))

    def extract_features(self, inputs: torch.Tensor, layer_mask: torch.Tensor) -> torch.Tensor:
        return self.aggregator(inputs, layer_mask)

    def forward(
        self,
        inputs: torch.Tensor,
        layer_mask: torch.Tensor,
        *,
        lambda_value: float | None = None,
        return_domain: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.extract_features(inputs, layer_mask)
        label_logits = self.head(embedding)
        if self.output_dim == 1:
            label_logits = label_logits.squeeze(1)
        if not return_domain:
            return label_logits
        reversed_embedding = _gradient_reverse(embedding, float(lambda_value or 0.0))
        domain_logits = self.domain_head(reversed_embedding)
        return label_logits, domain_logits
