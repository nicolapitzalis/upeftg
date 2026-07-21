from __future__ import annotations

try:  # pragma: no cover - exercised when torch is installed
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - soft dependency
    torch = None
    nn = None


TRANSFORMER_POOLING_MEAN = "mean"
TRANSFORMER_POOLING_MEAN_MAX = "mean_max"
SUPPORTED_TRANSFORMER_POOLING = {
    TRANSFORMER_POOLING_MEAN,
    TRANSFORMER_POOLING_MEAN_MAX,
}
TorchModuleBase = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("transformer requires the optional 'torch' dependency, but torch is not installed")


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    mask_bool = mask.to(dtype=torch.bool)
    mask_float = mask_bool.to(dtype=hidden.dtype).unsqueeze(-1)
    count = torch.clamp(mask_float.sum(dim=dim), min=1.0)
    return (hidden * mask_float).sum(dim=dim) / count


def _masked_max(hidden: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    mask_bool = mask.to(dtype=torch.bool)
    pooled = hidden.masked_fill(~mask_bool.unsqueeze(-1), float("-inf")).max(dim=dim).values
    valid = mask_bool.any(dim=dim).unsqueeze(-1)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


class _TransformerHierarchyClassifier(TorchModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        nhead: int,
        slot_num_layers: int,
        layer_num_layers: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
        pooling: str,
        max_layers: int,
        max_slots: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        _require_torch()
        assert nn is not None

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.output_dim = int(output_dim)
        self.max_layers = int(max_layers)
        self.max_slots = int(max_slots)
        self.pooling = str(pooling)
        if self.pooling not in SUPPORTED_TRANSFORMER_POOLING:
            raise ValueError(
                f"transformer pooling must be one of {sorted(SUPPORTED_TRANSFORMER_POOLING)}, got {self.pooling!r}"
            )
        self.input_projection = (
            nn.Identity() if int(input_dim) == int(d_model) else nn.Linear(int(input_dim), int(d_model))
        )
        self.layer_embedding = nn.Embedding(int(max_layers), int(d_model))
        self.slot_embedding = nn.Embedding(int(max_slots), int(d_model))

        def build_encoder(num_layers: int) -> nn.TransformerEncoder | None:
            if int(num_layers) <= 0:
                return None
            layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=bool(norm_first),
            )
            return nn.TransformerEncoder(layer, num_layers=int(num_layers))

        self.slot_encoder = build_encoder(slot_num_layers)
        self.layer_encoder = build_encoder(layer_num_layers)
        self.slot_pool_projection = (
            nn.Sequential(
                nn.Linear(2 * int(d_model), int(d_model)),
                nn.GELU(),
            )
            if self.pooling == TRANSFORMER_POOLING_MEAN_MAX
            else None
        )
        self.head = nn.Linear(int(self.embedding_dim), int(output_dim))

    @property
    def embedding_dim(self) -> int:
        if self.pooling == TRANSFORMER_POOLING_MEAN_MAX:
            return 2 * int(self.d_model)
        return int(self.d_model)

    def _pool_slots(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = _masked_mean(hidden, mask, dim=2)
        if self.pooling == TRANSFORMER_POOLING_MEAN:
            return mean
        if self.slot_pool_projection is None:
            raise RuntimeError("transformer mean_max slot projection is not initialized")
        maximum = _masked_max(hidden, mask, dim=2)
        return self.slot_pool_projection(torch.cat([mean, maximum], dim=-1))

    def _pool_layers(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = _masked_mean(hidden, mask, dim=1)
        if self.pooling == TRANSFORMER_POOLING_MEAN:
            return mean
        maximum = _masked_max(hidden, mask, dim=1)
        return torch.cat([mean, maximum], dim=-1)

    def extract_features(
        self,
        values: torch.Tensor,
        slot_mask: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_layers, max_slots, _input_dim = values.shape
        if int(max_layers) > int(self.max_layers) or int(max_slots) > int(self.max_slots):
            raise ValueError(
                "transformer checkpoint layout cannot encode this bundle shape: "
                f"bundle layers/slots=({int(max_layers)}, {int(max_slots)}), "
                f"checkpoint max=({int(self.max_layers)}, {int(self.max_slots)})"
            )

        hidden = self.input_projection(values)
        layer_ids = torch.arange(max_layers, device=values.device, dtype=torch.long)
        slot_ids = torch.arange(max_slots, device=values.device, dtype=torch.long)
        hidden = (
            hidden
            + self.layer_embedding(layer_ids).view(1, max_layers, 1, self.d_model)
            + self.slot_embedding(slot_ids).view(1, 1, max_slots, self.d_model)
        )
        slot_mask_bool = slot_mask.to(dtype=torch.bool)
        layer_mask_bool = layer_mask.to(dtype=torch.bool)
        hidden = hidden * slot_mask_bool.unsqueeze(-1).to(dtype=hidden.dtype)

        flat_hidden = hidden.reshape(batch_size * max_layers, max_slots, self.d_model)
        flat_slot_mask = slot_mask_bool.reshape(batch_size * max_layers, max_slots)
        flat_valid = flat_slot_mask.any(dim=1)
        flat_output = torch.zeros_like(flat_hidden)
        if bool(torch.any(flat_valid)):
            selected_hidden = flat_hidden[flat_valid]
            selected_mask = flat_slot_mask[flat_valid]
            if self.slot_encoder is not None:
                selected_hidden = self.slot_encoder(
                    selected_hidden,
                    src_key_padding_mask=~selected_mask,
                )
            selected_hidden = selected_hidden * selected_mask.unsqueeze(-1).to(dtype=selected_hidden.dtype)
            flat_output[flat_valid] = selected_hidden
        slot_hidden = flat_output.reshape(batch_size, max_layers, max_slots, self.d_model)
        layer_hidden = self._pool_slots(slot_hidden, slot_mask_bool)
        layer_hidden = layer_hidden * layer_mask_bool.unsqueeze(-1).to(dtype=layer_hidden.dtype)

        sample_valid = layer_mask_bool.any(dim=1)
        encoded_layers = torch.zeros_like(layer_hidden)
        if bool(torch.any(sample_valid)):
            selected_layers = layer_hidden[sample_valid]
            selected_layer_mask = layer_mask_bool[sample_valid]
            if self.layer_encoder is not None:
                selected_layers = self.layer_encoder(
                    selected_layers,
                    src_key_padding_mask=~selected_layer_mask,
                )
            selected_layers = selected_layers * selected_layer_mask.unsqueeze(-1).to(dtype=selected_layers.dtype)
            encoded_layers[sample_valid] = selected_layers

        return self._pool_layers(encoded_layers, layer_mask_bool)

    def forward(
        self,
        values: torch.Tensor,
        slot_mask: torch.Tensor,
        layer_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.extract_features(values, slot_mask, layer_mask)
        logits = self.head(embedding)
        if self.output_dim == 1:
            return logits.squeeze(1)
        return logits
