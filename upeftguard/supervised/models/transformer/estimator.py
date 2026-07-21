from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..common.losses import (
    compute_balanced_class_loss_config,
    compute_balanced_rank_label_loss_config,
)
from ..common.prediction import TorchSequencePredictionMixin, selection_auroc_from_logits
from ...contracts import (
    SupervisedFeatureBundle,
    SupervisedTaskSpec,
)
from ...tasks import (
    default_binary_task_spec as _default_binary_task_spec,
)
from ..common.torch_runtime import (
    load_torch_checkpoint_payload,
    set_torch_random_seeds,
    set_torch_threads,
)
from .data import (
    TransformerFeatureTensors,
    TransformerLayerSequenceLayout,
    TransformerNormalizationStats,
    layout_from_payload as _layout_from_payload,
    prepare_transformer_layer_sequence,
)
from .network import (
    SUPPORTED_TRANSFORMER_POOLING,
    TRANSFORMER_POOLING_MEAN,
    _TransformerHierarchyClassifier,
)

try:  # pragma: no cover - exercised in environments that install torch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - soft dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


TRANSFORMER_MAX_EPOCHS = 50
TRANSFORMER_BATCH_SIZE = 8
TRANSFORMER_PATIENCE = 5
_torch_load_checkpoint = load_torch_checkpoint_payload


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ModuleNotFoundError("transformer requires the optional 'torch' dependency, but torch is not installed")


class TransformerSupervisedModel(TorchSequencePredictionMixin):
    def __init__(
        self,
        *,
        d_model: int = 18,
        nhead: int = 2,
        slot_num_layers: int = 1,
        layer_num_layers: int = 2,
        dim_feedforward: int = 44,
        dropout: float = 0.1,
        norm_first: bool = True,
        pooling: str = TRANSFORMER_POOLING_MEAN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = TRANSFORMER_MAX_EPOCHS,
        batch_size: int = TRANSFORMER_BATCH_SIZE,
        patience: int = TRANSFORMER_PATIENCE,
        class_weight_loss: bool = False,
        rank_label_weight_loss: bool = False,
        normalize_input_features: bool = True,
        include_value_mask_channels: bool = True,
    ) -> None:
        _require_torch()
        if int(d_model) <= 0:
            raise ValueError("transformer d_model must be positive")
        if int(nhead) <= 0:
            raise ValueError("transformer nhead must be positive")
        if int(d_model) % int(nhead) != 0:
            raise ValueError("transformer d_model must be divisible by nhead")
        if int(slot_num_layers) < 0 or int(layer_num_layers) < 0:
            raise ValueError("transformer encoder layer counts must be non-negative")
        if int(slot_num_layers) + int(layer_num_layers) <= 0:
            raise ValueError("transformer requires at least one encoder layer")
        if int(dim_feedforward) <= 0:
            raise ValueError("transformer dim_feedforward must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("transformer dropout must be in [0, 1)")
        if str(pooling) not in SUPPORTED_TRANSFORMER_POOLING:
            raise ValueError(
                f"transformer pooling must be one of {sorted(SUPPORTED_TRANSFORMER_POOLING)}, got {pooling!r}"
            )
        if bool(class_weight_loss) and bool(rank_label_weight_loss):
            raise ValueError("transformer supports either class_weight_loss or rank_label_weight_loss, not both")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.slot_num_layers = int(slot_num_layers)
        self.layer_num_layers = int(layer_num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.norm_first = bool(norm_first)
        self.pooling = str(pooling)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.task_spec = task_spec if task_spec is not None else _default_binary_task_spec()
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.class_weight_loss = bool(class_weight_loss)
        self.rank_label_weight_loss = bool(rank_label_weight_loss)
        self.normalize_input_features = bool(normalize_input_features)
        self.include_value_mask_channels = bool(include_value_mask_channels)
        self.model_: _TransformerHierarchyClassifier | None = None
        self.normalization_stats_: TransformerNormalizationStats | None = None
        self.layout_: TransformerLayerSequenceLayout | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.classes_ = np.arange(self.task_spec.n_classes, dtype=np.int32)
        self.class_names_ = tuple(str(x) for x in self.task_spec.class_names)
        self.backend_name_ = "transformer"
        self._fit_summary: dict[str, Any] = {}
        self.checkpoint_dir_: Path | None = None
        self.checkpoint_interval_seconds_: float | None = None
        self.resume_checkpoint_path_: Path | None = None
        self._fit_checkpoint_artifacts: dict[str, Any] = {}

    def configure_training_checkpoints(
        self,
        *,
        checkpoint_dir: Path,
        interval_seconds: float,
        resume_checkpoint: Path | None = None,
    ) -> None:
        resolved_interval = float(interval_seconds)
        if resolved_interval <= 0.0:
            raise ValueError("transformer checkpoint interval must be positive")
        resolved_dir = Path(checkpoint_dir).expanduser().resolve()
        self.checkpoint_dir_ = resolved_dir
        self.checkpoint_interval_seconds_ = resolved_interval
        self.resume_checkpoint_path_ = (
            Path(resume_checkpoint).expanduser().resolve()
            if resume_checkpoint is not None
            else resolved_dir / "resume_latest.pt"
        )

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: Path, *, overwrite: bool) -> None:
        assert torch is not None
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and resolved.exists():
            return
        temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, temporary)
            if not overwrite and resolved.exists():
                temporary.unlink(missing_ok=True)
                return
            os.replace(temporary, resolved)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temporary, resolved)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        assert torch is not None
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    @staticmethod
    def _restore_rng_state(payload: Any) -> None:
        assert torch is not None
        if not isinstance(payload, dict):
            return
        if payload.get("python") is not None:
            random.setstate(payload["python"])
        if payload.get("numpy") is not None:
            np.random.set_state(payload["numpy"])
        if payload.get("torch") is not None:
            torch.set_rng_state(payload["torch"])

    def _set_random_seeds(self) -> None:
        set_torch_random_seeds(self.random_state)

    def _set_threads(self, n_jobs: int | None) -> None:
        set_torch_threads(n_jobs)

    def _prepare_numpy_inputs(self, bundle: SupervisedFeatureBundle) -> TransformerFeatureTensors:
        tensors = prepare_transformer_layer_sequence(
            bundle,
            normalization_stats=self.normalization_stats_,
            normalize_values=bool(self.normalize_input_features),
            include_value_mask_channels=bool(self.include_value_mask_channels),
        )
        if self.layout_ is None:
            self.layout_ = tensors.layout
        elif self.layout_ != tensors.layout:
            raise ValueError("transformer encountered an incompatible layer-sequence layout between bundles")

        if self.normalization_stats_ is None:
            self.normalization_stats_ = tensors.normalization_stats
            self.feature_mean_ = np.asarray(tensors.normalization_stats.feature_mean, dtype=np.float32)
            self.feature_std_ = np.asarray(tensors.normalization_stats.feature_std, dtype=np.float32)
        elif self.feature_mean_ is None or self.feature_std_ is None:
            self.feature_mean_ = np.asarray(self.normalization_stats_.feature_mean, dtype=np.float32)
            self.feature_std_ = np.asarray(self.normalization_stats_.feature_std, dtype=np.float32)
        return tensors

    def _build_model(self, *, layout: TransformerLayerSequenceLayout) -> _TransformerHierarchyClassifier:
        return _TransformerHierarchyClassifier(
            input_dim=int(layout.input_dim),
            d_model=int(self.d_model),
            nhead=int(self.nhead),
            slot_num_layers=int(self.slot_num_layers),
            layer_num_layers=int(self.layer_num_layers),
            dim_feedforward=int(self.dim_feedforward),
            dropout=float(self.dropout),
            norm_first=bool(self.norm_first),
            pooling=str(self.pooling),
            max_layers=int(layout.max_layers),
            max_slots=int(layout.max_slots),
            output_dim=(1 if self.task_spec.is_binary else self.task_spec.n_classes),
        )

    def _loader_from_tensors(
        self,
        tensors: TransformerFeatureTensors,
        *,
        labels: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        assert torch is not None and TensorDataset is not None and DataLoader is not None
        tensor_args: list[torch.Tensor] = [
            torch.from_numpy(tensors.values),
            torch.from_numpy(tensors.slot_mask),
            torch.from_numpy(tensors.layer_mask),
        ]
        if labels is not None:
            tensor_args.append(torch.from_numpy(labels))
        if sample_weights is not None:
            tensor_args.append(torch.from_numpy(sample_weights))
        dataset = TensorDataset(*tensor_args)
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=bool(shuffle),
        )

    def _logits_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("transformer model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_values, batch_slot_mask, batch_layer_mask in loader:
                logits = self.model_(batch_values, batch_slot_mask, batch_layer_mask)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        if not outputs:
            if self.task_spec.is_binary:
                return np.asarray([], dtype=np.float64)
            return np.asarray([], dtype=np.float64).reshape(0, self.task_spec.n_classes)
        combined = np.concatenate(outputs, axis=0)
        if self.task_spec.is_binary:
            return combined.reshape(-1)
        return np.asarray(combined, dtype=np.float64)

    def _features_from_loader(self, loader: DataLoader) -> np.ndarray:
        assert torch is not None
        if self.model_ is None:
            raise RuntimeError("transformer model has not been fit")
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_values, batch_slot_mask, batch_layer_mask in loader:
                embeddings = self.model_.extract_features(
                    batch_values,
                    batch_slot_mask,
                    batch_layer_mask,
                )
                outputs.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
        if not outputs:
            embedding_dim = int(self.model_.embedding_dim)
            return np.asarray([], dtype=np.float32).reshape(0, embedding_dim)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def _checkpoint_payload(
        self,
        *,
        state_dict: dict[str, Any] | None = None,
        fit_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.model_ is None or self.layout_ is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("transformer model has not been initialized for checkpointing")
        return {
            "backend": self.backend_name_,
            "config": {
                "d_model": int(self.d_model),
                "nhead": int(self.nhead),
                "slot_num_layers": int(self.slot_num_layers),
                "layer_num_layers": int(self.layer_num_layers),
                "dim_feedforward": int(self.dim_feedforward),
                "dropout": float(self.dropout),
                "norm_first": bool(self.norm_first),
                "pooling": str(self.pooling),
                "learning_rate": float(self.learning_rate),
                "weight_decay": float(self.weight_decay),
                "random_state": int(self.random_state),
                "max_epochs": int(self.max_epochs),
                "batch_size": int(self.batch_size),
                "patience": int(self.patience),
                "class_weight_loss": bool(self.class_weight_loss),
                "rank_label_weight_loss": bool(self.rank_label_weight_loss),
                "normalize_input_features": bool(self.normalize_input_features),
                "include_value_mask_channels": bool(self.include_value_mask_channels),
                "task_mode": str(self.task_spec.task_mode),
                "num_classes": int(self.task_spec.n_classes),
            },
            "layout": asdict(self.layout_),
            "state_dict": state_dict if state_dict is not None else self.model_.state_dict(),
            "normalization": {
                "feature_mean": np.asarray(self.feature_mean_, dtype=np.float32),
                "feature_std": np.asarray(self.feature_std_, dtype=np.float32),
            },
            "classes": np.asarray(self.classes_, dtype=np.int32),
            "class_names": list(self.class_names_),
            "task": self.task_spec.to_dict(),
            "fit_summary": dict(self._fit_summary if fit_summary is None else fit_summary),
        }

    def _write_checkpoint_index(self, hourly_checkpoints: list[dict[str, Any]]) -> Path | None:
        if self.checkpoint_dir_ is None:
            return None
        index_path = self.checkpoint_dir_ / "checkpoint_index.json"
        self._atomic_json_write(
            {
                "backend": self.backend_name_,
                "checkpoint_interval_seconds": self.checkpoint_interval_seconds_,
                "hourly_checkpoints": hourly_checkpoints,
                "resume_checkpoint": (
                    str(self.resume_checkpoint_path_) if self.resume_checkpoint_path_ is not None else None
                ),
                "final_checkpoint": str(self.checkpoint_dir_ / "final.pt"),
            },
            index_path,
        )
        return index_path

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        rank_labels: np.ndarray | None = None,
    ) -> "TransformerSupervisedModel":
        _require_torch()
        assert torch is not None and nn is not None
        self._set_random_seeds()
        self._set_threads(n_jobs)

        train_tensors = self._prepare_numpy_inputs(bundle)
        if self.layout_ is None:
            raise RuntimeError("transformer failed to resolve a layer-sequence layout")
        if self.task_spec.is_binary:
            y_train = np.asarray(labels, dtype=np.float32).reshape(-1)
        else:
            y_train = np.asarray(labels, dtype=np.int64).reshape(-1)
        if train_tensors.values.shape[0] != y_train.shape[0]:
            raise ValueError("transformer training features/labels length mismatch")

        rank_label_loss_config: dict[str, Any] | None = None
        rank_label_sample_weights = np.ones(y_train.shape[0], dtype=np.float32)
        if self.rank_label_weight_loss:
            if rank_labels is None:
                raise ValueError("transformer rank_label_weight_loss requires rank_labels")
            rank_labels_np = np.asarray(rank_labels, dtype=np.int64).reshape(-1)
            if rank_labels_np.shape[0] != y_train.shape[0]:
                raise ValueError("transformer rank_labels must have the same length as labels")
            rank_label_sample_weights, rank_label_loss_config = compute_balanced_rank_label_loss_config(
                y_train,
                rank_labels_np,
                task_spec=self.task_spec,
            )

        train_loader = self._loader_from_tensors(
            train_tensors,
            labels=y_train,
            sample_weights=rank_label_sample_weights,
            shuffle=True,
        )

        valid_loader = None
        y_valid = None
        if validation_data is not None:
            valid_bundle, valid_labels = validation_data
            valid_tensors = self._prepare_numpy_inputs(valid_bundle)
            y_valid = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
            valid_loader = self._loader_from_tensors(valid_tensors, shuffle=False)

        self.model_ = self._build_model(layout=self.layout_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        class_loss_config: dict[str, Any] | None = None
        if self.class_weight_loss:
            class_loss_config = compute_balanced_class_loss_config(
                y_train,
                task_spec=self.task_spec,
            )

        if self.task_spec.is_binary:
            pos_weight = None
            if class_loss_config is not None:
                pos_weight = torch.tensor(
                    float(class_loss_config["binary_pos_weight"]),
                    dtype=torch.float32,
                )
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        else:
            class_weight_tensor = None
            if class_loss_config is not None:
                class_weight_tensor = torch.tensor(
                    np.asarray(class_loss_config["class_weights"], dtype=np.float32),
                    dtype=torch.float32,
                )
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weight_tensor,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )

        checkpointing_enabled = bool(self.checkpoint_dir_ is not None and self.checkpoint_interval_seconds_ is not None)
        if checkpointing_enabled:
            assert self.checkpoint_dir_ is not None
            self.checkpoint_dir_.mkdir(parents=True, exist_ok=True)

        session_started_perf = perf_counter()
        training_started_at_utc = datetime.now(timezone.utc).isoformat()
        elapsed_offset_seconds = 0.0
        resumed_from: str | None = None
        start_epoch = 0
        best_state = None
        best_metric = -math.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, Any]] = []
        hourly_checkpoints: list[dict[str, Any]] = []

        if checkpointing_enabled and self.checkpoint_dir_ is not None:
            index_path = self.checkpoint_dir_ / "checkpoint_index.json"
            if index_path.exists():
                try:
                    existing_index = json.loads(index_path.read_text(encoding="utf-8"))
                    existing_rows = existing_index.get("hourly_checkpoints", [])
                    if isinstance(existing_rows, list):
                        hourly_checkpoints = [dict(row) for row in existing_rows if isinstance(row, dict)]
                except (OSError, ValueError, TypeError):
                    hourly_checkpoints = []

        if checkpointing_enabled and self.resume_checkpoint_path_ is not None and self.resume_checkpoint_path_.exists():
            resume_payload = _torch_load_checkpoint(self.resume_checkpoint_path_)
            if str(resume_payload.get("backend") or "") != self.backend_name_:
                raise ValueError(f"Resume checkpoint backend mismatch: {self.resume_checkpoint_path_}")
            resume_layout = _layout_from_payload(resume_payload.get("layout"))
            if resume_layout != self.layout_:
                raise ValueError("Resume checkpoint layout does not match the current training feature layout")
            resume_state_dict = resume_payload.get("state_dict")
            training_state = resume_payload.get("training_state")
            if not isinstance(resume_state_dict, dict) or not isinstance(training_state, dict):
                raise ValueError(f"Invalid transformer training resume checkpoint: {self.resume_checkpoint_path_}")
            self.model_.load_state_dict(resume_state_dict)
            optimizer_state = training_state.get("optimizer_state_dict")
            if not isinstance(optimizer_state, dict):
                raise ValueError("Transformer resume checkpoint is missing optimizer state")
            optimizer.load_state_dict(optimizer_state)
            start_epoch = int(training_state.get("next_epoch", 0))
            elapsed_offset_seconds = float(training_state.get("elapsed_seconds", 0.0))
            training_started_at_utc = str(training_state.get("training_started_at_utc") or training_started_at_utc)
            best_state_value = training_state.get("best_state_dict")
            best_state = best_state_value if isinstance(best_state_value, dict) else None
            best_metric = float(training_state.get("best_metric", -math.inf))
            best_epoch = int(training_state.get("best_epoch", -1))
            stale_epochs = int(training_state.get("stale_epochs", 0))
            history_value = training_state.get("history", [])
            history = [dict(row) for row in history_value if isinstance(row, dict)]
            self._restore_rng_state(training_state.get("rng_state"))
            resumed_from = str(self.resume_checkpoint_path_)

        interval_seconds = float(self.checkpoint_interval_seconds_ or 0.0)
        next_hour_index = max(1, int(elapsed_offset_seconds // max(interval_seconds, 1.0)) + 1)

        def elapsed_seconds() -> float:
            return float(elapsed_offset_seconds + (perf_counter() - session_started_perf))

        def partial_fit_summary(*, current_epoch: int | None = None) -> dict[str, Any]:
            return {
                "best_epoch": int(best_epoch),
                "selection_metric": (float(best_metric) if math.isfinite(best_metric) else None),
                "selection_metric_name": str(self.task_spec.selection_metric_name),
                "epochs_ran": int(len(history)),
                "current_epoch": current_epoch,
                "history": list(history),
                "class_weight_loss": bool(self.class_weight_loss),
                "class_loss_weights": class_loss_config,
                "rank_label_weight_loss": bool(self.rank_label_weight_loss),
                "rank_label_loss_weights": rank_label_loss_config,
                "training_started_at_utc": training_started_at_utc,
                "training_elapsed_seconds": elapsed_seconds(),
                "resumed_from": resumed_from,
            }

        def write_resume_checkpoint(*, next_epoch: int, completed: bool) -> None:
            if not checkpointing_enabled or self.resume_checkpoint_path_ is None:
                return
            payload = self._checkpoint_payload(
                fit_summary=partial_fit_summary(current_epoch=max(-1, next_epoch - 1)),
            )
            payload["checkpoint_kind"] = "training_resume"
            payload["training_state"] = {
                "next_epoch": int(next_epoch),
                "elapsed_seconds": elapsed_seconds(),
                "training_started_at_utc": training_started_at_utc,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "best_state_dict": best_state,
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "stale_epochs": int(stale_epochs),
                "history": list(history),
                "rng_state": self._capture_rng_state(),
                "completed": bool(completed),
            }
            self._atomic_torch_save(payload, self.resume_checkpoint_path_, overwrite=True)

        for epoch_idx in range(start_epoch, self.max_epochs):
            self.model_.train()
            train_loss_sum = 0.0
            train_count = 0.0
            for (
                batch_values,
                batch_slot_mask,
                batch_layer_mask,
                batch_labels,
                batch_sample_weights,
            ) in train_loader:
                optimizer.zero_grad()
                logits = self.model_(batch_values, batch_slot_mask, batch_layer_mask)
                loss_raw = loss_fn(logits, batch_labels)
                if self.rank_label_weight_loss:
                    weights = batch_sample_weights.to(dtype=loss_raw.dtype)
                    loss_numerator = torch.sum(loss_raw.reshape(-1) * weights.reshape(-1))
                    loss_denominator = torch.clamp(torch.sum(weights), min=1.0)
                    loss = loss_numerator / loss_denominator
                else:
                    loss = loss_raw
                loss.backward()
                optimizer.step()
                batch_size = int(batch_labels.shape[0])
                if self.rank_label_weight_loss:
                    train_loss_sum += float(loss_numerator.item())
                    train_count += float(loss_denominator.item())
                else:
                    train_loss_sum += float(loss.item()) * batch_size
                    train_count += float(batch_size)

                if checkpointing_enabled and self.checkpoint_dir_ is not None:
                    current_elapsed = elapsed_seconds()
                    while current_elapsed >= float(next_hour_index) * interval_seconds:
                        checkpoint_path = self.checkpoint_dir_ / f"hour_{next_hour_index:03d}.pt"
                        checkpoint_payload = self._checkpoint_payload(
                            fit_summary=partial_fit_summary(current_epoch=int(epoch_idx)),
                        )
                        checkpoint_payload["checkpoint_kind"] = "hourly_inference"
                        checkpoint_payload["checkpoint_hour"] = int(next_hour_index)
                        checkpoint_payload["training_elapsed_seconds"] = float(current_elapsed)
                        self._atomic_torch_save(
                            checkpoint_payload,
                            checkpoint_path,
                            overwrite=False,
                        )
                        if not any(int(row.get("hour", -1)) == int(next_hour_index) for row in hourly_checkpoints):
                            hourly_checkpoints.append(
                                {
                                    "hour": int(next_hour_index),
                                    "path": str(checkpoint_path),
                                    "training_elapsed_seconds": float(current_elapsed),
                                    "epoch": int(epoch_idx),
                                }
                            )
                        next_hour_index += 1
                        self._write_checkpoint_index(hourly_checkpoints)

            train_loss = train_loss_sum / max(1.0, train_count)
            metric = -train_loss
            if valid_loader is not None and y_valid is not None:
                valid_logits = self._logits_from_loader(valid_loader)
                metric = selection_auroc_from_logits(
                    y_valid,
                    valid_logits,
                    task_spec=self.task_spec,
                )

            history.append(
                {
                    "epoch": float(epoch_idx),
                    "train_loss": float(train_loss),
                    "selection_metric": float(metric),
                    "selection_metric_name": str(self.task_spec.selection_metric_name),
                }
            )

            if metric > best_metric:
                best_metric = float(metric)
                best_epoch = int(epoch_idx)
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1

            write_resume_checkpoint(next_epoch=int(epoch_idx + 1), completed=False)

            if valid_loader is not None and stale_epochs >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("transformer training did not produce a valid checkpoint")
        write_resume_checkpoint(next_epoch=int(len(history)), completed=True)
        self.model_.load_state_dict(best_state)
        training_ended_at_utc = datetime.now(timezone.utc).isoformat()
        total_elapsed_seconds = elapsed_seconds()
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": float(best_metric),
            "selection_metric_name": str(self.task_spec.selection_metric_name),
            "epochs_ran": int(len(history)),
            "history": history,
            "class_weight_loss": bool(self.class_weight_loss),
            "class_loss_weights": class_loss_config,
            "rank_label_weight_loss": bool(self.rank_label_weight_loss),
            "rank_label_loss_weights": rank_label_loss_config,
            "training_started_at_utc": training_started_at_utc,
            "training_ended_at_utc": training_ended_at_utc,
            "training_elapsed_seconds": float(total_elapsed_seconds),
            "training_elapsed_seconds_this_process": float(perf_counter() - session_started_perf),
            "resumed_from": resumed_from,
        }
        if checkpointing_enabled and self.checkpoint_dir_ is not None:
            final_checkpoint = self.checkpoint_dir_ / "final.pt"
            final_payload = self._checkpoint_payload()
            final_payload["checkpoint_kind"] = "final_inference"
            final_payload["training_elapsed_seconds"] = float(total_elapsed_seconds)
            self._atomic_torch_save(final_payload, final_checkpoint, overwrite=True)
            checkpoint_index = self._write_checkpoint_index(hourly_checkpoints)
            self._fit_checkpoint_artifacts = {
                "checkpoint_dir": str(self.checkpoint_dir_),
                "checkpoint_index": str(checkpoint_index) if checkpoint_index is not None else None,
                "hourly_checkpoints": list(hourly_checkpoints),
                "resume_checkpoint": (
                    str(self.resume_checkpoint_path_) if self.resume_checkpoint_path_ is not None else None
                ),
                "final_checkpoint": str(final_checkpoint),
            }
        return self

    def decision_function(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        if self.model_ is None:
            raise RuntimeError("transformer model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        loader = self._loader_from_tensors(tensors, shuffle=False)
        return self._logits_from_loader(loader)

    def extract_features(self, bundle: SupervisedFeatureBundle) -> np.ndarray:
        _require_torch()
        if self.model_ is None:
            raise RuntimeError("transformer model has not been fit")
        tensors = self._prepare_numpy_inputs(bundle)
        loader = self._loader_from_tensors(tensors, shuffle=False)
        return self._features_from_loader(loader)

    def save(self, path: Path) -> None:
        _require_torch()
        assert torch is not None
        self._atomic_torch_save(
            self._checkpoint_payload(),
            Path(path).expanduser().resolve(),
            overwrite=True,
        )
