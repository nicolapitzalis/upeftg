from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...contracts import SupervisedFeatureBundle, SupervisedTaskSpec
from ..common.losses import (
    compute_balanced_class_loss_config,
    compute_balanced_rank_label_loss_config,
)
from .estimator import (
    CNN_BATCH_SIZE,
    CNN_MAX_EPOCHS,
    CNN_PATIENCE,
    CNN1DSupervisedModel,
    DataLoader,
    TensorDataset,
    _require_torch,
    nn,
    torch,
)
from ..common.prediction import selection_auroc_from_logits
from .network import _CNNLayerSequenceDANNClassifier


class CNN1DDANNSupervisedModel(CNN1DSupervisedModel):
    def __init__(
        self,
        *,
        conv_channels: int,
        num_conv_layers: int = 3,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float,
        use_residual: bool = True,
        normalization: str = "layernorm",
        pooling: str = "mean_max",
        include_total_layer_count: bool = True,
        depth_feature_mode: str = "both",
        learning_rate: float,
        weight_decay: float,
        random_state: int,
        task_spec: SupervisedTaskSpec | None = None,
        max_epochs: int = CNN_MAX_EPOCHS,
        batch_size: int = CNN_BATCH_SIZE,
        patience: int = CNN_PATIENCE,
        class_weight_loss: bool = False,
        rank_label_weight_loss: bool = False,
        normalize_input_features: bool = True,
        source_rank: int = 256,
        dann_lambda_max: float = 1.0,
        dann_lambda_gamma: float = 10.0,
    ) -> None:
        super().__init__(
            conv_channels=conv_channels,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            use_residual=use_residual,
            normalization=normalization,
            pooling=pooling,
            include_total_layer_count=include_total_layer_count,
            depth_feature_mode=depth_feature_mode,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            task_spec=task_spec,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            class_weight_loss=class_weight_loss,
            rank_label_weight_loss=rank_label_weight_loss,
            normalize_input_features=normalize_input_features,
        )
        if float(dann_lambda_max) < 0.0:
            raise ValueError("cnn_1d_dann dann_lambda_max must be non-negative")
        if float(dann_lambda_gamma) < 0.0:
            raise ValueError("cnn_1d_dann dann_lambda_gamma must be non-negative")
        self.source_rank = int(source_rank)
        self.dann_lambda_max = float(dann_lambda_max)
        self.dann_lambda_gamma = float(dann_lambda_gamma)
        self.backend_name_ = "cnn_1d_dann"
        self.domain_classes_: np.ndarray | None = None
        self.domain_class_names_: tuple[str, ...] = ()
        self.domain_rank_values_: tuple[int, ...] = ()
        self.domain_class_to_index_: dict[str, int] = {}

    def _build_model(self, *, input_channels: int) -> _CNNLayerSequenceDANNClassifier:
        _require_torch()
        if self.domain_classes_ is None or int(self.domain_classes_.shape[0]) < 2:
            raise ValueError("cnn_1d_dann requires domain classes before building the model")
        model = _CNNLayerSequenceDANNClassifier(
            input_dim=int(input_channels),
            config=self.layer_vector_config,
            output_dim=(1 if self.task_spec.is_binary else self.task_spec.n_classes),
            domain_output_dim=int(self.domain_classes_.shape[0]),
        )
        self.input_channels_ = int(input_channels)
        return model

    def _dann_lambda(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return float(self.dann_lambda_max * (2.0 / (1.0 + math.exp(-float(self.dann_lambda_gamma) * p)) - 1.0))

    def fit(
        self,
        bundle: SupervisedFeatureBundle,
        labels: np.ndarray,
        *,
        validation_data: tuple[SupervisedFeatureBundle, np.ndarray] | None = None,
        n_jobs: int | None = None,
        domain_labels: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
        rank_labels: np.ndarray | None = None,
        domain_class_names: list[str] | tuple[str, ...] | None = None,
        domain_rank_values: list[int] | tuple[int, ...] | np.ndarray | None = None,
    ) -> "CNN1DDANNSupervisedModel":
        _require_torch()
        assert torch is not None
        self._set_random_seeds()
        self._set_threads(n_jobs)

        if domain_labels is None:
            raise ValueError("cnn_1d_dann requires domain_labels for rank-adversarial training")

        train_tensors = self._prepare_numpy_inputs(bundle)
        y_raw = np.asarray(labels).reshape(-1)
        domain_np = np.asarray(domain_labels, dtype=np.int64).reshape(-1)
        n_rows = int(train_tensors.inputs.shape[0])
        if label_mask is None:
            label_mask_np = np.ones(n_rows, dtype=bool)
            label_loss_scope = "all_training_ranks"
        else:
            label_mask_np = np.asarray(label_mask, dtype=bool).reshape(-1)
            label_loss_scope = "all_training_ranks" if bool(np.all(label_mask_np)) else "masked_training_rows"
        if y_raw.shape[0] != n_rows or domain_np.shape[0] != n_rows or label_mask_np.shape[0] != n_rows:
            raise ValueError("cnn_1d_dann training features, labels, domains, and masks length mismatch")
        if not bool(np.any(label_mask_np)):
            raise ValueError("cnn_1d_dann requires at least one labeled training row")
        rank_label_loss_config: dict[str, Any] | None = None
        rank_label_sample_weights = np.ones(n_rows, dtype=np.float32)
        if self.rank_label_weight_loss:
            if rank_labels is None:
                raise ValueError("cnn_1d_dann rank_label_weight_loss requires rank_labels")
            rank_labels_np = np.asarray(rank_labels, dtype=np.int64).reshape(-1)
            if rank_labels_np.shape[0] != n_rows:
                raise ValueError("cnn_1d_dann rank_labels must have the same length as labels")
            rank_label_sample_weights, rank_label_loss_config = compute_balanced_rank_label_loss_config(
                y_raw,
                rank_labels_np,
                task_spec=self.task_spec,
                sample_mask=label_mask_np,
            )

        observed_domain_classes = np.unique(domain_np)
        if observed_domain_classes.size < 2:
            raise ValueError("cnn_1d_dann requires at least two observed rank-domain classes")
        expected_classes = np.arange(int(np.max(observed_domain_classes)) + 1, dtype=np.int64)
        if not np.array_equal(observed_domain_classes, expected_classes):
            raise ValueError("cnn_1d_dann domain_labels must be contiguous integer class ids starting at zero")
        self.domain_classes_ = expected_classes.astype(np.int64, copy=False)
        if domain_class_names is None:
            self.domain_class_names_ = tuple(f"domain_{idx}" for idx in self.domain_classes_.tolist())
        else:
            names = tuple(str(x) for x in domain_class_names)
            if len(names) != int(self.domain_classes_.shape[0]):
                raise ValueError("cnn_1d_dann domain_class_names length must match domain classes")
            self.domain_class_names_ = names
        if domain_rank_values is None:
            self.domain_rank_values_ = tuple(int(x) for x in self.domain_classes_.tolist())
        else:
            ranks = tuple(int(x) for x in np.asarray(domain_rank_values, dtype=np.int64).reshape(-1).tolist())
            if len(ranks) != int(self.domain_classes_.shape[0]):
                raise ValueError("cnn_1d_dann domain_rank_values length must match domain classes")
            self.domain_rank_values_ = ranks
        self.domain_class_to_index_ = {name: int(idx) for idx, name in enumerate(self.domain_class_names_)}

        class_loss_config: dict[str, Any] | None = None
        if self.class_weight_loss:
            class_loss_config = compute_balanced_class_loss_config(
                y_raw,
                task_spec=self.task_spec,
                sample_mask=label_mask_np,
            )

        if self.task_spec.is_binary:
            y_train = np.asarray(y_raw, dtype=np.float32).reshape(-1)
            pos_weight = None
            if class_loss_config is not None:
                pos_weight = torch.tensor(
                    float(class_loss_config["binary_pos_weight"]),
                    dtype=torch.float32,
                )
            label_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        else:
            y_train = np.asarray(y_raw, dtype=np.int64).reshape(-1)
            class_weight_tensor = None
            if class_loss_config is not None:
                class_weight_tensor = torch.tensor(
                    np.asarray(class_loss_config["class_weights"], dtype=np.float32),
                    dtype=torch.float32,
                )
            label_loss_fn = nn.CrossEntropyLoss(
                weight=class_weight_tensor,
                reduction=("none" if self.rank_label_weight_loss else "mean"),
            )
        domain_loss_fn = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(
            torch.from_numpy(train_tensors.inputs),
            torch.from_numpy(train_tensors.layer_mask),
            torch.from_numpy(y_train),
            torch.from_numpy(domain_np),
            torch.from_numpy(label_mask_np.astype(np.float32)),
            torch.from_numpy(rank_label_sample_weights),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(1, len(train_dataset))),
            shuffle=True,
        )

        valid_loader = None
        y_valid = None
        if validation_data is not None:
            valid_bundle, valid_labels = validation_data
            valid_tensors = self._prepare_numpy_inputs(valid_bundle)
            y_valid = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
            valid_dataset = TensorDataset(
                torch.from_numpy(valid_tensors.inputs),
                torch.from_numpy(valid_tensors.layer_mask),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=min(self.batch_size, max(1, len(valid_dataset))),
                shuffle=False,
            )

        self.model_ = self._build_model(input_channels=int(train_tensors.inputs.shape[2]))
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        best_state = None
        best_metric = -math.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, float]] = []
        total_steps = max(1, int(self.max_epochs) * max(1, len(train_loader)))
        global_step = 0

        for epoch_idx in range(self.max_epochs):
            self.model_.train()
            train_loss_sum = 0.0
            label_loss_sum = 0.0
            domain_loss_sum = 0.0
            train_count = 0
            label_count = 0.0
            last_lambda = 0.0
            last_lr = float(self.learning_rate)

            for (
                batch_inputs,
                batch_layer_mask,
                batch_labels,
                batch_domains,
                batch_label_mask,
                batch_label_weights,
            ) in train_loader:
                progress = float(global_step) / float(max(1, total_steps - 1))
                lambda_value = self._dann_lambda(progress)
                current_lr = float(self.learning_rate)

                optimizer.zero_grad()
                label_logits, domain_logits = self.model_(
                    batch_inputs,
                    batch_layer_mask,
                    lambda_value=float(lambda_value),
                    return_domain=True,
                )
                domain_loss = domain_loss_fn(domain_logits, batch_domains)

                supervised_mask = batch_label_mask > 0.5
                if bool(torch.any(supervised_mask)):
                    if self.task_spec.is_binary:
                        label_loss_raw = label_loss_fn(
                            label_logits[supervised_mask],
                            batch_labels[supervised_mask],
                        )
                    else:
                        label_loss_raw = label_loss_fn(
                            label_logits[supervised_mask],
                            batch_labels[supervised_mask].to(dtype=torch.long),
                        )
                    current_label_count = float(torch.sum(supervised_mask).item())
                    if self.rank_label_weight_loss:
                        selected_weights = batch_label_weights[supervised_mask].to(dtype=label_loss_raw.dtype)
                        label_loss_numerator = torch.sum(label_loss_raw.reshape(-1) * selected_weights.reshape(-1))
                        label_loss_denominator = torch.clamp(torch.sum(selected_weights), min=1.0)
                        label_loss = label_loss_numerator / label_loss_denominator
                        current_label_count = float(label_loss_denominator.item())
                    else:
                        label_loss = label_loss_raw
                else:
                    label_loss = domain_loss.new_zeros(())
                    current_label_count = 0.0

                loss = label_loss + domain_loss
                loss.backward()
                optimizer.step()

                batch_size = int(batch_domains.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                domain_loss_sum += float(domain_loss.item()) * batch_size
                if current_label_count > 0:
                    if self.rank_label_weight_loss:
                        label_loss_sum += float(label_loss_numerator.item())
                    else:
                        label_loss_sum += float(label_loss.item()) * current_label_count
                    label_count += current_label_count
                train_count += batch_size
                last_lambda = float(lambda_value)
                last_lr = float(current_lr)
                global_step += 1

            train_loss = train_loss_sum / max(1, train_count)
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
                    "label_loss": float(label_loss_sum / max(1, label_count)),
                    "domain_loss": float(domain_loss_sum / max(1, train_count)),
                    "label_rows": float(label_count),
                    "domain_rows": float(train_count),
                    "dann_lambda": float(last_lambda),
                    "learning_rate": float(last_lr),
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

            if valid_loader is not None and stale_epochs >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("cnn_1d_dann training did not produce a valid checkpoint")
        self.model_.load_state_dict(best_state)
        self._fit_summary = {
            "best_epoch": int(best_epoch),
            "selection_metric": float(best_metric),
            "selection_metric_name": str(self.task_spec.selection_metric_name),
            "epochs_ran": int(len(history)),
            "history": history,
            "domain_loss_weight": 1.0,
            "label_loss_scope": str(label_loss_scope),
            "class_weight_loss": bool(self.class_weight_loss),
            "class_loss_weights": class_loss_config,
            "rank_label_weight_loss": bool(self.rank_label_weight_loss),
            "rank_label_loss_weights": rank_label_loss_config,
        }
        return self

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {
            "domain_adaptation": {
                "source_rank": int(self.source_rank),
                "domain_class_names": list(self.domain_class_names_),
                "domain_rank_values": [int(x) for x in self.domain_rank_values_],
                "domain_class_to_index": dict(self.domain_class_to_index_),
                "label_loss_scope": str(self._fit_summary.get("label_loss_scope", "all_training_ranks")),
                "domain_loss": "multiclass_rank_cross_entropy",
                "domain_loss_weight": 1.0,
                "lambda_schedule": {
                    "type": "dann_paper_logistic",
                    "lambda_max": float(self.dann_lambda_max),
                    "gamma": float(self.dann_lambda_gamma),
                },
                "learning_rate_schedule": {
                    "type": "fixed",
                    "learning_rate": float(self.learning_rate),
                },
            }
        }
