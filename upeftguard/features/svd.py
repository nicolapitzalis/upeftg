from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .adapters import extract_layers_from_keys
from .delta import check_consistency, discover_delta_pairs
from ..utilities.manifest import ManifestItem


SVD_EXTRACTOR_VERSION = "2.0.0"


@dataclass(frozen=True)
class StreamingSVDBasis:
    z_train_max: np.ndarray
    singular_values: np.ndarray
    u: np.ndarray
    expected_pairs: tuple[tuple[str, str], ...]
    expected_a_shapes: tuple[tuple[int, ...], ...]
    expected_b_shapes: tuple[tuple[int, ...], ...]
    layers: list[int]
    n_features: int
    resolved_component_grid: list[int]
    max_rank: int
    gram_centered: np.ndarray
    gram_stream_time_seconds: float
    train_row_mean_raw: np.ndarray
    train_grand_mean_raw: float


def sanitize_component_grid(requested: list[int], n_samples: int, n_features: int) -> tuple[list[int], list[str], int]:
    max_rank = max(1, min(n_samples - 1, n_features))
    warnings: list[str] = []
    sanitized: list[int] = []

    for k in requested:
        if k <= 0:
            warnings.append(f"Ignoring non-positive component request: {k}")
            continue
        clipped = min(k, max_rank)
        if clipped != k:
            warnings.append(f"Clipped n_components from {k} to {clipped} (rank limit={max_rank})")
        sanitized.append(clipped)

    sanitized = sorted(set(sanitized))
    if not sanitized:
        sanitized = [max_rank]
        warnings.append(f"No valid n_components requested; using fallback [{max_rank}]")

    return sanitized, warnings, max_rank


def _delta_feature_dim(
    *,
    a_shapes: tuple[tuple[int, ...], ...],
    b_shapes: tuple[tuple[int, ...], ...],
) -> int:
    if len(a_shapes) != len(b_shapes):
        raise ValueError(
            f"Mismatched A/B block counts for delta schema: len(A)={len(a_shapes)}, len(B)={len(b_shapes)}"
        )

    total = 0
    for i, (a_shape, b_shape) in enumerate(zip(a_shapes, b_shapes)):
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(
                f"Expected rank-2 LoRA tensors for block {i}, got A{a_shape}, B{b_shape}"
            )
        if int(a_shape[0]) != int(b_shape[1]):
            raise ValueError(
                f"LoRA rank mismatch for block {i}: A{a_shape}, B{b_shape}"
            )
        out_dim = int(b_shape[0])
        in_dim = int(a_shape[1])
        total += out_dim * in_dim
    return int(total)


def _discover_delta_schema(
    adapter_path: Path,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], list[int], int]:
    pairs, _, a_shapes, b_shapes = discover_delta_pairs(adapter_path)
    expected_pairs = tuple((str(a_key), str(b_key)) for a_key, b_key in pairs)
    expected_a_shapes = tuple(tuple(int(x) for x in s) for s in a_shapes)
    expected_b_shapes = tuple(tuple(int(x) for x in s) for s in b_shapes)

    layer_keys = tuple(key for pair in expected_pairs for key in pair)
    layers = extract_layers_from_keys(layer_keys)
    n_features = _delta_feature_dim(a_shapes=expected_a_shapes, b_shapes=expected_b_shapes)
    return expected_pairs, expected_a_shapes, expected_b_shapes, layers, n_features


def _load_block_factors(
    readers: list[Any],
    *,
    a_key: str,
    b_key: str,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    first_a = np.asarray(readers[0].get_tensor(a_key), dtype=dtype)
    first_b = np.asarray(readers[0].get_tensor(b_key), dtype=dtype)

    a_stack = np.empty((len(readers), *first_a.shape), dtype=dtype)
    b_stack = np.empty((len(readers), *first_b.shape), dtype=dtype)
    a_stack[0] = first_a
    b_stack[0] = first_b

    for i in range(1, len(readers)):
        a_stack[i] = np.asarray(readers[i].get_tensor(a_key), dtype=dtype)
        b_stack[i] = np.asarray(readers[i].get_tensor(b_key), dtype=dtype)

    return a_stack, b_stack


def _delta_factor_cross_inner_products(
    *,
    left_a: np.ndarray,
    left_b: np.ndarray,
    right_a: np.ndarray,
    right_b: np.ndarray,
) -> np.ndarray:
    n_left = int(left_a.shape[0])
    n_right = int(right_a.shape[0])
    out = np.empty((n_left, n_right), dtype=np.float64)

    # vec(B@A) inner products can be computed from rank-space products only.
    for i in range(n_left):
        b_cross = np.tensordot(right_b, left_b[i], axes=([1], [0]))
        a_cross = np.tensordot(right_a, left_a[i], axes=([2], [1]))
        out[i, :] = np.einsum("jlk,jlk->j", b_cross, a_cross, dtype=np.float64)

    return out


def compute_train_gram_from_delta_factors(
    adapter_paths: list[Path],
    *,
    expected_pairs: tuple[tuple[str, str], ...],
    dtype: np.dtype,
) -> np.ndarray:
    n_samples = len(adapter_paths)
    gram_raw = np.zeros((n_samples, n_samples), dtype=np.float64)
    if n_samples == 0:
        return gram_raw

    with ExitStack() as stack:
        readers = [stack.enter_context(safe_open(path, framework="numpy")) for path in adapter_paths]

        for a_key, b_key in expected_pairs:
            a_stack, b_stack = _load_block_factors(
                readers=readers,
                a_key=a_key,
                b_key=b_key,
                dtype=dtype,
            )
            gram_raw += _delta_factor_cross_inner_products(
                left_a=a_stack,
                left_b=b_stack,
                right_a=a_stack,
                right_b=b_stack,
            )

    return 0.5 * (gram_raw + gram_raw.T)


def compute_cross_gram_from_delta_factors(
    *,
    train_paths: list[Path],
    target_paths: list[Path],
    expected_pairs: tuple[tuple[str, str], ...],
    dtype: np.dtype,
) -> np.ndarray:
    n_train = len(train_paths)
    n_target = len(target_paths)
    cross_raw = np.zeros((n_target, n_train), dtype=np.float64)
    if n_train == 0 or n_target == 0:
        return cross_raw

    with ExitStack() as stack:
        train_readers = [stack.enter_context(safe_open(path, framework="numpy")) for path in train_paths]
        target_readers = [stack.enter_context(safe_open(path, framework="numpy")) for path in target_paths]

        for a_key, b_key in expected_pairs:
            train_a, train_b = _load_block_factors(
                readers=train_readers,
                a_key=a_key,
                b_key=b_key,
                dtype=dtype,
            )
            target_a, target_b = _load_block_factors(
                readers=target_readers,
                a_key=a_key,
                b_key=b_key,
                dtype=dtype,
            )
            cross_raw += _delta_factor_cross_inner_products(
                left_a=target_a,
                left_b=target_b,
                right_a=train_a,
                right_b=train_b,
            )

    return cross_raw


def center_sample_gram(gram_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if gram_raw.ndim != 2 or gram_raw.shape[0] != gram_raw.shape[1]:
        raise ValueError(f"gram_raw must be square, got shape={gram_raw.shape}")

    row_mean = gram_raw.mean(axis=1, dtype=np.float64)
    grand_mean = float(row_mean.mean())
    centered = gram_raw - row_mean[:, None] - row_mean[None, :] + grand_mean
    return 0.5 * (centered + centered.T), row_mean, grand_mean


def pairwise_distances_from_gram(gram: np.ndarray) -> np.ndarray:
    diag = np.diag(gram).astype(np.float64, copy=False)
    dist2 = diag[:, None] + diag[None, :] - (2.0 * gram)
    np.maximum(dist2, 0.0, out=dist2)
    np.fill_diagonal(dist2, 0.0)
    return np.sqrt(dist2, out=dist2)


def truncated_svd_dual(
    gram_centered: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(gram_centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    positive = eigvals > 0.0
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]
    if eigvals.size == 0:
        raise RuntimeError("No positive singular values found in dual SVD")

    rank = min(n_components, eigvals.size)
    s = np.sqrt(eigvals[:rank])
    u = eigvecs[:, :rank]
    z = u * s
    return z, s, u


def compute_nn_indices(distance_matrix: np.ndarray) -> np.ndarray:
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    return np.argmin(d, axis=1)


def compute_representation_metrics(
    dx: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray | None,
    run_offline_label_diagnostics: bool,
) -> dict[str, Any]:
    dz = pairwise_distances(z, metric="euclidean")
    tri = np.triu_indices_from(dx, k=1)

    x_flat = dx[tri]
    z_flat = dz[tri]

    metrics: dict[str, Any] = {
        "pairwise_distance_spearman": float(spearmanr(x_flat, z_flat).correlation),
        "pairwise_distance_pearson": float(pearsonr(x_flat, z_flat).statistic),
        "mean_relative_distance_error": float(np.mean(np.abs(x_flat - z_flat) / (x_flat + 1e-8))),
    }

    if labels is None:
        return metrics

    nn_x = compute_nn_indices(dx)
    nn_z = compute_nn_indices(dz)
    metrics["nn_overlap"] = float(np.mean(nn_x == nn_z))
    metrics["z_1nn_label_agreement"] = float(np.mean(labels[nn_z] == labels))

    if run_offline_label_diagnostics:
        unique_labels, counts = np.unique(labels, return_counts=True)
        if unique_labels.size >= 2 and counts.min() >= 2:
            n_splits = min(5, int(counts.min()))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=5000)
            scores = cross_val_score(clf, z, labels, cv=cv, scoring="accuracy")
            metrics["linear_probe_cv_accuracy_mean"] = float(scores.mean())
            metrics["linear_probe_cv_accuracy_std"] = float(scores.std())

            z0 = z[labels == 0]
            z1 = z[labels == 1]
            c0 = z0.mean(axis=0)
            c1 = z1.mean(axis=0)
            metrics["class_centroid_distance"] = float(np.linalg.norm(c0 - c1))
            metrics["within_class_mean_distance_label0"] = float(np.mean(np.linalg.norm(z0 - c0, axis=1)))
            metrics["within_class_mean_distance_label1"] = float(np.mean(np.linalg.norm(z1 - c1, axis=1)))

    return metrics


def fit_dual_svd_basis_from_items(
    *,
    items: list[ManifestItem],
    component_grid: list[int],
    block_size: int,
    dtype: np.dtype,
) -> tuple[StreamingSVDBasis, list[str]]:
    if not items:
        raise ValueError("No adapters provided for SVD fitting")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    first_adapter_path = items[0].adapter_path
    expected_pairs, expected_a_shapes, expected_b_shapes, layers, n_features = _discover_delta_schema(
        adapter_path=first_adapter_path,
    )

    adapter_paths = [item.adapter_path for item in items]
    for adapter_path in adapter_paths:
        check_consistency(
            adapter_path=adapter_path,
            expected_pairs=list(expected_pairs),
            expected_a_shapes=list(expected_a_shapes),
            expected_b_shapes=list(expected_b_shapes),
        )

    gram_start = perf_counter()
    gram_raw = compute_train_gram_from_delta_factors(
        adapter_paths=adapter_paths,
        expected_pairs=expected_pairs,
        dtype=dtype,
    )
    gram_stream_time_seconds = float(perf_counter() - gram_start)

    gram_centered, train_row_mean_raw, train_grand_mean_raw = center_sample_gram(gram_raw)
    resolved_grid, warnings, max_rank = sanitize_component_grid(
        requested=component_grid,
        n_samples=len(items),
        n_features=n_features,
    )

    z_train_max, singular_values, u = truncated_svd_dual(
        gram_centered=gram_centered,
        n_components=max(resolved_grid),
    )

    if singular_values.size < max(resolved_grid):
        resolved_grid = [k for k in resolved_grid if k <= singular_values.size]
        warnings.append(
            f"Requested max k={max(component_grid)}, but only {singular_values.size} positive singular values were available"
        )
        if not resolved_grid:
            raise RuntimeError("No valid SVD component counts after positive-singular-value filtering")

    basis = StreamingSVDBasis(
        z_train_max=z_train_max,
        singular_values=singular_values,
        u=u,
        expected_pairs=expected_pairs,
        expected_a_shapes=expected_a_shapes,
        expected_b_shapes=expected_b_shapes,
        layers=layers,
        n_features=n_features,
        resolved_component_grid=resolved_grid,
        max_rank=max_rank,
        gram_centered=gram_centered,
        gram_stream_time_seconds=gram_stream_time_seconds,
        train_row_mean_raw=train_row_mean_raw,
        train_grand_mean_raw=train_grand_mean_raw,
    )
    return basis, warnings


def project_items_with_dual_basis_streamed(
    *,
    train_items: list[ManifestItem],
    target_items: list[ManifestItem],
    basis: StreamingSVDBasis,
    rank: int,
    block_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    train_paths = [item.adapter_path for item in train_items]
    target_paths = [item.adapter_path for item in target_items]
    n_train = len(train_paths)

    if basis.u.shape[0] != n_train:
        raise ValueError(f"basis.u rows must match n_train={n_train}, got {basis.u.shape[0]}")
    if rank > basis.singular_values.size:
        raise ValueError(f"Requested rank={rank} exceeds available singular values={basis.singular_values.size}")

    expected_pairs = basis.expected_pairs
    expected_a_shapes = basis.expected_a_shapes
    expected_b_shapes = basis.expected_b_shapes

    for adapter_path in train_paths:
        check_consistency(
            adapter_path=adapter_path,
            expected_pairs=list(expected_pairs),
            expected_a_shapes=list(expected_a_shapes),
            expected_b_shapes=list(expected_b_shapes),
        )
    for adapter_path in target_paths:
        check_consistency(
            adapter_path=adapter_path,
            expected_pairs=list(expected_pairs),
            expected_a_shapes=list(expected_a_shapes),
            expected_b_shapes=list(expected_b_shapes),
        )

    cross_raw = compute_cross_gram_from_delta_factors(
        train_paths=train_paths,
        target_paths=target_paths,
        expected_pairs=expected_pairs,
        dtype=dtype,
    )
    row_mean_target = cross_raw.mean(axis=1, dtype=np.float64)
    centered_cross = (
        cross_raw
        - row_mean_target[:, None]
        - basis.train_row_mean_raw[None, :]
        + basis.train_grand_mean_raw
    )

    u_rank = basis.u[:, :rank]
    s_rank = basis.singular_values[:rank]
    inv_s = np.divide(1.0, s_rank, out=np.zeros_like(s_rank), where=s_rank > 0.0)
    z = (centered_cross @ u_rank) * inv_s[None, :]
    return z


def extract_svd_embeddings(
    *,
    items: list[ManifestItem],
    n_components: int | None,
    component_grid: list[int],
    block_size: int,
    dtype: np.dtype,
    acceptance_spearman_threshold: float,
    acceptance_variance_threshold: float,
    run_offline_label_diagnostics: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[str], dict[str, Any], list[str]]:
    basis, warnings = fit_dual_svd_basis_from_items(
        items=items,
        component_grid=component_grid,
        block_size=block_size,
        dtype=dtype,
    )

    labels_list = [item.label for item in items]
    labels = np.asarray(labels_list, dtype=np.int32) if all(x is not None for x in labels_list) else None
    model_names = [item.model_name for item in items]

    if n_components is None:
        chosen_k = basis.resolved_component_grid[-1]
    else:
        if n_components not in basis.resolved_component_grid:
            raise ValueError(
                f"Requested n_components={n_components}, available values={basis.resolved_component_grid}"
            )
        chosen_k = n_components

    n_samples = len(items)
    total_energy = float(np.trace(basis.gram_centered))
    total_variance = float(total_energy / max(1, n_samples - 1))
    dx = pairwise_distances_from_gram(basis.gram_centered)

    svd_info: dict[str, Any] = {}
    acceptance_candidates: list[dict[str, Any]] = []
    for k in basis.resolved_component_grid:
        z_k = basis.z_train_max[:, :k]
        s_k = basis.singular_values[:k]
        explained_variance = (s_k ** 2) / max(1, n_samples - 1)
        explained_variance_ratio = explained_variance / max(1e-12, total_variance)
        cumulative_variance = float(np.sum(explained_variance_ratio))
        captured_energy = float(np.sum(s_k ** 2))
        residual_energy = max(0.0, total_energy - captured_energy)
        reconstruction_mse = residual_energy / max(1, n_samples * basis.n_features)
        relative_reconstruction_error = residual_energy / (total_energy + 1e-12)

        representativeness = compute_representation_metrics(
            dx=dx,
            z=z_k,
            labels=labels,
            run_offline_label_diagnostics=run_offline_label_diagnostics,
        )
        svd_info[str(k)] = {
            "n_components": int(k),
            "cumulative_variance": cumulative_variance,
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "singular_values": s_k.tolist(),
            "reconstruction_error": float(reconstruction_mse),
            "relative_error": float(relative_reconstruction_error),
            "representativeness": representativeness,
        }

        spearman_val = representativeness["pairwise_distance_spearman"]
        acceptance_candidates.append(
            {
                "k": int(k),
                "spearman": float(spearman_val),
                "cumulative_variance": float(cumulative_variance),
                "passes_gate": bool(
                    spearman_val >= acceptance_spearman_threshold
                    and cumulative_variance >= acceptance_variance_threshold
                ),
            }
        )

    gate_winner = next((x for x in acceptance_candidates if x["passes_gate"]), None)

    metadata: dict[str, Any] = {
        "extractor": "svd",
        "extractor_version": SVD_EXTRACTOR_VERSION,
        "n_models": len(model_names),
        "n_features_raw": int(basis.n_features),
        "layers": basis.layers,
        "resolved_component_grid": basis.resolved_component_grid,
        "chosen_k": int(chosen_k),
        "max_rank": int(basis.max_rank),
        "acceptance_spearman_threshold": acceptance_spearman_threshold,
        "acceptance_variance_threshold": acceptance_variance_threshold,
        "acceptance_winner": gate_winner,
        "acceptance_candidates": acceptance_candidates,
        "svd_info": svd_info,
        "gram_stream_time_seconds": basis.gram_stream_time_seconds,
        "input_representation": "delta_factors",
    }

    z = basis.z_train_max[:, :chosen_k].astype(np.float32)
    return z, labels, model_names, metadata, warnings
