"""Feature artifact path and companion-file resolution."""

from __future__ import annotations

from pathlib import Path


DEFAULT_FEATURE_EXTRACT_ROOT = Path("runs")


def candidate_companion_paths(feature_path: Path, suffix: str) -> list[Path]:
    stem = feature_path.stem
    candidates: list[Path] = []
    if stem.endswith("_features"):
        prefix = stem[: -len("_features")]
        candidates.append(feature_path.with_name(f"{prefix}{suffix}"))
    candidates.append(feature_path.with_name(f"{stem}{suffix}"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_existing_companion_path(feature_path: Path, suffix: str, *, required: bool) -> Path:
    candidates = candidate_companion_paths(feature_path, suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find required companion file for {feature_path}. Tried: {joined}")
    return candidates[0]


def default_output_companion_path(feature_path: Path, suffix: str) -> Path:
    return candidate_companion_paths(feature_path, suffix)[0]


def resolve_feature_extract_root(feature_root: Path) -> Path:
    root = feature_root.expanduser()
    if not root.is_absolute():
        root = Path.cwd().resolve() / root
    return root.resolve()


def looks_like_explicit_path(path_spec: Path) -> bool:
    return path_spec.is_absolute() or len(path_spec.parts) > 1 or path_spec.suffix == ".npy"


def resolve_input_feature_path(feature_spec: Path, *, feature_root: Path) -> Path:
    candidate = feature_spec.expanduser()
    if looks_like_explicit_path(candidate):
        resolved = candidate if candidate.is_absolute() else Path.cwd().resolve() / candidate
        return resolved.resolve()

    run_name = candidate.name
    search_paths = [
        feature_root / run_name / "aggregation" / "features" / "spectral_features.npy",
        feature_root / run_name / "extraction" / "features" / "spectral_features.npy",
        feature_root / run_name / "merged" / "spectral_features.npy",
        feature_root / run_name / "features" / "spectral_features.npy",
    ]
    for path in search_paths:
        if path.exists():
            return path.resolve()

    joined = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Could not resolve feature run name '{run_name}' under {feature_root}. Tried: {joined}")


def resolve_output_feature_path(output_spec: Path, *, feature_root: Path) -> Path:
    candidate = output_spec.expanduser()
    if looks_like_explicit_path(candidate):
        resolved = candidate if candidate.is_absolute() else Path.cwd().resolve() / candidate
        output_path = resolved.resolve()
    else:
        output_path = (feature_root / candidate.name / "aggregation" / "features" / "spectral_features.npy").resolve()

    if output_path.suffix == "":
        output_path = output_path.with_name(output_path.name + ".npy")
    elif output_path.suffix != ".npy":
        raise ValueError(f"--output-filename must end with .npy: {output_path}")
    return output_path
