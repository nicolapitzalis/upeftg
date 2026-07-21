"""Dataset splitting and cross-validation policy."""

from .cross_validation import build_cv_splits, resolve_cv_strategy

__all__ = ["build_cv_splits", "resolve_cv_strategy"]
