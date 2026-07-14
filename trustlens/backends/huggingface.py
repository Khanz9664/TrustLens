"""
trustlens.backends.huggingface
===============================
Prediction resolver for Hugging Face `transformers` text-classification pipelines.

Architecture
------------
This backend translates a `transformers.TextClassificationPipeline` (and raw string
inputs) into a standardized `PredictionBundle`.

Probability Extraction Strategy
--------------------------------
* Calls the pipeline with `top_k=None` to retrieve scores for every class.
* Parses the resulting list of lists of `{'lab el': str, 'score': float}` dicts into
  a rectangular (n_samples, n_classes) array.

Label Mapping Behavior
-----------------------
* Prefers the pipeline's native `model.config.id2label` mapping to establish a
  stable, canonical column order.
* Falls back to the label order observed in the first sample's output if no
  `id2label` mapping is available, and validates that every subsequent sample uses
  the same label set.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from trustlens.backends.types import PredictionBundle, UnsupportedModelError


def _get_id2label(model: Any) -> Optional[dict]:
    """Best-effort extraction of the pipeline's id2label mapping."""
    config = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(config, "id2label", None)
    if not id2label:
        return None
    return id2label


def _parse_pipeline_output(model: Any, raw_output: list) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Parse `TextClassificationPipeline(..., top_k=None)` output into a rectangular
    (n_samples, n_classes) probability array plus the ordered class labels.
    """
    id2label = _get_id2label(model)

    if id2label:
        ordered_ids = sorted(id2label)
        ordered_labels = [id2label[i] for i in ordered_ids]
        label2col = {label: col for col, label in enumerate(ordered_labels)}
    else:
        # No config available (e.g. mocked pipeline) — derive canonical order
        ordered_labels = None
        label2col = None

    rows = []
    for i, sample_scores in enumerate(raw_output):
        if label2col is None:
            ordered_labels = [entry["label"] for entry in sample_scores]
            label2col = {label: col for col, label in enumerate(ordered_labels)}

        row = [0.0] * len(ordered_labels)
        for entry in sample_scores:
            col = label2col.get(entry["label"])
            if col is None:
                raise ValueError(
                    f"Sample {i} produced label '{entry['label']}', which is not "
                    "present in the canonical label set derived from the pipeline "
                    "(either model.config.id2label or the first sample's labels). "
                    "Inconsistent per-sample label ordering is not supported."
                )
            row[col] = entry["score"]
        rows.append(row)

    y_prob = np.asarray(rows, dtype=float)
    class_labels = np.asarray(ordered_labels) if ordered_labels is not None else None
    return y_prob, class_labels


def resolve(
    model: Any,
    X: Any,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[np.ndarray] = None,
) -> PredictionBundle:
    """
    Resolve predictions and probabilities from a Hugging Face
    `TextClassificationPipeline`.

    Runs the pipeline over raw string inputs `X` (unless `y_prob` is already
    supplied), parses per-sample label/score dicts into a stable (n, n_classes)
    probability matrix, and derives predictions via argmax — mapped back to
    semantic string labels (via `id2label`) whenever a label mapping is
    available, so `y_pred` lands in the same space as a typical string-labeled
    `y_true`. Falls back to raw ordinal indices only when no label mapping
    exists at all.
    """
    try:
        import transformers
        from transformers import TextClassificationPipeline
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required to use the huggingface "
            "resolver. Install it via `pip install trustlens[transformers,torch]`."
        ) from exc

    # 1. Pipeline type validation
    is_text_classification = isinstance(model, TextClassificationPipeline) or (
        getattr(model, "task", None) == "text-classification"
    )
    if not is_text_classification:
        raise UnsupportedModelError(
            model_type=f"{type(model).__module__}.{type(model).__name__}",
            supported_frameworks=["huggingface"],
        )

    # 2. Resolve probabilities
    resolved_labels_from_output: Optional[np.ndarray] = None
    if y_prob is None:
        raw_output = model(list(X), top_k=None)
        y_prob, resolved_labels_from_output = _parse_pipeline_output(model, raw_output)
    else:
        y_prob = np.asarray(y_prob)

    # 3. Normalize probabilities (defensive: (n,) -> (n, 2), mirrors sklearn.py)
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)

    # 4. Resolve predictions
    id2label = _get_id2label(model)
    if y_pred is None:
        y_pred_indices = np.argmax(y_prob, axis=1)
        if id2label:
            ordered_ids = sorted(id2label)
            ordered_labels_arr = np.asarray([id2label[i] for i in ordered_ids])
            if len(ordered_labels_arr) == y_prob.shape[1]:
                y_pred = ordered_labels_arr[y_pred_indices]
            else:
                y_pred = y_pred_indices
        elif (
            resolved_labels_from_output is not None
            and len(resolved_labels_from_output) == y_prob.shape[1]
        ):
            y_pred = resolved_labels_from_output[y_pred_indices]
        elif class_labels is not None and len(class_labels) == y_prob.shape[1]:
            y_pred = np.asarray(class_labels)[y_pred_indices]
        else:
            y_pred = y_pred_indices
    else:
        y_pred = np.asarray(y_pred)
    if id2label:
        ordered_ids = sorted(id2label)
        resolved_class_labels = np.asarray([id2label[i] for i in ordered_ids])
    elif resolved_labels_from_output is not None:
        resolved_class_labels = resolved_labels_from_output
    elif class_labels is not None:
        resolved_class_labels = np.asarray(class_labels)
    else:
        resolved_class_labels = None

    metadata = {
        "resolver": "huggingface",
        "detection_method": "manual",  # Default value. Registry may override.
        "framework_version": getattr(transformers, "__version__", "unknown"),
    }

    return PredictionBundle(
        y_pred=y_pred,
        y_prob=y_prob,
        framework="huggingface",
        class_labels=resolved_class_labels,
        metadata=metadata,
    )
