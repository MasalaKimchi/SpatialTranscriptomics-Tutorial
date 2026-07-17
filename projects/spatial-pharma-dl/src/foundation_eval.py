"""Leakage-resistant classification evaluation for frozen slide embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from .validation import require_non_empty


TASK_CLASSES = {
    "all_4class": (
        "tumor_epithelial",
        "immune_enriched",
        "stromal",
        "other",
    ),
    "confident_3class": (
        "tumor_epithelial",
        "immune_enriched",
        "stromal",
    ),
}


@dataclass(frozen=True)
class ProbeCandidate:
    """A fold-selectable embedding transform and logistic-probe setting."""

    preprocessing: str
    c: float
    class_weight: str | None = "balanced"

    @property
    def name(self) -> str:
        weight = self.class_weight or "none"
        return f"{self.preprocessing}|C={self.c:g}|weight={weight}"


DEFAULT_CANDIDATES = (
    ProbeCandidate("raw", c=0.01, class_weight="balanced"),
    ProbeCandidate("l2", c=0.1, class_weight="balanced"),
    ProbeCandidate("slide_zscore_l2", c=0.01, class_weight="balanced"),
    ProbeCandidate("slide_zscore_l2", c=0.01, class_weight=None),
)


def preprocess_slide_embeddings(
    embeddings: np.ndarray, mode: str, eps: float = 1e-6
) -> np.ndarray:
    """Apply an unlabeled, per-slide transform before combining slide data.

    ``slide_zscore_l2`` is transductive: it uses the feature distribution of the
    complete inference slide, but never its labels. This matches whole-slide
    deployment and must not be used for isolated single-patch inference.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "l2":
        return normalize(x, norm="l2")
    if mode == "slide_zscore_l2":
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)
        return normalize(x, norm="l2")
    raise ValueError(f"Unknown embedding preprocessing mode: {mode!r}")


def prepare_classification_task(
    embeddings: np.ndarray,
    labels: pd.DataFrame,
    task: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, tuple[str, ...]]:
    """Filter/remap a slide to a globally consistent classification task."""
    if task not in TASK_CLASSES:
        raise ValueError(f"Unknown task {task!r}; choose from {tuple(TASK_CLASSES)}")
    class_names = TASK_CLASSES[task]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    keep = labels["tme_class"].isin(class_names).to_numpy()
    filtered_labels = labels.loc[keep].reset_index(drop=True)
    require_non_empty(
        filtered_labels,
        stage="foundation_task_filter",
        subject=f"retained rows for task {task}",
        guidance="Provide labels belonging to one of the configured task classes.",
    )
    y = filtered_labels["tme_class"].map(class_to_id).to_numpy(dtype=np.int64)
    return embeddings[keep], y, filtered_labels, class_names


def _fit_probe(
    train_parts: list[tuple[np.ndarray, np.ndarray]],
    candidate: ProbeCandidate,
    seed: int,
) -> Any:
    require_non_empty(
        train_parts,
        stage="nested_loso_probe_training",
        subject="training slide parts",
        guidance="Provide at least one non-empty outer-training slide part.",
    )
    for part_index, (embeddings, labels) in enumerate(train_parts):
        require_non_empty(
            embeddings,
            stage="nested_loso_probe_training",
            subject=f"training embeddings part {part_index}",
            guidance="Retain at least one embedding in every training slide part.",
        )
        require_non_empty(
            labels,
            stage="nested_loso_probe_training",
            subject=f"training labels part {part_index}",
            guidance="Retain at least one label in every training slide part.",
        )
    x_train = np.concatenate(
        [
            preprocess_slide_embeddings(x, candidate.preprocessing)
            for x, _ in train_parts
        ]
    )
    y_train = np.concatenate([y for _, y in train_parts])
    if np.unique(y_train).size < 2:
        raise ValueError("A classification probe needs at least two training classes.")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=candidate.c,
            class_weight=candidate.class_weight,
            max_iter=2000,
            random_state=seed,
        ),
    )
    model.fit(x_train, y_train)
    return model


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, float]:
    fixed_labels = np.arange(n_classes)
    present_labels = np.unique(y_true)
    balanced_accuracy = np.mean(
        [np.mean(y_pred[y_true == label] == label) for label in present_labels]
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        # sklearn's helper warns when a globally supported class is absent from one
        # held-out slide. Averaging recalls over classes present in that slide is the
        # same definition without treating this expected LOSO condition as an error.
        "balanced_accuracy": float(balanced_accuracy),
    }


def _majority_baseline(
    train_parts: list[tuple[np.ndarray, np.ndarray]],
    y_val: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    y_train = np.concatenate([y for _, y in train_parts])
    majority_class = int(np.bincount(y_train, minlength=n_classes).argmax())
    return _metrics(y_val, np.full_like(y_val, majority_class), n_classes)


def _select_candidate(
    train_slide_ids: list[str],
    task_data: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    n_classes: int,
    candidates: tuple[ProbeCandidate, ...],
    seed: int,
) -> tuple[ProbeCandidate, pd.DataFrame]:
    rows = []
    for candidate in candidates:
        fold_scores = []
        for val_slide in train_slide_ids:
            inner_train = [task_data[s][:2] for s in train_slide_ids if s != val_slide]
            model = _fit_probe(inner_train, candidate, seed)
            x_val, y_val, _ = task_data[val_slide]
            pred = model.predict(
                preprocess_slide_embeddings(x_val, candidate.preprocessing)
            )
            fold_scores.append(_metrics(y_val, pred, n_classes)["macro_f1"])
        rows.append(
            {
                "candidate": candidate.name,
                "inner_mean_macro_f1": float(np.mean(fold_scores)),
                "inner_min_macro_f1": float(np.min(fold_scores)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["inner_mean_macro_f1", "inner_min_macro_f1", "candidate"],
        ascending=[False, False, True],
    )
    selected_name = scores.iloc[0]["candidate"]
    selected = next(c for c in candidates if c.name == selected_name)
    return selected, scores.reset_index(drop=True)


def nested_loso_classification(
    slide_ids: list[str],
    slide_data: dict[str, tuple[np.ndarray, pd.DataFrame]],
    task: str,
    candidates: tuple[ProbeCandidate, ...] = DEFAULT_CANDIDATES,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Run nested LOSO selection and unbiased outer-slide evaluation."""
    task_data: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    class_names = TASK_CLASSES[task]
    unique_non_empty = [slide_id for slide_id in dict.fromkeys(slide_ids) if slide_id]
    require_non_empty(
        unique_non_empty,
        stage="nested_loso_admission",
        subject="unique non-empty slide IDs",
        minimum=3,
        guidance="Provide at least three distinct slides for nested LOSO.",
    )
    for slide_id in slide_ids:
        embeddings, labels = slide_data[slide_id]
        x, y, filtered, _ = prepare_classification_task(embeddings, labels, task)
        task_data[slide_id] = (x, y, filtered)

    rows = []
    details: dict[int, dict[str, Any]] = {}
    for fold, val_slide in enumerate(slide_ids):
        train_slides = [s for s in slide_ids if s != val_slide]
        selected, inner_scores = _select_candidate(
            train_slides,
            task_data,
            len(class_names),
            candidates,
            seed,
        )
        train_parts = [task_data[s][:2] for s in train_slides]
        model = _fit_probe(train_parts, selected, seed)
        x_val, y_val, val_labels = task_data[val_slide]
        require_non_empty(
            x_val,
            stage="nested_loso_probe_prediction",
            subject=f"held-out embeddings for slide {val_slide}",
            guidance="Retain at least one held-out task embedding before prediction.",
        )
        require_non_empty(
            y_val,
            stage="nested_loso_probe_prediction",
            subject=f"held-out labels for slide {val_slide}",
            guidance="Retain at least one held-out task label before prediction.",
        )
        y_pred = model.predict(
            preprocess_slide_embeddings(x_val, selected.preprocessing)
        )
        metrics = _metrics(y_val, y_pred, len(class_names))
        baseline = _majority_baseline(train_parts, y_val, len(class_names))
        coverage = len(y_val) / len(slide_data[val_slide][1])
        rows.append(
            {
                "fold": fold,
                "held_out_slide": val_slide,
                "task": task,
                "n_train": int(sum(len(y) for _, y in train_parts)),
                "n_test": int(len(y_val)),
                "coverage": float(coverage),
                "selected_candidate": selected.name,
                "inner_macro_f1": float(inner_scores.iloc[0]["inner_mean_macro_f1"]),
                **metrics,
                "majority_macro_f1": baseline["macro_f1"],
                "majority_balanced_accuracy": baseline["balanced_accuracy"],
            }
        )
        details[fold] = {
            "held_out_slide": val_slide,
            "task": task,
            "class_names": class_names,
            "selected": selected,
            "inner_scores": inner_scores,
            "labels": val_labels,
            "y_true": y_val,
            "y_pred": y_pred,
        }
    return pd.DataFrame(rows), details
