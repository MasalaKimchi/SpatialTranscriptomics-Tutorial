#!/usr/bin/env python3
"""Build the nested-LOSO foundation-model comparison notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PHARMA = Path(__file__).resolve().parents[1]
OUTPUT = PHARMA / "notebooks" / "07_foundation_model_comparison.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "spatial-tx",
        "language": "python",
        "name": "spatial-tx",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    nb.cells = [
        md(
            """
# Improving Frozen Foundation-Model TME Classification

This experiment addresses the low macro-F1 from the initial raw probe without fine-tuning
either encoder. It compares two independent pathology foundation models, repairs
the unsupported label space, and selects preprocessing/regularization using
**nested leave-one-slide-out validation**.

## tl;dr

> **Phikon is the stronger frozen encoder, but the dataset—not model capacity—is
> now the main bottleneck.** Under nested slide-held-out validation, Phikon reaches
> **0.320 macro-F1** on all four labels versus **0.294** for Kaiko and **0.187** for
> a fold-matched majority baseline. Excluding the ambiguous `other` label raises
> Phikon to **0.366** and Kaiko to **0.326**, but retains only **39.0%** of spots.
> The weakest Phikon fold remains **0.227**, so the result is improved rather than
> solved. Hyperparameters are selected using only the three training slides in
> each outer fold.
"""
        ),
        md(
            """
## Context & Methods

### Changes from the initial probe

1. Compare **Kaiko ViT-S/16** (384-D) with **Phikon ViT-B** (768-D).
2. Evaluate both the original four-class task and a three-class task that excludes
   ambiguous `other` spots.
3. Test raw, L2-normalized, and transductive slide-z-scored + L2 embeddings.
4. Select the transform and logistic regularization inside nested slide folds.
5. Include a majority-class macro-F1 baseline in every outer fold.

### Key assumptions

- Slide-wise standardization uses no target labels, but it does use the complete
  target-slide feature distribution. It is valid for whole-slide batch inference,
  not isolated patch inference.
- Excluding `other` improves label confidence but reduces coverage. Four-class and
  three-class F1 values therefore answer different questions and are not directly
  interchangeable.
- Both checkpoints are research/non-commercial. Results remain methodological,
  not decision-grade.

Model sources: [Kaiko](https://huggingface.co/1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms),
[Phikon](https://huggingface.co/owkin/phikon).
"""
        ),
        md("## Setup"),
        code(
            """
from pathlib import Path
import os, sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path.cwd()
if not (ROOT / "projects" / "spatial-pharma-dl").exists():
    ROOT = Path.cwd().resolve().parents[2]
PHARMA = ROOT / "projects" / "spatial-pharma-dl"
sys.path[:0] = [str(ROOT), str(PHARMA)]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix, silhouette_score

from src.data import load_config, pharma_outputs_dir
from src.foundation import foundation_model_spec, load_or_extract_slide_embeddings
from src.foundation_eval import (
    DEFAULT_CANDIDATES,
    TASK_CLASSES,
    nested_loso_classification,
    preprocess_slide_embeddings,
)

SEED = 0
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"kaiko_vits16": "#326B91", "phikon": "#C4912E"}

cfg = load_config()
slides = cfg["cohorts"]["oncology"]
model_names = ["kaiko_vits16", "phikon"]
tasks = ["all_4class", "confident_3class"]
figure_dir = pharma_outputs_dir() / "foundation_v2"
figure_dir.mkdir(parents=True, exist_ok=True)
print("Slides:", len(slides), "| models:", model_names, "| tasks:", tasks)
"""
        ),
        md("## Data"),
        md("### 1. Audit class support and three-class coverage"),
        code(
            """
labels = pd.concat(
    [pd.read_parquet(pharma_outputs_dir() / f"labels_{s}.parquet") for s in slides],
    ignore_index=True,
)
support = pd.crosstab(labels["slide_id"], labels["tme_class"])
display(support)

supported = set(TASK_CLASSES["confident_3class"])
coverage = (
    labels.assign(confident=labels["tme_class"].isin(supported))
    .groupby("slide_id", observed=True)["confident"]
    .agg(["sum", "count", "mean"])
    .reset_index()
)
coverage["excluded_other"] = coverage["count"] - coverage["sum"]
display(coverage.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
support.plot(kind="barh", stacked=True, ax=axes[0], colormap="tab20c", width=.72)
axes[0].set(title="Observed TME labels by slide", xlabel="Spots", ylabel="")
axes[0].legend(title="TME class", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
sns.barplot(data=coverage, y="slide_id", x="mean", color="#326B91", ax=axes[1])
axes[1].set(title="Coverage after excluding ambiguous 'other'", xlabel="Retained share", ylabel="", xlim=(0, 1))
axes[1].xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.savefig(figure_dir / "01_label_support_and_coverage.png", dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        md("### 2. Load validated cached embeddings for both encoders"),
        code(
            """
all_model_data = {}
embedding_audit = []
for model_name in model_names:
    model_cfg = load_config()
    model_cfg["foundation"]["model"] = model_name
    _, spec = foundation_model_spec(model_cfg)
    slide_data = {}
    for slide_id in slides:
        embeddings, aligned = load_or_extract_slide_embeddings(slide_id, labels, cfg=model_cfg)
        assert embeddings.shape == (len(aligned), spec.embedding_dim)
        assert np.isfinite(embeddings).all()
        assert aligned["spot_id"].is_unique
        slide_data[slide_id] = (embeddings, aligned)
        embedding_audit.append({
            "model": model_name,
            "slide_id": slide_id,
            "spots": len(aligned),
            "embedding_dim": embeddings.shape[1],
            "license": spec.license,
        })
    all_model_data[model_name] = slide_data

audit = pd.DataFrame(embedding_audit)
display(audit)
"""
        ),
        md("## Results"),
        md("### 3. Run nested LOSO model selection and outer evaluation"),
        code(
            """
result_frames = []
all_details = {}
for model_name in model_names:
    for task in tasks:
        print("Evaluating", model_name, task)
        frame, details = nested_loso_classification(
            slides,
            all_model_data[model_name],
            task=task,
            candidates=DEFAULT_CANDIDATES,
            seed=SEED,
        )
        frame.insert(0, "model", model_name)
        result_frames.append(frame)
        all_details[(model_name, task)] = details

results = pd.concat(result_frames, ignore_index=True)
results.to_csv(figure_dir / "nested_loso_results.csv", index=False)
display(results.round(3))
"""
        ),
        md("### 4. Compare encoders, tasks, and fold-matched baselines"),
        code(
            """
summary = (
    results.groupby(["task", "model"], observed=True)
    .agg(
        mean_macro_f1=("macro_f1", "mean"),
        min_macro_f1=("macro_f1", "min"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        majority_macro_f1=("majority_macro_f1", "mean"),
        mean_coverage=("coverage", "mean"),
    )
    .reset_index()
)
summary["f1_lift_vs_majority"] = summary["mean_macro_f1"] - summary["majority_macro_f1"]
display(summary.round(3))
summary.to_csv(figure_dir / "model_task_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, task in zip(axes, tasks):
    subset = summary[summary.task == task]
    sns.barplot(data=subset, x="model", y="mean_macro_f1", hue="model", palette=PALETTE, legend=False, ax=ax)
    baseline = subset["majority_macro_f1"].mean()
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.2, label=f"majority baseline ({baseline:.3f})")
    ax.set(title=task.replace("_", " "), xlabel="", ylabel="Mean outer-fold macro-F1", ylim=(0, max(.5, subset.mean_macro_f1.max() * 1.2)))
    ax.legend(frameon=False)
plt.suptitle("Frozen foundation-model classification under nested slide-level validation")
plt.tight_layout()
plt.savefig(figure_dir / "02_model_task_macro_f1.png", dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        md("### 5. Inspect fold stability"),
        code(
            """
fold_view = results.copy()
fold_view["model_task"] = fold_view["model"] + " | " + fold_view["task"]
fold_view["slide"] = fold_view["held_out_slide"].map({s: f"Slide {i+1}" for i, s in enumerate(slides)})
heatmap = fold_view.pivot(index="model_task", columns="slide", values="macro_f1")

plt.figure(figsize=(8.5, 4.5))
sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=max(.5, heatmap.max().max()), linewidths=.5, cbar_kws={"label": "Macro-F1"})
plt.title("Outer-fold macro-F1 by held-out slide")
plt.xlabel(""); plt.ylabel("")
plt.tight_layout()
plt.savefig(figure_dir / "03_fold_macro_f1_heatmap.png", dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        md("### 6. Verify what nested tuning selected"),
        code(
            """
selection = (
    results.groupby(["model", "task", "selected_candidate"], observed=True)
    .size().rename("outer_folds_selected").reset_index()
    .sort_values(["model", "task", "outer_folds_selected"], ascending=[True, True, False])
)
display(selection)

selection_plot = selection.copy()
selection_plot["model_task"] = selection_plot["model"] + " | " + selection_plot["task"]
plt.figure(figsize=(10, 5))
sns.barplot(data=selection_plot, y="model_task", x="outer_folds_selected", hue="selected_candidate", palette="muted")
plt.title("Probe settings selected by inner slide folds")
plt.xlabel("Number of outer folds"); plt.ylabel("")
plt.legend(title="Selected setting", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(figure_dir / "04_nested_selection_counts.png", dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        md("### 7. Compare confident-class confusion patterns"),
        code(
            """
task = "confident_3class"
class_names = TASK_CLASSES[task]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, model_name in zip(axes, model_names):
    details = all_details[(model_name, task)]
    y_true = np.concatenate([details[f]["y_true"] for f in sorted(details)])
    y_pred = np.concatenate([details[f]["y_pred"] for f in sorted(details)])
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True, cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set(title=model_name, xlabel="Predicted", ylabel="Observed")
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
plt.suptitle("Aggregated outer-fold confusion — confident three-class task")
plt.tight_layout()
plt.savefig(figure_dir / "05_confident_class_confusion.png", dpi=170, bbox_inches="tight")
plt.show()
"""
        ),
        md("### 8. Check whether normalization reduces slide structure"),
        code(
            """
from sklearn.decomposition import PCA
import umap

diagnostic_model = summary.sort_values("mean_macro_f1", ascending=False).iloc[0]["model"]
slide_data = all_model_data[diagnostic_model]
rng = np.random.default_rng(SEED)
sample_parts, sample_meta = [], []
for slide_id in slides:
    embeddings, metadata = slide_data[slide_id]
    n = min(1000, len(metadata))
    idx = np.sort(rng.choice(len(metadata), n, replace=False))
    sample_parts.append((embeddings[idx], slide_id))
    sample_meta.append(metadata.iloc[idx][["slide_id", "tme_class"]])
meta = pd.concat(sample_meta, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
silhouette_rows = []
for ax, mode, title in zip(axes, ["raw", "slide_zscore_l2"], ["Raw embeddings", "Slide-z-scored + L2"]):
    transformed = np.concatenate([preprocess_slide_embeddings(x, mode) for x, _ in sample_parts])
    pca = PCA(n_components=min(50, transformed.shape[1]), random_state=SEED).fit_transform(transformed)
    coords = umap.UMAP(n_neighbors=30, min_dist=.2, random_state=SEED, n_jobs=1).fit_transform(pca)
    view = meta.copy(); view[["UMAP1", "UMAP2"]] = coords
    sns.scatterplot(data=view, x="UMAP1", y="UMAP2", hue="slide_id", s=8, alpha=.65, linewidth=0, ax=ax)
    ax.set(title=title, xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2"); ax.grid(False)
    ax.legend().remove()
    silhouette_rows.append({
        "model": diagnostic_model,
        "preprocessing": mode,
        "slide_silhouette": silhouette_score(pca[:, :20], meta["slide_id"]),
    })
handles, labels_ = axes[1].get_legend_handles_labels()
fig.legend(handles, labels_, title="Slide", bbox_to_anchor=(1.01, .9), loc="upper left", frameon=False)
plt.suptitle(f"Slide structure before and after normalization — {diagnostic_model}")
plt.tight_layout()
plt.savefig(figure_dir / "06_normalization_umap.png", dpi=170, bbox_inches="tight")
plt.show()
display(pd.DataFrame(silhouette_rows).round(3))
"""
        ),
        md("## Checks"),
        code(
            """
checks = {
    "two_encoders_complete": set(audit.model) == set(model_names) and len(audit) == 8,
    "all_embeddings_finite": all(np.isfinite(x).all() for model in all_model_data.values() for x, _ in model.values()),
    "four_outer_folds_per_experiment": results.groupby(["model", "task"]).size().eq(4).all(),
    "held_out_slides_unique": results.groupby(["model", "task"])["held_out_slide"].nunique().eq(4).all(),
    "nested_settings_recorded": results["selected_candidate"].notna().all(),
    "majority_baselines_recorded": results["majority_macro_f1"].notna().all(),
    "three_class_coverage_explicit": results.loc[results.task == "confident_3class", "coverage"].lt(1).all(),
}
display(pd.Series(checks, name="passed").to_frame())
assert all(checks.values())
"""
        ),
        md(
            """
## Takeaways

1. **The original low score was partly an evaluation/preprocessing problem.** The
   nested all-class Kaiko result is 0.294, substantially above the initial raw
   probe result (0.141), without changing any encoder weights.
2. **Phikon transfers better on this cohort.** It improves mean macro-F1 over Kaiko
   by 0.026 on all four labels and 0.039 on the confident three-class task.
3. **Label confidence matters more than another small model swap.** Removing
   `other` gives Phikon 0.366 macro-F1, but discards 61.0% of spots; it is a
   selective high-confidence task, not a directly comparable replacement metric.
4. **Batch correction is incomplete.** Slide z-scoring + L2 was selected in every
   all-class outer fold and lowers the linear PCA slide silhouette from 0.357 to
   -0.001. The UMAP still separates slides, however, revealing residual nonlinear
   domain shift.
5. **Treat the ranking as preliminary.** There are only four slides, two are
   adjacent sections from one patient, labels are heuristic, and both encoders use
   non-commercial research weights.

### Next steps

1. Replace heuristic cluster annotations with consistent expert or molecular-rule labels.
2. Add independent patients and institutions before any fine-tuning.
3. Test multiscale and spatial-neighbor embedding aggregation.
4. Evaluate commercially permissible encoders after governance review.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    build_notebook()
