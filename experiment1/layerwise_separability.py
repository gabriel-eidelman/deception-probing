"""
Experiment 1: Layer-wise linear separability of deceptive intent.

Pipeline:
  1. Load the pilot dataset (PILOT_SET) and labels.
  2. Extract residual stream activations at every layer of Llama-3.1-Instruct
     via Modal, at the last-prompt-token position.
  3. Train an L2 logistic probe at each layer with stratified 5-fold CV.
  4. Compute baselines: majority class, and bag-of-words logistic regression
     on the user message + system prompt (per the PAP's baseline spec).
  5. Plot AUROC vs layer with error bars and a baseline overlay.
  6. Report the reference layer (peak-AUROC layer) for use in downstream
     experiments.
  7. Save activations, results, and the figure to disk.

Usage:
    python experiment_1_layerwise.py

You can override the model and a few CV settings via CLI flags; see the
argparse block at the bottom.

Note on dataset size: with n=54 the CV variance will be appreciable. The
script reports per-fold AUROC alongside the mean so you can sanity-check
whether the layer-profile shape is consistent across folds or driven by
a single noisy split.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# Local modules
from pilot_dataset import PILOT_SET
from extract_activations import extract_activations_batch
from probing import probe_all_layers, find_reference_layer, LayerProbeResult


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pilot_with_labels(label_key: str = "label") -> tuple[list[dict], np.ndarray]:
    """Load PILOT_SET and pull out labels.

    Assumes each prompt dict in PILOT_SET has a `label` field (0=honest,
    1=deceptive) from the output-verified labeling pipeline. If labels are
    stored elsewhere (separate JSON, CSV), modify this function to merge
    them in.

    Returns:
        prompts: list of dicts (passed to the Modal extractor as-is).
        labels:  np.ndarray of shape (n,), dtype int.
    """
    if not all(label_key in p for p in PILOT_SET):
        # Friendlier error than a KeyError mid-loop.
        missing = sum(1 for p in PILOT_SET if label_key not in p)
        raise KeyError(
            f"{missing}/{len(PILOT_SET)} PILOT_SET entries missing '{label_key}' field. "
            "Update load_pilot_with_labels() to join labels from your labeling pipeline."
        )

    labels = np.array([int(p[label_key]) for p in PILOT_SET], dtype=int)
    # Strip the label out of the dict we send to the model — we don't want
    # the label to be visible in any way, and extra keys are ignored by the
    # extractor anyway, but this is clean.
    prompts = [
        {"system": p["system"], "message": p["message"]}
        for p in PILOT_SET
    ]
    return prompts, labels


# ---------------------------------------------------------------------------
# Baselines (PAP Section 3.2: "Baselines for all experiments are majority-class
# prediction and a bag-of-words logistic regression on prompt text.")
# ---------------------------------------------------------------------------

def majority_class_baseline(labels: np.ndarray) -> dict:
    """Majority-class predictor. AUROC is by definition 0.5; we still report
    accuracy as a sanity-check on class balance."""
    majority = int(np.bincount(labels).argmax())
    acc = float(np.mean(labels == majority))
    return {
        "name": "majority_class",
        "auroc_mean": 0.5,
        "auroc_std": 0.0,
        "accuracy_mean": acc,
        "accuracy_std": 0.0,
        "majority_class": majority,
        "class_balance": {
            "n_honest": int(np.sum(labels == 0)),
            "n_deceptive": int(np.sum(labels == 1)),
        },
    }


def bow_baseline(
    prompts: list[dict],
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    random_state: int = 0,
) -> dict:
    """TF-IDF bag-of-words logistic regression on the concatenated system +
    user text. This baseline tells us how much of the deception signal is
    recoverable from prompt text alone, without any model internals. A high
    BoW AUROC would indicate that the labels are partly predictable from
    prompt surface features (e.g., elicitation strategies are textually
    distinct), and the probe needs to beat this convincingly to claim it's
    measuring something about the model's state rather than the prompt."""

    texts = [p["system"] + " " + p["message"] for p in prompts]
    texts_arr = np.array(texts)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aurocs, accs = [], []

    for train_idx, val_idx in skf.split(texts_arr, labels):
        # Fit vectorizer on train only — important to avoid vocab leakage.
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
            lowercase=True,
        )
        X_train = vec.fit_transform(texts_arr[train_idx])
        X_val = vec.transform(texts_arr[val_idx])
        y_train, y_val = labels[train_idx], labels[val_idx]

        clf = LogisticRegression(
            penalty="l2", C=C, solver="liblinear",
            max_iter=2000, random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_score = clf.decision_function(X_val)
        y_pred = clf.predict(X_val)

        if len(np.unique(y_val)) < 2:
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(y_val, y_score))
        accs.append(accuracy_score(y_val, y_pred))

    return {
        "name": "bow_tfidf_logreg",
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "auroc_per_fold": [float(a) for a in aurocs],
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_layerwise_auroc(
    results: list[LayerProbeResult],
    baselines: dict,
    reference_layer: int,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """AUROC vs layer with shaded ±1 SD band, baselines as dashed lines,
    reference layer highlighted."""
    layers = np.array([r.layer for r in results])
    means = np.array([r.auroc_mean for r in results])
    stds = np.array([r.auroc_std for r in results])

    fig, ax = plt.subplots(figsize=(9, 5))

    # Probe AUROC curve with ±SD band.
    ax.plot(layers, means, marker="o", linewidth=1.5, label="Probe (L2 logistic)")
    ax.fill_between(layers, means - stds, means + stds, alpha=0.2,
                    label="±1 SD across CV folds")

    # Baselines.
    ax.axhline(baselines["majority"]["auroc_mean"], linestyle=":", color="grey",
               label=f"Majority class (AUROC=0.5)")
    bow_mean = baselines["bow"]["auroc_mean"]
    ax.axhline(bow_mean, linestyle="--", color="firebrick",
               label=f"BoW baseline (AUROC={bow_mean:.2f})")

    # Reference layer marker.
    ax.axvline(reference_layer, linestyle="-", color="seagreen", alpha=0.5,
               label=f"Reference layer = {reference_layer}")
    ax.scatter(
        [reference_layer], [means[reference_layer]],
        s=120, facecolor="none", edgecolor="seagreen", linewidth=2, zorder=5,
    )

    ax.set_xlabel("Layer index (0 = embedding)")
    ax.set_ylabel("Cross-validated AUROC")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Layer-wise linear separability of deception{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    results: list[LayerProbeResult],
    baselines: dict,
    reference_layer: int,
    labels: np.ndarray,
) -> None:
    """Pretty-print the headline stats for interpreting the experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 SUMMARY: Layer-wise linear separability")
    print("=" * 70)

    n = len(labels)
    n_dec = int(np.sum(labels == 1))
    n_hon = n - n_dec
    print(f"\nDataset: n = {n} (honest = {n_hon}, deceptive = {n_dec}, "
          f"prevalence = {n_dec / n:.2f})")

    ref = results[reference_layer]
    print(f"\nReference layer (peak AUROC): layer {reference_layer}")
    print(f"  AUROC      = {ref.auroc_mean:.3f} ± {ref.auroc_std:.3f}")
    print(f"  Accuracy   = {ref.accuracy_mean:.3f} ± {ref.accuracy_std:.3f}")
    print(f"  Per-fold AUROC: "
          f"{[f'{a:.3f}' for a in ref.auroc_per_fold]}")

    # Top 5 layers.
    sorted_idx = sorted(range(len(results)),
                        key=lambda i: results[i].auroc_mean, reverse=True)[:5]
    print("\nTop 5 layers by mean AUROC:")
    for i in sorted_idx:
        r = results[i]
        print(f"  layer {r.layer:>3d}: AUROC = {r.auroc_mean:.3f} ± {r.auroc_std:.3f}")

    print("\nBaselines:")
    print(f"  Majority class: AUROC = 0.500, accuracy = "
          f"{baselines['majority']['accuracy_mean']:.3f}")
    print(f"  BoW TF-IDF:     AUROC = {baselines['bow']['auroc_mean']:.3f} "
          f"± {baselines['bow']['auroc_std']:.3f}")

    # Probe-over-BoW lift at reference layer.
    lift = ref.auroc_mean - baselines["bow"]["auroc_mean"]
    print(f"\nProbe AUROC lift over BoW baseline at reference layer: "
          f"{lift:+.3f}")
    if lift < 0.05:
        print("  ⚠  Probe barely beats text-only baseline. Activations may not "
              "carry construct signal beyond prompt surface features.")
    elif lift < 0.10:
        print("  ⚠  Modest lift over text baseline; interpret reference layer "
              "with caution.")
    else:
        print("  ✓  Probe substantially exceeds text baseline.")

    # Variance across folds at reference layer.
    if ref.auroc_std > 0.10:
        print(f"  ⚠  High CV variance at reference layer (SD = {ref.auroc_std:.3f}). "
              "Layer selection may be unstable; n=54 is small.")

    print("=" * 70 + "\n")


def save_results(
    results: list[LayerProbeResult],
    baselines: dict,
    reference_layer: int,
    activations: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """Persist everything needed for downstream experiments and reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Activations + labels: needed for Experiments 2-6.
    np.savez_compressed(
        out_dir / "activations.npz",
        activations=activations,
        labels=labels,
    )

    # Per-layer results as JSON.
    results_dict = {
        "model_name": model_name,
        "n_examples": int(len(labels)),
        "reference_layer": int(reference_layer),
        "layer_results": [
            {
                "layer": r.layer,
                "auroc_mean": r.auroc_mean,
                "auroc_std": r.auroc_std,
                "auroc_per_fold": r.auroc_per_fold,
                "accuracy_mean": r.accuracy_mean,
                "accuracy_std": r.accuracy_std,
            }
            for r in results
        ],
        "baselines": baselines,
    }
    with open(out_dir / "experiment_1_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Saved activations to {out_dir / 'activations.npz'}")
    print(f"Saved results to {out_dir / 'experiment_1_results.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: layer-wise probing")
    parser.add_argument(
        "--model-subdir", default="Llama-3.1-8B-Instruct",
        help="Model directory under /models in the Modal volume. "
             "Use 'Llama-3.1-70B-Instruct' for the full run.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse L2 strength for the probe.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./experiment_1_outputs"),
    )
    parser.add_argument(
        "--cached-activations", type=Path, default=None,
        help="Path to a precomputed activations.npz to skip Modal extraction.",
    )
    args = parser.parse_args()

    # ----- Step 1: load data -----
    print("Loading PILOT_SET...")
    prompts, labels = load_pilot_with_labels()
    print(f"  {len(prompts)} prompts loaded. "
          f"Class balance: {np.bincount(labels).tolist()}")

    # ----- Step 2: get activations -----
    if args.cached_activations is not None and args.cached_activations.exists():
        print(f"Loading cached activations from {args.cached_activations}...")
        data = np.load(args.cached_activations)
        activations = data["activations"]
        # Allow labels to be regenerated from PILOT_SET; just sanity-check shape.
        assert activations.shape[0] == len(prompts), \
            "Cached activations don't match current PILOT_SET length."
        model_name = args.model_subdir + " (cached)"
    else:
        print(f"Extracting activations via Modal ({args.model_subdir})...")
        result = extract_activations_batch(prompts, model_subdir=args.model_subdir)
        activations = result["activations"]
        model_name = result["model_name"]
        print(f"  Activations shape: {activations.shape}  "
              f"(n_examples, n_layers+1, hidden_dim)")

    # ----- Step 3: layer-wise probing -----
    print("\nTraining probes at each layer...")
    layer_results = probe_all_layers(
        activations, labels,
        n_folds=args.n_folds, C=args.C, random_state=args.seed,
        verbose=True,
    )
    reference_layer = find_reference_layer(layer_results)

    # ----- Step 4: baselines -----
    print("\nComputing baselines...")
    baselines = {
        "majority": majority_class_baseline(labels),
        "bow": bow_baseline(prompts, labels, n_folds=args.n_folds,
                            random_state=args.seed),
    }
    print(f"  Majority-class accuracy: {baselines['majority']['accuracy_mean']:.3f}")
    print(f"  BoW AUROC: {baselines['bow']['auroc_mean']:.3f} "
          f"± {baselines['bow']['auroc_std']:.3f}")

    # ----- Step 5: report + save -----
    print_summary(layer_results, baselines, reference_layer, labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_results(
        layer_results, baselines, reference_layer,
        activations, labels, model_name, args.out_dir,
    )
    plot_layerwise_auroc(
        layer_results, baselines, reference_layer,
        args.out_dir / "layerwise_auroc.png",
        title_suffix=f"  ({model_name}, n={len(labels)})",
    )

    print(f"\nReference layer for downstream experiments: {reference_layer}")
    return layer_results, baselines, reference_layer


if __name__ == "__main__":
    main()