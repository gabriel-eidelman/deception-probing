"""
Layer-wise probing utilities for Experiment 1.

Given a (n_examples, n_layers+1, hidden_dim) activation tensor and binary
labels, this module trains an L2-regularized logistic regression probe at
each layer and reports cross-validated AUROC.

Design choices:
  - Stratified k-fold CV (k=5 by default). With n=54 the pilot is too small
    for a clean train/val/test split; stratified k-fold preserves class
    balance per fold and gives us a variance estimate across folds.
  - L2 regularization with C=1.0 by default. Experiment 5 in the PAP will
    sweep this; for layer selection we use a fixed sensible default.
  - Features are standardized within each fold (fit on train, apply to val)
    to avoid leakage. Llama residual stream activations have very different
    scales across layers, so standardization matters for the L2 penalty to
    behave consistently.
  - We report AUROC mean ± SD across folds, plus per-fold accuracy for
    sanity-checking.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


@dataclass
class LayerProbeResult:
    """Cross-validated probe performance at a single layer."""
    layer: int
    auroc_mean: float
    auroc_std: float
    auroc_per_fold: list[float]
    accuracy_mean: float
    accuracy_std: float
    n_train_per_fold: int  # constant across folds for clarity in logs
    n_val_per_fold: int


def probe_single_layer(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    random_state: int = 0,
) -> LayerProbeResult:
    """Train and CV-evaluate an L2 logistic probe on a single layer's activations.

    Args:
        X: (n_examples, hidden_dim) activation matrix for one layer.
        y: (n_examples,) binary labels in {0, 1}.
        n_folds: number of stratified CV folds.
        C: inverse L2 regularization strength.
        random_state: seed for fold shuffling.

    Returns:
        LayerProbeResult with mean/std AUROC and accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    aurocs: list[float] = []
    accs: list[float] = []
    n_train = n_val = 0

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize within the fold to avoid train/val leakage.
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_val_std = scaler.transform(X_val)

        # L2 logistic regression. max_iter bumped because high-dim activations
        # can need more iterations to converge under lbfgs.
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=5000,
            random_state=random_state,
        )
        clf.fit(X_train_std, y_train)

        # AUROC needs probabilities/scores; we use decision_function for the
        # log-odds (matches the PAP's "log-odds at the reference layer" language).
        y_score = clf.decision_function(X_val_std)
        y_pred = clf.predict(X_val_std)

        # Guard: AUROC is undefined if a fold's val set is single-class.
        # Stratified k-fold should prevent this, but check defensively.
        if len(np.unique(y_val)) < 2:
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(y_val, y_score))
        accs.append(accuracy_score(y_val, y_pred))

        n_train, n_val = len(train_idx), len(val_idx)

    return LayerProbeResult(
        layer=-1,  # caller sets this
        auroc_mean=float(np.nanmean(aurocs)),
        auroc_std=float(np.nanstd(aurocs)),
        auroc_per_fold=[float(a) for a in aurocs],
        accuracy_mean=float(np.mean(accs)),
        accuracy_std=float(np.std(accs)),
        n_train_per_fold=n_train,
        n_val_per_fold=n_val,
    )


def probe_all_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    random_state: int = 0,
    verbose: bool = True,
) -> list[LayerProbeResult]:
    """Run probe_single_layer at every layer.

    Args:
        activations: (n_examples, n_layers+1, hidden_dim).
                     Index 0 is the embedding layer; indices 1..n_layers are
                     transformer blocks. We probe all of them.
        labels: (n_examples,) in {0, 1}.
        n_folds, C, random_state: passed through to probe_single_layer.
        verbose: print progress per layer.

    Returns:
        list of LayerProbeResult, one per layer (in layer-index order).
    """
    n_examples, n_layers_plus_one, hidden_dim = activations.shape
    assert labels.shape == (n_examples,), \
        f"labels shape {labels.shape} doesn't match n_examples={n_examples}"

    results: list[LayerProbeResult] = []
    for layer_idx in range(n_layers_plus_one):
        X = activations[:, layer_idx, :]
        res = probe_single_layer(
            X, labels,
            n_folds=n_folds, C=C, random_state=random_state,
        )
        res.layer = layer_idx
        results.append(res)
        if verbose:
            print(
                f"  Layer {layer_idx:>3d}: "
                f"AUROC = {res.auroc_mean:.3f} ± {res.auroc_std:.3f}  "
                f"acc = {res.accuracy_mean:.3f} ± {res.accuracy_std:.3f}"
            )
    return results


def find_reference_layer(results: list[LayerProbeResult]) -> int:
    """Return the layer index with the highest mean AUROC.

    Per the PAP: 'The layer of peak accuracy becomes the reference layer for
    all downstream analyses.'
    """
    aurocs = [r.auroc_mean for r in results]
    return int(np.argmax(aurocs))