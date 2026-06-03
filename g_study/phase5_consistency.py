"""
phase5_consistency.py
=====================
Phase 5 -- Consistency / triangulation check on the variance decomposition.

The plan's Experiment 2 trains nine per-cell probes ON ACTIVATIONS and measures
AUROC degradation under shift. Your scored_responses.csv stores the scalar
probe_score, NOT the ~4096-dim activation vectors, so a faithful retrain needs
the GPU step again. This script supports BOTH paths:

  MODE A (default, CSV-only):  "shift transfer of the frozen probe".
      For each source cell, fit a single 1-D threshold/scaler on probe_score->label
      (logistic regression on the one frozen score), then measure AUROC when that
      same decision is applied to every other cell. AUROC itself is
      threshold-free, so concretely we compute, for every (train cell, eval cell)
      pair, the AUROC of probe_score vs label IN THE EVAL CELL, and read the
      degradation relative to the eval cell's own best-case. This is exactly the
      quantity that triangulates the G-study: it asks "does the frozen score
      separate honest/deceptive equally well across contexts?"

  MODE B (--activations FILE):  the plan's literal Experiment 2.
      If you re-emit activations to a .npz/.parquet with columns/arrays
      [response_id, domain, elicitation, label, act_0..act_{D-1}], this trains a
      fresh L2 logistic probe per cell on those activations and evaluates each on
      the other eight. Pass --activations to enable.

Either way it aggregates AUROC drop by shift type:
    pure domain shift  (same elicitation, different domain)
    pure elicitation shift (same domain, different elicitation)
    joint shift (both differ)
and compares the ordering to the G-study (expect: elicitation-dominant variance
=> elicitation shift degrades most).

Run:
    python phase5_consistency.py
    python phase5_consistency.py --activations outputs/activations.npz

Outputs (<outputs>/analysis/):
    consistency_auroc_matrix.csv     9x9 train-cell x eval-cell AUROC
    consistency_shift_summary.csv    mean AUROC drop by shift type (X/Y/Z)
    consistency_results.json
    fig3_auroc_transfer_matrix.png   heatmap of cross-cell AUROC
    fig3b_shift_degradation.png      bar chart X vs Y vs Z with G-study overlay
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from gstudy_common import (
    apply_style, ensure_dir, estimate_variance_components, load_scored,
    resolve_outputs, PALETTE,
)

DECEPTIVE = "deceptive"
HONEST = "honest"


def _binary(df):
    """Keep only honest/deceptive rows; y=1 for deceptive (probe targets it)."""
    d = df[df["label"].isin([HONEST, DECEPTIVE])].copy()
    d["y"] = (d["label"] == DECEPTIVE).astype(int)
    return d


def _safe_auroc(y, s):
    """AUROC, or NaN if a cell has only one class present."""
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


# ----------------------------------------------------------- MODE A: scores --
def transfer_matrix_scores(df):
    """Frozen-score transfer via THRESHOLD shift (AUROC-of-one-score can't show
    transfer degradation -- it's identical regardless of source cell).

    For each source cell we fit a decision threshold on probe_score->label
    (the point maximizing balanced accuracy in the source). We then apply that
    SAME threshold in every eval cell and record balanced accuracy. The drop
    relative to the eval cell's own optimal threshold is the context-transfer
    cost: if the honest/deceptive boundary sits at a different score in a
    different context, a threshold calibrated elsewhere misclassifies -- exactly
    the 'probe is a context meter' signature. Returns (matrix, cells) where the
    matrix holds balanced accuracy of src-threshold evaluated on dst.
    """
    cells = sorted(df["cell"].unique())

    def best_threshold(scores, y):
        cand = np.unique(scores)
        if len(cand) < 2:
            return float(np.mean(scores))
        mids = (cand[:-1] + cand[1:]) / 2
        best_t, best_ba = mids[0], -1
        for t in mids:
            pred = (scores > t).astype(int)
            ba = _balanced_acc(y, pred)
            if ba > best_ba:
                best_ba, best_t = ba, t
        return best_t

    thresholds = {}
    for c in cells:
        g = df[df["cell"] == c]
        if g["y"].nunique() == 2:
            thresholds[c] = best_threshold(g["probe_score"].values, g["y"].values)
        else:
            thresholds[c] = np.nan

    M = pd.DataFrame(index=cells, columns=cells, dtype=float)
    for src in cells:
        t = thresholds[src]
        for dst in cells:
            g = df[df["cell"] == dst]
            if g["y"].nunique() < 2 or np.isnan(t):
                M.loc[src, dst] = np.nan
            else:
                pred = (g["probe_score"].values > t).astype(int)
                M.loc[src, dst] = _balanced_acc(g["y"].values, pred)
    return M, cells


def _balanced_acc(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (1 - pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float(np.nanmean([tpr, tnr]))


# ------------------------------------------------- MODE B: real activations --
def load_activations(path, df_scores):
    """Load activations file and align to the scored frame by response_id.

    Accepts .npz (arrays: response_id, act [N,D], and optionally domain/
    elicitation/label) or .parquet/.csv with act_* columns. Falls back to
    domain/elicitation/label from scored_responses.csv via response_id join.
    """
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        rid = z["response_id"]
        X = z["act"]
        adf = pd.DataFrame({"response_id": rid})
        adf["__X_index"] = np.arange(len(rid))
    else:
        adf = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        act_cols = [c for c in adf.columns if c.startswith("act_")]
        if not act_cols:
            raise SystemExit(f"No act_* columns in {path}")
        X = adf[act_cols].to_numpy(dtype=float)
        adf = adf[["response_id"]].copy()
        adf["__X_index"] = np.arange(len(adf))
    meta = df_scores[["response_id", "domain", "elicitation", "cell", "label", "y"]]
    merged = adf.merge(meta, on="response_id", how="inner")
    Xal = X[merged["__X_index"].to_numpy()]
    return Xal, merged.reset_index(drop=True)


def auroc_matrix_activations(X, meta):
    """Train an L2 logistic probe per cell on activations; eval on all cells."""
    cells = sorted(meta["cell"].unique())
    M = pd.DataFrame(index=cells, columns=cells, dtype=float)
    idx_by_cell = {c: meta.index[meta["cell"] == c].to_numpy() for c in cells}
    for src in cells:
        si = idx_by_cell[src]
        ys = meta.loc[si, "y"].to_numpy()
        if len(np.unique(ys)) < 2:
            continue  # cannot train
        clf = LogisticRegression(C=0.1, penalty="l2", max_iter=2000)
        clf.fit(X[si], ys)
        for dst in cells:
            di = idx_by_cell[dst]
            yd = meta.loc[di, "y"].to_numpy()
            sd = clf.decision_function(X[di])
            M.loc[src, dst] = _safe_auroc(yd, sd)
    return M, cells


# ------------------------------------------------------- shift aggregation ---
def parse_cell(c):
    """cell == 'domain__elicitation' (or 'domain_elicitation' fallback)."""
    if "__" in c:
        d, e = c.split("__", 1)
    else:
        d, e = c.rsplit("_", 1)
    return d, e


def aggregate_shifts(M, cells):
    """Average off-diagonal AUROC drop relative to each EVAL cell's diagonal,
    bucketed by shift type. Drop = AUROC(eval on itself) - AUROC(src probe on eval).
    """
    parsed = {c: parse_cell(c) for c in cells}
    buckets = {"domain": [], "elicitation": [], "joint": []}
    records = []
    for src, dst in product(cells, cells):
        if src == dst:
            continue
        a_self = M.loc[dst, dst]
        a_cross = M.loc[src, dst]
        if pd.isna(a_self) or pd.isna(a_cross):
            continue
        ds, es = parsed[src]
        dd, ee = parsed[dst]
        same_d, same_e = (ds == dd), (es == ee)
        if same_e and not same_d:
            kind = "domain"
        elif same_d and not same_e:
            kind = "elicitation"
        else:
            kind = "joint"
        drop = a_self - a_cross
        buckets[kind].append(drop)
        records.append({"src": src, "dst": dst, "shift": kind,
                        "metric_self": a_self, "metric_cross": a_cross,
                        "metric_drop": drop})
    summary = {k: (float(np.mean(v)) if v else float("nan"),
                   float(np.std(v)) if v else float("nan"), len(v))
               for k, v in buckets.items()}
    return summary, pd.DataFrame(records)


def fig_matrix(M, cells, path, metric="AUROC"):
    import matplotlib.pyplot as plt
    apply_style()
    short = [c.replace("__", "\n") for c in cells]
    arr = M.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.05 * len(cells) + 2.5,
                                    0.95 * len(cells) + 2.0))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(short, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(short, fontsize=7.5)
    ax.set_xlabel("evaluated on cell")
    ax.set_ylabel("decision fit on cell")
    for i in range(len(cells)):
        for j in range(len(cells)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="#222" if 0.55 < v < 0.92 else "white")
    ax.set_title(f"Phase 5 - Cross-cell {metric} transfer\n(diagonal = within-cell)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_shift(summary, vc_pct, path, metric="AUROC"):
    import matplotlib.pyplot as plt
    apply_style()
    order = ["domain", "elicitation", "joint"]
    means = [summary[k][0] for k in order]
    stds = [summary[k][1] for k in order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1]})
    colors = [PALETTE["domain"], PALETTE["elicitation"], PALETTE["de"]]
    x = np.arange(3)
    ax1.bar(x, means, yerr=stds, color=colors, edgecolor="white", linewidth=1.2,
            capsize=4, error_kw={"elinewidth": 1.1, "ecolor": "#444"})
    ax1.set_xticks(x)
    ax1.set_xticklabels(["pure domain\nshift (X)", "pure elicit.\nshift (Y)",
                         "joint\nshift (Z)"])
    ax1.set_ylabel(f"mean {metric} drop")
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.set_title(f"{metric} degradation by shift type")
    for xi, m in zip(x, means):
        if not np.isnan(m):
            ax1.text(xi, m, f"{m:+.3f}", ha="center",
                     va="bottom" if m >= 0 else "top", fontsize=10,
                     fontweight="bold")

    # G-study overlay: facet variance shares for triangulation
    gnames = ["domain", "elicit.", "d x e"]
    gvals = [vc_pct.get("domain (d)", np.nan),
             vc_pct.get("elicitation (e)", np.nan),
             vc_pct.get("domain x elicitation (de)", np.nan)]
    ax2.bar(np.arange(3), gvals, color=colors, edgecolor="white", linewidth=1.2)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(gnames)
    ax2.set_ylabel("% of total variance (G-study)")
    ax2.set_title("G-study facet variance (for comparison)")
    for xi, v in zip(np.arange(3), gvals):
        if not np.isnan(v):
            ax2.text(xi, v, f"{v:.0f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    fig.suptitle("Phase 5 - Consistency check vs Phase 4 decomposition",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--activations", default=None,
                    help="optional .npz/.parquet/.csv of per-response activations "
                         "(enables the plan's literal Experiment 2)")
    args = ap.parse_args()

    out = resolve_outputs(args.outdir)
    ana = ensure_dir(out / "analysis")
    df = _binary(load_scored(out))

    mode = "activations" if args.activations else "scores"
    metric = "AUROC" if mode == "activations" else "balanced accuracy"
    print(f"[phase5] mode = {mode} (metric: {metric})")
    if mode == "activations":
        X, meta = load_activations(args.activations, df)
        print(f"[phase5] activations: {X.shape[0]} rows x {X.shape[1]} dims")
        M, cells = auroc_matrix_activations(X, meta)
    else:
        print("[phase5] no --activations given; using frozen-score THRESHOLD "
              "transfer (balanced accuracy), the non-vacuous quantity computable "
              "from scored_responses.csv. For the plan's literal AUROC-of-"
              "retrained-probes Experiment 2, re-export activations and pass "
              "--activations.")
        M, cells = transfer_matrix_scores(df)

    M.to_csv(ana / "consistency_auroc_matrix.csv")

    summary, records = aggregate_shifts(M, cells)
    records.to_csv(ana / "consistency_shift_records.csv", index=False)
    sdf = pd.DataFrame([
        {"shift_type": k, "mean_metric_drop": v[0], "std": v[1], "n_pairs": v[2]}
        for k, v in summary.items()
    ])
    sdf.to_csv(ana / "consistency_shift_summary.csv", index=False)

    # G-study comparison (recompute pct so this script is standalone)
    vc = estimate_variance_components(load_scored(out))
    vc_pct = {r["component"]: r["pct_of_total"] for r in vc.as_rows()}

    fig_matrix(M, cells, ana / "fig3_auroc_transfer_matrix.png", metric=metric)
    fig_shift(summary, vc_pct, ana / "fig3b_shift_degradation.png", metric=metric)

    # coherence verdict
    X_, Y_, Z_ = summary["domain"][0], summary["elicitation"][0], summary["joint"][0]
    elicit_dominant_var = vc.var_e > vc.var_d
    elicit_dominant_drop = (not np.isnan(Y_)) and (not np.isnan(X_)) and (Y_ > X_)
    coherent = elicit_dominant_var == elicit_dominant_drop

    payload = {
        "mode": mode,
        "cells": cells,
        "shift_summary": {k: {"mean_drop": v[0], "std": v[1], "n_pairs": v[2]}
                          for k, v in summary.items()},
        "gstudy_facet_pct": {"domain": vc_pct.get("domain (d)"),
                             "elicitation": vc_pct.get("elicitation (e)"),
                             "de": vc_pct.get("domain x elicitation (de)")},
        "coherence": {
            "elicitation_variance_gt_domain": bool(elicit_dominant_var),
            "elicitation_drop_gt_domain": bool(elicit_dominant_drop)
            if not (np.isnan(Y_) or np.isnan(X_)) else None,
            "coherent": bool(coherent),
        },
    }
    with open(ana / "consistency_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[phase5] mean {metric} drop  domain(X)={X_:+.3f}  "
          f"elicitation(Y)={Y_:+.3f}  joint(Z)={Z_:+.3f}")
    print(f"[phase5] G-study: var_e {'>' if elicit_dominant_var else '<='} var_d")
    verdict = ("CONSISTENT" if coherent else
               "MISMATCH (flag: drop pattern not driven by facet variance -- "
               "check label noise / cell-size effects)")
    print(f"[phase5] coherence with decomposition: {verdict}")
    print(f"[phase5] wrote results to {ana}/")


if __name__ == "__main__":
    main()
