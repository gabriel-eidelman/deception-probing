"""
phase4_gstudy.py
================
Phase 4 -- Generalizability-theory variance decomposition of probe scores.

Fits the random-effects design
    probe_score ~ (1|domain) + (1|elicitation) + (1|domain:elicitation) + p:de
via the EMS/Henderson estimator in gstudy_common, extracts the four variance
components, computes the three G-coefficients, and attaches 95% bootstrap CIs
(resampling RESPONSES WITHIN CELLS, B=1000, as specified in the plan).

Run:
    python phase4_gstudy.py                 # auto-finds ./outputs
    python phase4_gstudy.py --outdir PATH --boot 1000 --seed 0

Outputs (under <outputs>/analysis/):
    variance_components.csv     point est, 95% CI, pct-of-total for each component
    g_coefficients.csv          G_nested, G_d, G_e with 95% CIs
    gstudy_results.json         machine-readable everything (incl. design summary)
    gstudy_report.txt           human-readable summary you can paste into notes
    fig1_variance_components.png  stacked + labeled bar of variance sources
    fig2_g_coefficients.png       three G-coefficients with bootstrap error bars
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gstudy_common import (
    apply_style, design_summary, ensure_dir, estimate_variance_components,
    g_coefficients, load_scored, resolve_outputs, PALETTE,
)


def bootstrap(df: pd.DataFrame, B: int, seed: int):
    """Resample responses WITHIN each cell (stratified), refit each replicate.

    Returns dict of arrays: var_d, var_e, var_de, var_resid, total,
    G_nested, G_d, G_e  -- each length <= B (failed fits skipped).
    """
    rng = np.random.default_rng(seed)
    cells = {c: g.index.to_numpy() for c, g in df.groupby("cell")}
    keys = ["var_d", "var_e", "var_de", "var_resid", "total",
            "G_nested", "G_d", "G_e"]
    acc = {k: [] for k in keys}

    for b in range(B):
        idx = np.concatenate([
            rng.choice(ix, size=len(ix), replace=True) for ix in cells.values()
        ])
        rep = df.loc[idx]
        # A bootstrap replicate can drop a facet level only if a whole cell
        # vanished; stratified resampling preserves all cells, so design is safe.
        try:
            vc = estimate_variance_components(rep)
            gc = g_coefficients(vc)
        except Exception:
            continue
        acc["var_d"].append(vc.var_d)
        acc["var_e"].append(vc.var_e)
        acc["var_de"].append(vc.var_de)
        acc["var_resid"].append(vc.var_resid)
        acc["total"].append(vc.total)
        acc["G_nested"].append(gc["G_nested"])
        acc["G_d"].append(gc["G_d"])
        acc["G_e"].append(gc["G_e"])

    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


def ci(arr: np.ndarray, lo=2.5, hi=97.5):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def fig_variance_components(vc, boot, path):
    import matplotlib.pyplot as plt
    apply_style()
    rows = vc.as_rows()[:-1]  # drop TOTAL
    names = ["domain", "elicitation", "d x e", "residual (p:de)"]
    colors = [PALETTE["domain"], PALETTE["elicitation"],
              PALETTE["de"], PALETTE["resid"]]
    vals = [r["variance"] for r in rows]
    pcts = [r["pct_of_total"] for r in rows]
    keys = ["var_d", "var_e", "var_de", "var_resid"]
    errs = []
    for k, v in zip(keys, vals):
        lo, hi = ci(boot[k])
        errs.append([max(v - lo, 0), max(hi - v, 0)])
    errs = np.array(errs).T

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: grouped bars with bootstrap CIs
    x = np.arange(len(names))
    axA.bar(x, vals, color=colors, edgecolor="white", linewidth=1.2,
            yerr=errs, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#444"})
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=12, ha="right")
    axA.set_ylabel("variance (probe-score$^2$)")
    axA.set_title("Variance components (95% bootstrap CI)")
    for xi, v, p in zip(x, vals, pcts):
        axA.text(xi, v, f"{p:.0f}%", ha="center", va="bottom", fontsize=9.5,
                 fontweight="bold")

    # Right: stacked single bar = composition of total variance
    bottom = 0.0
    for n, v, c in zip(names, vals, colors):
        axB.bar(0, v, bottom=bottom, color=c, edgecolor="white",
                linewidth=1.2, label=n, width=0.6)
        if v / max(sum(vals), 1e-9) > 0.04:
            axB.text(0, bottom + v / 2, n, ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")
        bottom += v
    axB.set_xlim(-0.6, 0.6)
    axB.set_xticks([])
    axB.set_ylabel("cumulative variance")
    axB.set_title("Composition of total variance")
    axB.grid(False)

    fig.suptitle("Phase 4 - G-study variance decomposition of probe scores",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_g_coefficients(gc, boot, path):
    import matplotlib.pyplot as plt
    apply_style()
    labels = ["$\\hat{G}$ (both facets)", "$\\hat{G}_d$ (domain)",
              "$\\hat{G}_e$ (elicitation)"]
    keys = ["G_nested", "G_d", "G_e"]
    vals = [gc[k] for k in keys]
    errs = []
    for k, v in zip(keys, vals):
        lo, hi = ci(boot[k])
        errs.append([max(v - lo, 0), max(hi - v, 0)])
    errs = np.array(errs).T

    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=[PALETTE["de"], PALETTE["domain"],
                                  PALETTE["elicitation"]],
                  edgecolor="white", linewidth=1.2, yerr=errs, capsize=5,
                  error_kw={"elinewidth": 1.3, "ecolor": "#444"})
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    for y, lab in [(0.8, "0.80  conventional 'reliable'"),
                   (0.7, "0.70")]:
        ax.axhline(y, color="#888", ls="--", lw=0.9)
        ax.text(len(labels) - 0.5, y + 0.005, lab, ha="right", va="bottom",
                fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.0, max(vals) * 1.15))
    ax.set_ylabel("generalizability coefficient")
    ax.set_title("Phase 4 - Generalizability coefficients (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="dir containing scored_responses.csv")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = resolve_outputs(args.outdir)
    ana = ensure_dir(out / "analysis")
    df = load_scored(out)
    summ = design_summary(df)

    print(f"[phase4] {summ['n_total']} responses, "
          f"{summ['n_domains']}x{summ['n_elicitations']} design, "
          f"cell n in [{summ['min_cell_n']},{summ['max_cell_n']}] "
          f"({'balanced' if summ['balanced'] else 'unbalanced'})")
    if summ["n_cells_present"] < summ["n_cells_expected"]:
        print(f"  WARNING: only {summ['n_cells_present']}/"
              f"{summ['n_cells_expected']} cells present. "
              f"Variance decomposition assumes the full crossing.")

    vc = estimate_variance_components(df)
    gc = g_coefficients(vc)
    for n in vc.notes:
        print(f"  note: {n}")

    print(f"[phase4] bootstrapping ({args.boot} resamples, within-cell)...")
    boot = bootstrap(df, args.boot, args.seed)

    # ---- variance_components.csv ----
    vrows = []
    keymap = {"domain (d)": "var_d", "elicitation (e)": "var_e",
              "domain x elicitation (de)": "var_de",
              "response within cell (p:de, residual)": "var_resid",
              "TOTAL": "total"}
    for r in vc.as_rows():
        k = keymap[r["component"]]
        lo, hi = ci(boot[k])
        vrows.append({**r, "ci_lo": lo, "ci_hi": hi})
    vc_df = pd.DataFrame(vrows)[["component", "variance", "ci_lo", "ci_hi",
                                 "pct_of_total"]]
    vc_df.to_csv(ana / "variance_components.csv", index=False)

    # ---- g_coefficients.csv ----
    grows = []
    gdesc = {"G_nested": "generalize over domain AND elicitation",
             "G_d": "generalize over domain (elicitation fixed)",
             "G_e": "generalize over elicitation (domain fixed)"}
    for k in ["G_nested", "G_d", "G_e"]:
        lo, hi = ci(boot[k])
        grows.append({"coefficient": k, "description": gdesc[k],
                      "value": gc[k], "ci_lo": lo, "ci_hi": hi})
    g_df = pd.DataFrame(grows)
    g_df.to_csv(ana / "g_coefficients.csv", index=False)

    # ---- figures ----
    fig_variance_components(vc, boot, ana / "fig1_variance_components.png")
    fig_g_coefficients(gc, boot, ana / "fig2_g_coefficients.png")

    # ---- json ----
    payload = {
        "design": summ,
        "method": "Henderson Type-I EMS variance components; "
                  "random facets d, e; responses nested in d x e cells.",
        "n_bar_harmonic_cell_size": vc.n_bar,
        "grand_mean": vc.grand_mean,
        "variance_components": {
            "var_d": {"est": vc.var_d, "ci": ci(boot["var_d"])},
            "var_e": {"est": vc.var_e, "ci": ci(boot["var_e"])},
            "var_de": {"est": vc.var_de, "ci": ci(boot["var_de"])},
            "var_resid_p_de": {"est": vc.var_resid, "ci": ci(boot["var_resid"])},
            "total": {"est": vc.total, "ci": ci(boot["total"])},
        },
        "pct_of_total": {r["component"]: r["pct_of_total"] for r in vc.as_rows()},
        "g_coefficients": {k: {"est": gc[k], "ci": ci(boot[k])}
                           for k in ["G_nested", "G_d", "G_e"]},
        "bootstrap": {"B_requested": args.boot,
                      "B_effective": int(boot["total"].size), "seed": args.seed},
        "truncation_notes": vc.notes,
    }
    with open(ana / "gstudy_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ---- human-readable report ----
    pct = {r["component"]: r["pct_of_total"] for r in vc.as_rows()}
    dominant = max(
        [("domain", vc.var_d), ("elicitation", vc.var_e),
         ("interaction", vc.var_de), ("within-cell signal", vc.var_resid)],
        key=lambda t: t[1])[0]
    lines = [
        "=" * 70,
        "PHASE 4 - G-STUDY VARIANCE DECOMPOSITION",
        "=" * 70,
        f"Design: {summ['n_domains']} domains x {summ['n_elicitations']} "
        f"elicitations, {summ['n_total']} responses "
        f"(cell n {summ['min_cell_n']}-{summ['max_cell_n']}).",
        "",
        "Variance components (point est | % of total | 95% bootstrap CI):",
    ]
    for r in vrows[:-1]:
        lines.append(f"  {r['component']:<42} {r['variance']:>9.4f}  "
                     f"{r['pct_of_total']:>5.1f}%  "
                     f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    lines += [
        "",
        "Generalizability coefficients (95% bootstrap CI):",
        f"  G_nested (over d AND e) = {gc['G_nested']:.3f}  "
        f"[{ci(boot['G_nested'])[0]:.3f}, {ci(boot['G_nested'])[1]:.3f}]",
        f"  G_d      (over domain)  = {gc['G_d']:.3f}  "
        f"[{ci(boot['G_d'])[0]:.3f}, {ci(boot['G_d'])[1]:.3f}]",
        f"  G_e      (over elicit.) = {gc['G_e']:.3f}  "
        f"[{ci(boot['G_e'])[0]:.3f}, {ci(boot['G_e'])[1]:.3f}]",
        "",
        f"Largest variance source: {dominant}.",
        f"Elicitation share ({pct['elicitation (e)']:.1f}%) vs "
        f"domain share ({pct['domain (d)']:.1f}%): "
        f"{'elicitation dominates -> probe tracks HOW deception is elicited' if vc.var_e > vc.var_d else 'domain >= elicitation'}.",
        "",
        "Reading: within-cell (p:de) variance is treated as universe-score",
        "(context-invariant) signal; facet variance is measurement error for the",
        "claim 'the probe measures deceptiveness independent of context'. Low G",
        "=> probe score is driven by which context it saw, not a stable signal.",
    ]
    if vc.notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in vc.notes]
    if summ["n_total"] < 200:
        lines += ["", f"CAUTION: n={summ['n_total']} (reduced from planned ~600). "
                  "CIs are wide; report them and avoid over-interpreting point "
                  "estimates of small components."]
    report = "\n".join(lines)
    (ana / "gstudy_report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\n[phase4] wrote results to {ana}/")


if __name__ == "__main__":
    main()
