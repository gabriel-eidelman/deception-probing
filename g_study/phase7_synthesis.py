"""
phase7_synthesis.py
===================
Phase 7 -- Synthesis. Reads the JSON artifacts produced by phases 4-6 and
assembles the paper-ready deliverables:

    main_results_table.csv     one tidy table: variance components (abs + %),
                               G-coefficients, all with 95% CIs
    main_results.md            the same as a Markdown table you can paste into
                               the writeup, plus an auto-drafted results paragraph
    fig0_summary_panel.png     a single 2x2 panel combining the headline figures
                               (variance comp, G-coeffs, label-conditioned means,
                               within-cell r) for a poster/appendix at-a-glance

Run AFTER phase4/5/6:
    python phase7_synthesis.py

If a phase's JSON is missing, that section is skipped with a warning so you can
still get partial synthesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gstudy_common import apply_style, ensure_dir, resolve_outputs, PALETTE


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def build_table(g):
    rows = []
    vc = g["variance_components"]
    name_map = {"var_d": "Domain (d)", "var_e": "Elicitation (e)",
                "var_de": "Domain x Elicitation (de)",
                "var_resid_p_de": "Within-cell / residual (p:de)",
                "total": "Total"}
    pct = g["pct_of_total"]
    pct_map = {"var_d": "domain (d)", "var_e": "elicitation (e)",
               "var_de": "domain x elicitation (de)",
               "var_resid_p_de": "response within cell (p:de, residual)",
               "total": "TOTAL"}
    for key in ["var_d", "var_e", "var_de", "var_resid_p_de", "total"]:
        est = vc[key]["est"]
        lo, hi = vc[key]["ci"]
        rows.append({
            "quantity": name_map[key],
            "type": "variance component",
            "estimate": round(est, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "pct_of_total": round(pct.get(pct_map[key], float("nan")), 1),
        })
    for key in ["G_nested", "G_d", "G_e"]:
        gc = g["g_coefficients"][key]
        lo, hi = gc["ci"]
        rows.append({
            "quantity": key, "type": "G-coefficient",
            "estimate": round(gc["est"], 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "pct_of_total": "",
        })
    return pd.DataFrame(rows)


def draft_paragraph(g, v):
    vc = g["variance_components"]
    pct = g["pct_of_total"]
    gn = g["g_coefficients"]["G_nested"]
    e_pct = pct.get("elicitation (e)", float("nan"))
    d_pct = pct.get("domain (d)", float("nan"))
    resid_pct = pct.get("response within cell (p:de, residual)", float("nan"))
    n = g["design"]["n_total"]
    parts = [
        f"Across {n} on-policy responses spanning a "
        f"{g['design']['n_domains']}x{g['design']['n_elicitations']} "
        f"domain-by-elicitation design, the generalizability coefficient for the "
        f"frozen Instructed-Pairs probe was G\u0302 = {gn['est']:.2f} "
        f"(95% CI [{gn['ci'][0]:.2f}, {gn['ci'][1]:.2f}]), "
        f"indicating {'poor' if gn['est'] < 0.7 else 'moderate'} reliability as a "
        f"context-invariant deception measure. "
        f"Elicitation strategy accounted for {e_pct:.0f}% of total probe-score "
        f"variance versus {d_pct:.0f}% for task domain, "
        f"with {resid_pct:.0f}% within cells. "
    ]
    if v:
        c2 = v["check2_label_conditioned"]
        p = c2["honest_elicitation_anova_p"]
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            sig = "significantly" if p < 0.05 else "not significantly"
            parts.append(
                f"Restricting to responses graded honest, mean probe score still "
                f"varied {sig} across elicitation strategies "
                f"(one-way ANOVA p = {p:.3g}), evidence that the probe partly "
                f"tracks elicitation context rather than deceptiveness per se. ")
        r = v["check1_within_cell"]["mean_pearson_r"]
        parts.append(
            f"Mean within-cell correlation between probe score and grader label "
            f"was r = {r:.2f}.")
    return "".join(parts)


def summary_panel(out, g, v, path):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    apply_style()
    ana = out / "analysis"
    imgs = [
        ("fig1_variance_components.png", "Variance decomposition"),
        ("fig2_g_coefficients.png", "Generalizability coefficients"),
        ("fig5_label_conditioned_means.png", "Probe on honest responses"),
        ("fig4_within_cell_corr.png", "Within-cell probe-label r"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (fn, title) in zip(axes.ravel(), imgs):
        fp = ana / fn
        if fp.exists():
            ax.imshow(mpimg.imread(fp))
        else:
            ax.text(0.5, 0.5, f"missing:\n{fn}", ha="center", va="center")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Probe G-study - summary panel", fontsize=15,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    out = resolve_outputs(args.outdir)
    ana = ensure_dir(out / "analysis")

    g = _load(ana / "gstudy_results.json")
    v = _load(ana / "validity_results.json")
    c = _load(ana / "consistency_results.json")

    if g is None:
        raise SystemExit("Missing gstudy_results.json -- run phase4 first.")

    table = build_table(g)
    table.to_csv(ana / "main_results_table.csv", index=False)

    # markdown
    md = ["# Probe G-study -- main results", "",
          "## Variance components & generalizability", "",
          table.to_markdown(index=False), "", "## Drafted results paragraph", "",
          draft_paragraph(g, v)]
    if c:
        co = c["coherence"]
        md += ["", "## Phase 5 coherence check", "",
               f"- Elicitation variance > domain variance: "
               f"**{co['elicitation_variance_gt_domain']}**",
               f"- Elicitation AUROC drop > domain drop: "
               f"**{co['elicitation_drop_gt_domain']}**",
               f"- Decomposition & transfer consistent: **{co['coherent']}**"]
    if v is None:
        md += ["", "_validity_results.json missing -- run phase6 for Check 1-3._"]
    (ana / "main_results.md").write_text("\n".join(str(x) for x in md) + "\n")

    summary_panel(out, g, v, ana / "fig0_summary_panel.png")

    print("[phase7] wrote main_results_table.csv, main_results.md, "
          "fig0_summary_panel.png")
    print("\n----- drafted paragraph -----\n")
    print(draft_paragraph(g, v))
    print(f"\n[phase7] all synthesis artifacts in {ana}/")


if __name__ == "__main__":
    main()
