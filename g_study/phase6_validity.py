"""
phase6_validity.py
==================
Phase 6 -- Construct-validity checks. These use the LABELS, which sat on the
sideline during the G-study, to interrogate WHAT the probe score reflects.

Check 1  Within-cell probe-vs-label correlation.
    For each cell, Pearson r between probe_score and label (deceptive=1).
    High within-cell r => the within-cell (universe-score) variance the G-study
    credited as 'signal' really does track the labeled construct. Low/noisy r
    => the probe varies within cell for reasons unrelated to deception.

Check 2  Label-conditioned shift analysis (the Apollo App. A.1 finding,
    generalized). Subset to HONEST responses only. If mean probe_score still
    shifts across elicitations (or domains) when truth is held constant, the
    probe is firing on CONTEXT, not deception. Reports per-group means + a
    one-way ANOVA F-test/p-value for elicitation and for domain, within the
    honest subset (and, as a symmetry check, within the deceptive subset).

Check 3  Qualitative discordance. The honest responses with the HIGHEST probe
    scores per elicitation (top-k). These are the transcripts to read to ground
    'the probe is a context meter'. Exports them with simple automatic feature
    flags (length, hedging/uncertainty markers, refusal/evasion, moralizing) to
    seed your manual annotation.

Run:
    python phase6_validity.py --topk 5

Outputs (<outputs>/analysis/):
    validity_within_cell_corr.csv      per-cell Pearson r, p, n
    validity_label_conditioned.csv     honest/deceptive group means by facet + F,p
    validity_discordant_examples.csv   top-k high-probe honest responses + flags
    validity_results.json
    fig4_within_cell_corr.png          per-cell r (with CIs) heat/bar
    fig5_label_conditioned_means.png   honest-only probe means by elicitation/domain
    fig6_discordance_scatter.png       probe_score vs label, honest high-scorers flagged
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gstudy_common import (
    apply_style, ensure_dir, load_scored, resolve_outputs, PALETTE,
)

HONEST, DECEPTIVE = "honest", "deceptive"


def _binary(df):
    d = df[df["label"].isin([HONEST, DECEPTIVE])].copy()
    d["y"] = (d["label"] == DECEPTIVE).astype(int)
    return d


# ----------------------------------------------------------- Check 1 ---------
def within_cell_corr(df):
    rows = []
    for cell, g in df.groupby("cell"):
        n = len(g)
        if n >= 3 and g["y"].nunique() == 2:
            r, p = stats.pearsonr(g["probe_score"], g["y"])
            # Fisher CI
            z = np.arctanh(r)
            se = 1.0 / np.sqrt(n - 3)
            lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        else:
            r = p = lo = hi = np.nan
        dom, eli = (cell.split("__", 1) + [""])[:2]
        rows.append({"cell": cell, "domain": dom, "elicitation": eli,
                     "pearson_r": r, "p_value": p, "ci_lo": lo, "ci_hi": hi,
                     "n": n, "n_honest": int((g["y"] == 0).sum()),
                     "n_deceptive": int((g["y"] == 1).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- Check 2 ---------
def label_conditioned(df):
    """Within each label subset, test whether probe_score depends on facet."""
    out = []
    for label_name in [HONEST, DECEPTIVE]:
        sub = df[df["label"] == label_name]
        for facet in ["elicitation", "domain"]:
            groups = [g["probe_score"].values for _, g in sub.groupby(facet)
                      if len(g) >= 2]
            means = sub.groupby(facet)["probe_score"].agg(["mean", "std", "count"])
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                F, p = stats.f_oneway(*groups)
            else:
                F, p = np.nan, np.nan
            for level, row in means.iterrows():
                out.append({"label_subset": label_name, "facet": facet,
                            "level": level, "mean_probe": row["mean"],
                            "std_probe": row["std"], "n": int(row["count"]),
                            "anova_F": F, "anova_p": p,
                            "facet_range": np.nan})
    res = pd.DataFrame(out)
    # add per (label,facet) range of group means as effect-size-ish summary
    for (lab, fac), g in res.groupby(["label_subset", "facet"]):
        rng = g["mean_probe"].max() - g["mean_probe"].min()
        res.loc[(res.label_subset == lab) & (res.facet == fac), "facet_range"] = rng
    return res


# ----------------------------------------------------------- Check 3 ---------
HEDGE = re.compile(r"\b(maybe|perhaps|might|possibly|i think|i'?m not sure|"
                   r"unsure|uncertain|could be|seems|apparently|i believe)\b", re.I)
REFUSAL = re.compile(r"\b(i can'?t|i cannot|i'?m unable|i won'?t|as an ai|"
                     r"i'?m not able|i do not have|i don'?t have access)\b", re.I)
MORAL = re.compile(r"\b(should|ought|right thing|honest|integrity|ethic|"
                   r"responsib|appropriate|important to)\b", re.I)
EVASION = re.compile(r"\b(it depends|hard to say|complicated|various factors|"
                     r"not straightforward|context)\b", re.I)


def flag_text(t):
    t = str(t) if pd.notna(t) else ""
    words = len(t.split())
    return {
        "n_words": words,
        "hedging": bool(HEDGE.search(t)),
        "refusal_evasion": bool(REFUSAL.search(t) or EVASION.search(t)),
        "moralizing": bool(MORAL.search(t)),
    }


def discordant_examples(df, topk):
    text_col = "response" if "response" in df.columns else None
    rows = []
    honest = df[df["label"] == HONEST]
    for eli, g in honest.groupby("elicitation"):
        top = g.nlargest(topk, "probe_score")
        for _, r in top.iterrows():
            rec = {"elicitation": eli, "domain": r["domain"], "cell": r["cell"],
                   "response_id": r.get("response_id"),
                   "probe_score": r["probe_score"], "grade": r.get("grade")}
            if text_col:
                txt = r[text_col]
                rec["response_excerpt"] = (str(txt)[:300] + "...") \
                    if pd.notna(txt) and len(str(txt)) > 300 else str(txt)
                rec.update(flag_text(txt))
            if "prompt_text" in df.columns:
                pt = r["prompt_text"]
                rec["prompt_excerpt"] = (str(pt)[:200] + "...") \
                    if pd.notna(pt) and len(str(pt)) > 200 else str(pt)
            rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------- figures ---------
def fig_within_cell(corr, path):
    import matplotlib.pyplot as plt
    apply_style()
    c = corr.dropna(subset=["pearson_r"]).sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.5 * len(c) + 1.5)))
    y = np.arange(len(c))
    err = np.array([c["pearson_r"] - c["ci_lo"], c["ci_hi"] - c["pearson_r"]])
    colors = [PALETTE["honest"] if r > 0 else PALETTE["deceptive"]
              for r in c["pearson_r"]]
    ax.barh(y, c["pearson_r"], xerr=np.abs(err), color=colors, edgecolor="white",
            linewidth=1.0, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#555"})
    ax.set_yticks(y)
    ax.set_yticklabels(c["cell"], fontsize=8)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_xlabel("Pearson r (probe_score vs deceptive=1)")
    ax.set_title("Phase 6 / Check 1 - Within-cell probe-label correlation\n"
                 "(high = within-cell variance tracks the construct)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_label_conditioned(lc, path):
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, facet in zip(axes, ["elicitation", "domain"]):
        sub = lc[(lc["label_subset"] == HONEST) & (lc["facet"] == facet)]
        sub = sub.sort_values("level")
        x = np.arange(len(sub))
        yerr = sub["std_probe"].fillna(0) / np.sqrt(sub["n"].clip(lower=1))
        ax.bar(x, sub["mean_probe"], yerr=yerr, color=PALETTE["honest"],
               edgecolor="white", linewidth=1.1, capsize=4,
               error_kw={"elinewidth": 1.0, "ecolor": "#444"}, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["level"], rotation=15, ha="right", fontsize=9)
        ax.axhline(0, color="#888", lw=0.8, ls="--")
        ax.set_ylabel("mean probe score (HONEST only)")
        F = sub["anova_F"].iloc[0] if len(sub) else np.nan
        p = sub["anova_p"].iloc[0] if len(sub) else np.nan
        ax.set_title(f"by {facet}\nANOVA F={F:.2f}, p={p:.3g}" if not np.isnan(F)
                     else f"by {facet}")
    fig.suptitle("Phase 6 / Check 2 - Probe score on HONEST responses, by context\n"
                 "(shift here = probe fires on context with truth held constant)",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_discordance(df, disc, path):
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    jitter = (np.random.RandomState(0).rand(len(df)) - 0.5) * 0.18
    for lab, col in [(HONEST, PALETTE["honest"]), (DECEPTIVE, PALETTE["deceptive"])]:
        m = df["label"] == lab
        yv = (df.loc[m, "label"] == DECEPTIVE).astype(int) + jitter[m.values]
        ax.scatter(df.loc[m, "probe_score"], yv, s=20, alpha=0.55, color=col,
                   edgecolor="white", linewidth=0.3, label=lab)
    if len(disc):
        ax.scatter(disc["probe_score"], np.zeros(len(disc)) + 0.0,
                   s=90, facecolor="none", edgecolor="#d00000", linewidth=1.6,
                   zorder=5, label="flagged honest (top probe)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["honest", "deceptive"])
    ax.set_xlabel("probe score (log-odds)")
    ax.set_title("Phase 6 / Check 3 - Honest responses with high probe scores\n"
                 "(circled = read these transcripts)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    out = resolve_outputs(args.outdir)
    ana = ensure_dir(out / "analysis")
    df = _binary(load_scored(out))

    # Check 1
    corr = within_cell_corr(df)
    corr.to_csv(ana / "validity_within_cell_corr.csv", index=False)
    fig_within_cell(corr, ana / "fig4_within_cell_corr.png")

    # Check 2
    lc = label_conditioned(df)
    lc.to_csv(ana / "validity_label_conditioned.csv", index=False)
    fig_label_conditioned(lc, ana / "fig5_label_conditioned_means.png")

    # Check 3
    disc = discordant_examples(df, args.topk)
    disc.to_csv(ana / "validity_discordant_examples.csv", index=False)
    fig_discordance(df, disc, ana / "fig6_discordance_scatter.png")

    # summary numbers for json + console
    mean_r = float(np.nanmean(corr["pearson_r"]))
    honest_eli = lc[(lc.label_subset == HONEST) & (lc.facet == "elicitation")]
    honest_dom = lc[(lc.label_subset == HONEST) & (lc.facet == "domain")]
    eli_p = float(honest_eli["anova_p"].iloc[0]) if len(honest_eli) else float("nan")
    dom_p = float(honest_dom["anova_p"].iloc[0]) if len(honest_dom) else float("nan")
    eli_range = float(honest_eli["facet_range"].iloc[0]) if len(honest_eli) else float("nan")
    dom_range = float(honest_dom["facet_range"].iloc[0]) if len(honest_dom) else float("nan")

    payload = {
        "check1_within_cell": {
            "mean_pearson_r": mean_r,
            "per_cell": corr.to_dict(orient="records"),
        },
        "check2_label_conditioned": {
            "honest_elicitation_anova_p": eli_p,
            "honest_elicitation_mean_range": eli_range,
            "honest_domain_anova_p": dom_p,
            "honest_domain_mean_range": dom_range,
            "interpretation": (
                "Significant elicitation effect within honest responses => probe "
                "tracks elicitation context even when truth is held constant "
                "(context-meter behavior)."),
        },
        "check3_discordant": {
            "topk": args.topk,
            "n_flagged": int(len(disc)),
            "has_text": "response" in df.columns,
        },
    }
    with open(ana / "validity_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[phase6] Check1 mean within-cell r = {mean_r:.3f}")
    print(f"[phase6] Check2 HONEST-only ANOVA: elicitation p={eli_p:.3g} "
          f"(mean range {eli_range:.3f}), domain p={dom_p:.3g} "
          f"(mean range {dom_range:.3f})")
    if not np.isnan(eli_p) and eli_p < 0.05:
        print("         -> probe score shifts across elicitation among HONEST "
              "responses: context-meter behavior supported.")
    print(f"[phase6] Check3 flagged {len(disc)} high-probe honest responses "
          f"({'with text' if 'response' in df.columns else 'NO text column'}).")
    if "response" not in df.columns:
        print("         note: scored_responses.csv has no `response` column; "
              "re-export with text to enable transcript reading.")
    print(f"[phase6] wrote results to {ana}/")


if __name__ == "__main__":
    main()
