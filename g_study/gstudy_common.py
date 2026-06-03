"""
gstudy_common.py
================
Shared helpers for the deception-probe G-study / validity analysis (Phases 4-7).

Reads outputs/scored_responses.csv (the artifact emitted by probe_pipeline_v2.py)
and provides:
  * load_scored()        -> tidy, validated DataFrame
  * estimate_variance_components()  -> EMS-based G-theory variance decomposition
                                       for the p:(d x e) random-effects design
  * g_coefficients()     -> G-hat, G-hat_d, G-hat_e from variance components
  * plotting style + a small palette

WHY EMS AND NOT lmer/REML
-------------------------
Classic generalizability theory estimates variance components from the expected
mean squares (EMS) of a random-effects ANOVA. For the design here -- two crossed
random facets (domain d, elicitation e) with responses (p) nested in each d x e
cell -- the EMS equations have a closed form and do NOT depend on iterative REML
convergence, which is fragile with only 9 cells. We use Henderson's Method I /
Type-I sums of squares, which is unbiased for random effects and handles the
mild cell-size imbalance from on-policy generation. This is the textbook G-theory
estimator (Brennan, 2001), so it is also the most defensible choice for the paper.

NOTE ON THE PLAN'S lmer FORMULA
-------------------------------
The plan writes  lmer(probe_score ~ domain*elicitation + (1|cell)).  That is not
identifiable: `cell` == domain:elicitation, so a random cell intercept and a fixed
domain:elicitation interaction model the SAME variation. G-theory treats the
facets as RANDOM (that is the whole point -- we want to generalize over unsampled
domains/elicitations), so the correct model is a pure variance-components model:
    probe_score ~ (1|domain) + (1|elicitation) + (1|domain:elicitation) + residual
with residual = p:de (response-within-cell). That is what is implemented below and
what the G_nested formula in the plan actually assumes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths ----
# Resolve outputs/ relative to this file's parent by default, but allow override.
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = HERE / "outputs"


def resolve_outputs(outdir: str | Path | None = None) -> Path:
    """Find the outputs directory containing scored_responses.csv.

    Search order: explicit arg -> ./outputs -> ../outputs -> CWD/outputs.
    """
    candidates = []
    if outdir is not None:
        candidates.append(Path(outdir))
    candidates += [
        DEFAULT_OUTPUTS,
        HERE.parent / "outputs",
        Path.cwd() / "outputs",
    ]
    for c in candidates:
        if (c / "scored_responses.csv").exists():
            return c
    # Fall back to the first candidate so error messages are sensible.
    return candidates[0]


# ------------------------------------------------------------------ load -----
REQUIRED_COLS = ["response_id", "domain", "elicitation", "probe_score", "label"]


def load_scored(outdir: str | Path | None = None) -> pd.DataFrame:
    """Load and validate scored_responses.csv into a tidy frame.

    Guarantees on the returned frame:
      * columns: response_id, domain, elicitation, cell, probe_score, label,
        plus grade/description/prompt_text/response if present.
      * probe_score is float; rows with non-finite scores are dropped (warned).
      * `cell` is the "domain__elicitation" string (rebuilt if missing).
      * label normalized to lowercase {honest, deceptive, ...}.
    """
    out = resolve_outputs(outdir)
    csv = out / "scored_responses.csv"
    if not csv.exists():
        sys.exit(
            f"ERROR: could not find {csv}\n"
            f"Pass the outputs dir explicitly, e.g. --outdir path/to/outputs"
        )
    df = pd.read_csv(csv)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {csv} is missing required columns: {missing}\n"
                 f"Found columns: {list(df.columns)}")

    df["probe_score"] = pd.to_numeric(df["probe_score"], errors="coerce")
    bad = df["probe_score"].isna().sum()
    if bad:
        print(f"  [load] dropping {bad} rows with non-numeric probe_score")
        df = df[df["probe_score"].notna()].copy()

    if "cell" not in df.columns:
        df["cell"] = df["domain"].astype(str) + "__" + df["elicitation"].astype(str)

    df["domain"] = df["domain"].astype(str)
    df["elicitation"] = df["elicitation"].astype(str)
    df["cell"] = df["cell"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df.reset_index(drop=True)
    return df


def design_summary(df: pd.DataFrame) -> dict:
    """Counts that the analysis needs and that belong in the methods section."""
    doms = sorted(df["domain"].unique())
    elis = sorted(df["elicitation"].unique())
    cell_n = df.groupby("cell").size()
    return {
        "n_total": int(len(df)),
        "domains": doms,
        "elicitations": elis,
        "n_domains": len(doms),
        "n_elicitations": len(elis),
        "n_cells_present": int(df["cell"].nunique()),
        "n_cells_expected": len(doms) * len(elis),
        "cell_sizes": {k: int(v) for k, v in cell_n.items()},
        "min_cell_n": int(cell_n.min()),
        "max_cell_n": int(cell_n.max()),
        "balanced": bool(cell_n.min() == cell_n.max()),
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
    }


# ------------------------------------------------ variance components (EMS) --
@dataclass
class VarComps:
    """G-theory variance components for the p:(d x e) random design.

    All values are in (probe-score)^2 units. `n_d`, `n_e` are the number of
    sampled levels of each facet; `n_bar` is the harmonic-mean cell size used in
    the EMS coefficients for unbalanced data.
    """
    var_d: float
    var_e: float
    var_de: float
    var_resid: float          # p:de  (response within cell)
    n_d: int
    n_e: int
    n_bar: float
    grand_mean: float
    notes: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.var_d + self.var_e + self.var_de + self.var_resid

    def as_rows(self) -> list[dict]:
        comps = [
            ("domain (d)", self.var_d),
            ("elicitation (e)", self.var_e),
            ("domain x elicitation (de)", self.var_de),
            ("response within cell (p:de, residual)", self.var_resid),
        ]
        tot = self.total
        rows = []
        for name, v in comps:
            rows.append({
                "component": name,
                "variance": v,
                "pct_of_total": (100.0 * v / tot) if tot > 0 else float("nan"),
            })
        rows.append({"component": "TOTAL", "variance": tot, "pct_of_total": 100.0})
        return rows


def _harmonic_mean(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    return len(counts) / np.sum(1.0 / counts)


def estimate_variance_components(df: pd.DataFrame,
                                 score_col: str = "probe_score") -> VarComps:
    """Henderson Method-I (Type-I SS) variance-component estimator for
    probe_score ~ d (random) + e (random) + de (random) + p:de (residual).

    For balanced data this equals the classic G-theory EMS solution exactly;
    for the mild imbalance from on-policy generation it uses the harmonic-mean
    cell size n_bar in the EMS coefficients, which is the standard adjustment.
    Negative variance estimates are truncated to 0 (and noted), per convention.
    """
    d_levels = sorted(df["domain"].unique())
    e_levels = sorted(df["elicitation"].unique())
    n_d, n_e = len(d_levels), len(e_levels)

    grand = df[score_col].mean()
    cell_counts = df.groupby(["domain", "elicitation"]).size()
    n_bar = _harmonic_mean(cell_counts.values.astype(float))
    N = len(df)

    # --- Type-I sums of squares ---
    # SS_d: between domain means (weighted by n in each domain)
    dom_mean = df.groupby("domain")[score_col].mean()
    dom_n = df.groupby("domain").size()
    ss_d = float(np.sum(dom_n.values * (dom_mean.values - grand) ** 2))

    eli_mean = df.groupby("elicitation")[score_col].mean()
    eli_n = df.groupby("elicitation").size()
    ss_e = float(np.sum(eli_n.values * (eli_mean.values - grand) ** 2))

    cell_mean = df.groupby(["domain", "elicitation"])[score_col].mean()
    cell_n = df.groupby(["domain", "elicitation"]).size()
    # SS_cells (between all d x e cells), then de interaction = cells - d - e
    ss_cells = float(np.sum(cell_n.values * (cell_mean.values - grand) ** 2))
    ss_de = ss_cells - ss_d - ss_e

    # SS within cells (residual / p:de)
    merged = df.merge(cell_mean.rename("cell_mean"),
                      on=["domain", "elicitation"], how="left")
    ss_resid = float(np.sum((merged[score_col].values - merged["cell_mean"].values) ** 2))

    # --- degrees of freedom ---
    df_d = n_d - 1
    df_e = n_e - 1
    df_de = (n_d - 1) * (n_e - 1)
    df_resid = N - n_d * n_e
    df_resid = max(df_resid, 1)

    # --- mean squares ---
    ms_d = ss_d / df_d if df_d > 0 else 0.0
    ms_e = ss_e / df_e if df_e > 0 else 0.0
    ms_de = ss_de / df_de if df_de > 0 else 0.0
    ms_resid = ss_resid / df_resid

    # --- EMS solution (balanced-design coefficients with n_bar) ---
    # E[MS_resid] = var_resid
    # E[MS_de]    = var_resid + n_bar * var_de
    # E[MS_d]     = var_resid + n_bar * var_de + n_e * n_bar * var_d
    # E[MS_e]     = var_resid + n_bar * var_de + n_d * n_bar * var_e
    var_resid = ms_resid
    var_de = (ms_de - ms_resid) / n_bar
    var_d = (ms_d - ms_de) / (n_e * n_bar)
    var_e = (ms_e - ms_de) / (n_d * n_bar)

    notes = []
    fixed = {}
    for name, val in [("var_d", var_d), ("var_e", var_e),
                      ("var_de", var_de), ("var_resid", var_resid)]:
        if val < 0:
            notes.append(f"{name} estimated negative ({val:.4g}); truncated to 0 "
                         f"(standard for ANOVA variance components, indicates the "
                         f"true component is near zero / sampling noise).")
            fixed[name] = 0.0
        else:
            fixed[name] = float(val)

    return VarComps(
        var_d=fixed["var_d"], var_e=fixed["var_e"],
        var_de=fixed["var_de"], var_resid=fixed["var_resid"],
        n_d=n_d, n_e=n_e, n_bar=float(n_bar), grand_mean=float(grand),
        notes=notes,
    )


def g_coefficients(vc: VarComps) -> dict:
    """Generalizability coefficients from variance components.

    G_nested:  generalize over BOTH facets (the headline reliability of the
               probe as a deception measure across domains AND elicitations).
    G_d:       generalize over domain only (elicitation held fixed).
    G_e:       generalize over elicitation only (domain held fixed).

    Interpretation: the "universe score" variance here is the within-cell
    (p:de) variance -- the part of probe-score variation that is consistent
    once context (domain x elicitation) is fixed. Facet variance is measurement
    error from the standpoint of "does the probe measure deceptiveness
    independent of context". A LOW G means the probe's score is dominated by
    which context it was shown, not by a context-invariant signal.
    """
    sp = vc.var_resid          # signal: within-cell (universe-score) variance
    nd, ne = vc.n_d, vc.n_e

    def g(div_d, div_e, div_de):
        err = (vc.var_d / div_d if div_d else 0.0) \
            + (vc.var_e / div_e if div_e else 0.0) \
            + (vc.var_de / div_de if div_de else 0.0)
        denom = sp + err
        return float(sp / denom) if denom > 0 else float("nan")

    return {
        "G_nested": g(nd, ne, nd * ne),     # both facets random
        "G_d": g(nd, None, nd),             # domain random, elicitation fixed
        "G_e": g(None, ne, ne),             # elicitation random, domain fixed
    }


# ----------------------------------------------------------------- plotting --
PALETTE = {
    "honest": "#2a9d8f",
    "deceptive": "#e76f51",
    "ambiguous": "#e9c46a",
    "domain": "#264653",
    "elicitation": "#9b5de5",
    "de": "#577590",
    "resid": "#bcb8b1",
    "accent": "#e76f51",
    "grid": "#e6e6e6",
}


def apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
