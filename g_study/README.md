# Probe G-study analysis (Phases 4–7)

Drop-in analysis pipeline that reads `outputs/scored_responses.csv` (the artifact
from `probe_pipeline_v2.py`) and produces the G-study, consistency check, validity
checks, and synthesis deliverables. **No GPU, no re-running generation.** Pure
numpy/scipy/sklearn/pandas/matplotlib — no `statsmodels`/R needed.

## Install / requirements
```
pip install numpy pandas scipy scikit-learn matplotlib
```
(You already have these if `probe_pipeline_v2.py` ran. `tabulate` is optional —
only used to pretty-print the Markdown table; it degrades gracefully without it.)

## Layout expected
```
<your project>/
  outputs/
    scored_responses.csv      <- input (must exist)
    analysis/                 <- everything below is WRITTEN here
  phase4_gstudy.py
  phase5_consistency.py
  phase6_validity.py
  phase7_synthesis.py
  gstudy_common.py
  run_all.py
```
Put the six `.py` files anywhere; they auto-locate `outputs/` by searching `./`,
`../`, and CWD. Override with `--outdir /path/to/outputs` on any script.

## Run everything
```
python run_all.py --boot 1000
```
Or one phase at a time (4 must run before 5/6/7, which read its JSON):
```
python phase4_gstudy.py --boot 1000 --seed 0
python phase5_consistency.py
python phase6_validity.py --topk 5
python phase7_synthesis.py
```

## What each phase does

**Phase 4 — G-study (`phase4_gstudy.py`).**
Variance decomposition of `probe_score` over the random-effects design
`(1|domain) + (1|elicitation) + (1|domain:elicitation) + p:de`, via the
Henderson/EMS estimator (the classic G-theory method; closed-form, no REML to
converge). Bootstrap CIs (B=1000, resampling responses within cells). Computes
G_nested, G_d, G_e.
→ `variance_components.csv`, `g_coefficients.csv`, `gstudy_results.json`,
`gstudy_report.txt`, `fig1_variance_components.png`, `fig2_g_coefficients.png`.

> **Note on the plan's formula.** The plan wrote
> `lmer(probe_score ~ domain*elicitation + (1|cell))`. That isn't identifiable —
> `cell` *is* `domain:elicitation`, so a random cell intercept and the fixed
> interaction model the same variance. G-theory wants the facets **random** (the
> point is to generalize to unsampled domains/elicitations), so the implemented
> model is the pure variance-components one above, which is exactly what your
> `G_nested = σ²_p:de / (σ²_p:de + σ²_d/n_d + σ²_e/n_e + σ²_de/(n_d·n_e))`
> formula assumes. This is stated in `gstudy_report.txt` too.

**Phase 5 — consistency check (`phase5_consistency.py`).**
Triangulates the decomposition with a shift-transfer experiment.
- *Default (CSV-only):* fits a decision **threshold** per cell on
  `probe_score → label`, applies it to the other eight cells, measures
  balanced-accuracy drop, bucketed into pure-domain (X) / pure-elicitation (Y) /
  joint (Z) shift. If `σ²_e > σ²_d` you expect `Y > X`; the script reports
  CONSISTENT/MISMATCH automatically.
  *(Why not AUROC here: AUROC of the single frozen score in an eval cell is the
  same no matter which cell you "trained" on, so AUROC-transfer is vacuous for a
  1-D frozen score. Threshold transfer is the honest CSV-computable analogue.)*
- *Literal Experiment 2 (`--activations FILE`):* if you re-export per-response
  activations (`.npz` with arrays `response_id`, `act` `[N,D]`, or a
  `.parquet/.csv` with `act_*` columns), it trains a fresh L2-logistic probe per
  cell and reports true cross-cell **AUROC** degradation.
→ `consistency_auroc_matrix.csv`, `consistency_shift_summary.csv`,
`consistency_shift_records.csv`, `consistency_results.json`,
`fig3_auroc_transfer_matrix.png`, `fig3b_shift_degradation.png`.

**Phase 6 — validity checks (`phase6_validity.py`).**
- *Check 1:* within-cell Pearson r between `probe_score` and label (+ Fisher CIs).
- *Check 2:* label-conditioned shift — within HONEST responses, one-way ANOVA of
  `probe_score` across elicitation and across domain. Significant elicitation
  effect with truth held constant = "context-meter" evidence (the Apollo App. A.1
  finding, generalized). Same computed for the deceptive subset as a symmetry check.
- *Check 3:* top-k highest-probe HONEST responses per elicitation, exported with
  automatic flags (length, hedging, refusal/evasion, moralizing) to seed manual
  transcript annotation.
→ `validity_within_cell_corr.csv`, `validity_label_conditioned.csv`,
`validity_discordant_examples.csv`, `validity_results.json`,
`fig4_within_cell_corr.png`, `fig5_label_conditioned_means.png`,
`fig6_discordance_scatter.png`.

**Phase 7 — synthesis (`phase7_synthesis.py`).**
Assembles the paper-ready table (variance components abs + %, G-coeffs, all with
CIs), an auto-drafted results paragraph, and a 2×2 summary panel.
→ `main_results_table.csv`, `main_results.md`, `fig0_summary_panel.png`.

## n ≈ 100 caveat
With ~11/cell instead of ~67, bootstrap CIs on individual variance components are
wide and small components may bootstrap negative (truncated to 0, per convention,
and noted). The scripts surface this explicitly. Report the CIs; lean on the big
qualitative result (which facet dominates, is the honest-only shift significant)
rather than precise point estimates of tiny components.

## Column contract
Reads `response_id, domain, elicitation, probe_score, label` (required) and uses
`cell, grade, description, prompt_text, response` if present. `response` enables
Check-3 transcript excerpts; without it you still get the flagged IDs.
