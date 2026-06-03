# Probe G-study -- main results

## Variance components & generalizability

| quantity                      | type               |   estimate |   ci_lo |   ci_hi |   pct_of_total |
|:------------------------------|:-------------------|-----------:|--------:|--------:|---------------:|
| Domain (d)                    | variance component |     0      |  0      |  0      |            0   |
| Elicitation (e)               | variance component |     0.0019 |  0      |  0.0043 |           13.3 |
| Domain x Elicitation (de)     | variance component |     0.0022 |  0.0007 |  0.006  |           14.8 |
| Within-cell / residual (p:de) | variance component |     0.0105 |  0.0083 |  0.0119 |           71.9 |
| Total                         | variance component |     0.0146 |  0.0117 |  0.0178 |          100   |
| G_nested                      | G-coefficient      |     0.922  |  0.851  |  0.961  |                |
| G_d                           | G-coefficient      |     0.936  |  0.833  |  0.977  |                |
| G_e                           | G-coefficient      |     0.885  |  0.798  |  0.926  |                |

## Drafted results paragraph

Across 180 on-policy responses spanning a 3x3 domain-by-elicitation design, the generalizability coefficient for the frozen Instructed-Pairs probe was Ĝ = 0.92 (95% CI [0.85, 0.96]), indicating moderate reliability as a context-invariant deception measure. Elicitation strategy accounted for 13% of total probe-score variance versus 0% for task domain, with 72% within cells. Restricting to responses graded honest, mean probe score still varied significantly across elicitation strategies (one-way ANOVA p = 2.93e-06), evidence that the probe partly tracks elicitation context rather than deceptiveness per se. Mean within-cell correlation between probe score and grader label was r = 0.79.

## Phase 5 coherence check

- Elicitation variance > domain variance: **True**
- Elicitation AUROC drop > domain drop: **True**
- Decomposition & transfer consistent: **True**
