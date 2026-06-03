# Probe G-study -- main results

## Variance components & generalizability

| quantity                      | type               |   estimate |   ci_lo |   ci_hi |   pct_of_total |
|:------------------------------|:-------------------|-----------:|--------:|--------:|---------------:|
| Domain (d)                    | variance component |     0      |  0      |  0.0018 |            0   |
| Elicitation (e)               | variance component |     0.0026 |  0      |  0.0069 |           17.2 |
| Domain x Elicitation (de)     | variance component |     0.0019 |  0      |  0.0085 |           12.2 |
| Within-cell / residual (p:de) | variance component |     0.0108 |  0.0072 |  0.0122 |           70.6 |
| Total                         | variance component |     0.0153 |  0.0114 |  0.02   |          100   |
| G_nested                      | G-coefficient      |     0.909  |  0.771  |  0.956  |                |
| G_d                           | G-coefficient      |     0.946  |  0.759  |  0.991  |                |
| G_e                           | G-coefficient      |     0.878  |  0.729  |  0.922  |                |

## Drafted results paragraph

Across 90 on-policy responses spanning a 3x3 domain-by-elicitation design, the generalizability coefficient for the frozen Instructed-Pairs probe was Ĝ = 0.91 (95% CI [0.77, 0.96]), indicating moderate reliability as a context-invariant deception measure. Elicitation strategy accounted for 17% of total probe-score variance versus 0% for task domain, with 71% within cells. Restricting to responses graded honest, mean probe score still varied significantly across elicitation strategies (one-way ANOVA p = 0.000547), evidence that the probe partly tracks elicitation context rather than deceptiveness per se. Mean within-cell correlation between probe score and grader label was r = 0.77.

## Phase 5 coherence check

- Elicitation variance > domain variance: **True**
- Elicitation AUROC drop > domain drop: **True**
- Decomposition & transfer consistent: **True**
