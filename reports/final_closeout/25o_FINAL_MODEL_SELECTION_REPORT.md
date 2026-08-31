# Final Model Selection — Delivery Risk Intelligence

Generated UTC: 2026-08-31T05:09:49.814464+00:00

## Decision

**Recommended final model: GN_EQ0**

Benchmark:

**LOGIT_MLE**

Prequential ensemble:

**REJECTED**

---

## Temporal Robustness

Evaluated OOT months:

**14**

Mean AP difference:

`GN_EQ0 - LOGIT_MLE = 0.002410790747`

Median AP difference:

`0.003286028047`

Months with higher GN_EQ0 AP:

**8/14**

Months with higher LOGIT_MLE AP:

**6/14**

Wilcoxon p-value:

`0.267578125000`

Sign-test p-value:

`0.790527343750`

Leave-one-month-out AP mean range:

`[0.001479602285, 0.003550722563]`

Every leave-one-month-out mean remained positive:

**True**

Temporal classification:

**FAVORABLE_AND_LOMO_ROBUST_NOT_DECISIVE**

---

## Ensemble Audit

Classification:

**ENSEMBLE_REJECTED_KEEP_BASE_MODEL**

AP gain relative to best base model:

`-0.000180104849`

Brier gap relative to best base model:

`0.000111739642`

LogLoss gap relative to best base model:

`0.002191060272`

Guardrails passed:

**False**

Final ensemble decision:

**Do not retain the ensemble.**

---

## Final Modeling Interpretation

GN_EQ0 is retained as the final recommended model for this case.

Its Average Precision was higher on average than LOGIT_MLE, and the
leave-one-month-out analysis showed that the positive mean AP difference did
not depend on one isolated month.

However, paired temporal inference was not statistically decisive.

Therefore the project does not claim universal superiority of GN_EQ0 over
LOGIT_MLE.

The prequential blend was also tested and rejected because it failed to improve
the best base model under the frozen performance guardrails.

The final case therefore favors the simpler final model choice:

**GN_EQ0**

while retaining:

**LOGIT_MLE as the principal benchmark.**

---

## Claim Boundaries

- No causal claim.
- No universal model-superiority claim.
- No statistically decisive temporal-superiority claim.
- No additional model family is required.
- The ensemble is not retained.
- Modeling experimentation is closed.
