#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25N — FINAL MODEL SELECTION FREEZE
Delivery Risk Intelligence

Consolida:
- robustez temporal GN_EQ0 vs LOGIT_MLE;
- teste prequential de ensemble;
- decisão final do modelo.

Não:
- retreina modelos;
- altera RAW;
- recalcula features;
- cria matriz N x N;
- abre nova família de modelos.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
OUT = ROOT / "reports" / "final_closeout"

TEMPORAL = (
    OUT
    / "25h_TEMPORAL_ROBUSTNESS_DECISION.json"
)

BLEND = (
    OUT
    / "25l_FINAL_BLEND_DECISION.json"
)

SUMMARY_BLEND = (
    OUT
    / "25k_prequential_blend_summary.csv"
)

ALPHA_HISTORY = (
    OUT
    / "25m_prequential_alpha_history.csv"
)

FINAL_JSON = (
    OUT
    / "25n_FINAL_MODEL_SELECTION.json"
)

FINAL_MD = (
    OUT
    / "25o_FINAL_MODEL_SELECTION_REPORT.md"
)


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório ausente: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Arquivo vazio: {path}"
        )


def load_json(path):
    require(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


print("=" * 108)
print("25N — FINAL MODEL SELECTION FREEZE")
print("=" * 108)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("NEW_MODEL_FAMILY = false")
print("N_X_N_MATRIX_CREATED = false")
print()


# ============================================================================================
# 1. INPUTS
# ============================================================================================

for path in [
    TEMPORAL,
    BLEND,
    SUMMARY_BLEND,
    ALPHA_HISTORY,
]:

    require(path)

    print(
        "[PASS] exists:",
        path.relative_to(ROOT)
    )


temporal = load_json(
    TEMPORAL
)

blend = load_json(
    BLEND
)


if temporal.get("status") != "PASS":

    raise RuntimeError(
        "25F temporal robustness não está PASS."
    )


if blend.get("status") != "PASS":

    raise RuntimeError(
        "25I blend não está PASS."
    )


# ============================================================================================
# 2. VALIDAR RESULTADO TEMPORAL
# ============================================================================================

if (
    temporal.get("comparison")
    !=
    "GN_EQ0_vs_LOGIT_MLE"
):

    raise RuntimeError(
        "Comparação temporal inesperada."
    )


if (
    temporal.get("primary_metric")
    !=
    "AP"
):

    raise RuntimeError(
        "Métrica primária inesperada."
    )


n_months = int(
    temporal[
        "n_months"
    ]
)


mean_ap_delta = float(
    temporal[
        "mean_AP_delta"
    ]
)


median_ap_delta = float(
    temporal[
        "median_AP_delta"
    ]
)


months_gn_better = int(
    temporal[
        "months_GN_EQ0_better"
    ]
)


months_logit_better = int(
    temporal[
        "months_LOGIT_better"
    ]
)


wilcoxon_p = float(
    temporal[
        "wilcoxon_p"
    ]
)


sign_test_p = float(
    temporal[
        "sign_test_p"
    ]
)


lomo_min = float(
    temporal[
        "LOMO_mean_min"
    ]
)


lomo_max = float(
    temporal[
        "LOMO_mean_max"
    ]
)


temporal_classification = str(
    temporal[
        "classification"
    ]
)


if n_months != 14:

    raise RuntimeError(
        f"Esperados 14 meses; encontrados {n_months}."
    )


if mean_ap_delta <= 0:

    raise RuntimeError(
        "Resultado temporal não favorece GN_EQ0 em AP médio."
    )


if lomo_min <= 0:

    raise RuntimeError(
        "LOMO não permaneceu positivo."
    )


print()
print(
    "[PASS] mean AP delta positive =",
    mean_ap_delta
)

print(
    "[PASS] LOMO mean range positive =",
    [
        lomo_min,
        lomo_max,
    ]
)


# ============================================================================================
# 3. VALIDAR RESULTADO DO ENSEMBLE
# ============================================================================================

blend_classification = str(
    blend[
        "classification"
    ]
)


blend_recommendation = str(
    blend[
        "recommended_model"
    ]
)


ensemble_ap_gain = float(
    blend[
        "ensemble_AP_gain_vs_best_base"
    ]
)


ensemble_brier_gap = float(
    blend[
        "ensemble_Brier_gap_vs_best_base"
    ]
)


ensemble_logloss_gap = float(
    blend[
        "ensemble_LogLoss_gap_vs_best_base"
    ]
)


guardrails_pass = bool(
    blend[
        "guardrails_pass"
    ]
)


if (
    blend_classification
    !=
    "ENSEMBLE_REJECTED_KEEP_BASE_MODEL"
):

    raise RuntimeError(
        "Resultado do ensemble inesperado: "
        + blend_classification
    )


if blend_recommendation != "GN_EQ0":

    raise RuntimeError(
        "Modelo recomendado pelo 25I não é GN_EQ0."
    )


if guardrails_pass is not False:

    raise RuntimeError(
        "Ensemble passou guardrails inesperadamente."
    )


print()
print(
    "[PASS] ensemble rejected"
)

print(
    "[PASS] recommended base model = GN_EQ0"
)


# ============================================================================================
# 4. REGRAS FINAIS DE SELEÇÃO
# ============================================================================================

final_recommended_model = "GN_EQ0"


universal_superiority = False


statistically_decisive_temporal_superiority = bool(
    wilcoxon_p
    <
    0.05
)


lomo_robust_positive_mean = bool(
    lomo_min
    >
    0
)


ensemble_required = False


final_interpretation = (
    "GN_EQ0 is the recommended final model for the evaluated case because "
    "it provides the best final compromise under the frozen primary metric "
    "and probability-quality guardrails. Its average AP advantage over "
    "LOGIT_MLE remains positive under every leave-one-month-out exclusion, "
    "but paired temporal inference is not statistically decisive. "
    "The prequential probability blend did not improve the best base model "
    "under the predefined guardrails and was therefore rejected."
)


# ============================================================================================
# 5. MACHINE-READABLE FINAL FREEZE
# ============================================================================================

generated = datetime.now(
    timezone.utc
).isoformat()


final = {
    "status":
        "FINAL_MODEL_SELECTION_COMPLETE",

    "generated_utc":
        generated,

    "scope":
        "DELIVERY_RISK_INTELLIGENCE",

    "recommended_model":
        final_recommended_model,

    "benchmark_model":
        "LOGIT_MLE",

    "rejected_extension":
        "PREQUENTIAL_GN_EQ0_LOGIT_BLEND",

    "primary_metric":
        "AP",

    "model_refit":
        False,

    "raw_modified":
        False,

    "new_model_family":
        False,

    "n_x_n_matrix_created":
        False,

    "temporal_robustness": {

        "n_months":
            n_months,

        "mean_AP_delta_GN_EQ0_minus_LOGIT":
            mean_ap_delta,

        "median_AP_delta":
            median_ap_delta,

        "months_GN_EQ0_better":
            months_gn_better,

        "months_LOGIT_better":
            months_logit_better,

        "wilcoxon_p":
            wilcoxon_p,

        "sign_test_p":
            sign_test_p,

        "LOMO_mean_min":
            lomo_min,

        "LOMO_mean_max":
            lomo_max,

        "LOMO_all_positive":
            lomo_robust_positive_mean,

        "classification":
            temporal_classification,
    },

    "ensemble_audit": {

        "classification":
            blend_classification,

        "ensemble_AP_gain_vs_best_base":
            ensemble_ap_gain,

        "ensemble_Brier_gap_vs_best_base":
            ensemble_brier_gap,

        "ensemble_LogLoss_gap_vs_best_base":
            ensemble_logloss_gap,

        "guardrails_pass":
            guardrails_pass,

        "ensemble_required":
            ensemble_required,
    },

    "claim_boundaries": {

        "universal_model_superiority":
            universal_superiority,

        "statistically_decisive_temporal_superiority":
            statistically_decisive_temporal_superiority,

        "causal_claim":
            False,
    },

    "final_interpretation":
        final_interpretation,

    "evidence_files": {

        "temporal_decision":
            str(
                TEMPORAL.relative_to(ROOT)
            ),

        "blend_decision":
            str(
                BLEND.relative_to(ROOT)
            ),

        "blend_summary":
            str(
                SUMMARY_BLEND.relative_to(ROOT)
            ),

        "alpha_history":
            str(
                ALPHA_HISTORY.relative_to(ROOT)
            ),
    },

    "evidence_sha256": {

        "temporal_decision":
            sha256(
                TEMPORAL
            ),

        "blend_decision":
            sha256(
                BLEND
            ),

        "blend_summary":
            sha256(
                SUMMARY_BLEND
            ),

        "alpha_history":
            sha256(
                ALPHA_HISTORY
            ),
    },
}


FINAL_JSON.write_text(
    json.dumps(
        final,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================================================
# 6. HUMAN-READABLE REPORT
# ============================================================================================

report = f"""# Final Model Selection — Delivery Risk Intelligence

Generated UTC: {generated}

## Decision

**Recommended final model: GN_EQ0**

Benchmark:

**LOGIT_MLE**

Prequential ensemble:

**REJECTED**

---

## Temporal Robustness

Evaluated OOT months:

**{n_months}**

Mean AP difference:

`GN_EQ0 - LOGIT_MLE = {mean_ap_delta:.12f}`

Median AP difference:

`{median_ap_delta:.12f}`

Months with higher GN_EQ0 AP:

**{months_gn_better}/{n_months}**

Months with higher LOGIT_MLE AP:

**{months_logit_better}/{n_months}**

Wilcoxon p-value:

`{wilcoxon_p:.12f}`

Sign-test p-value:

`{sign_test_p:.12f}`

Leave-one-month-out AP mean range:

`[{lomo_min:.12f}, {lomo_max:.12f}]`

Every leave-one-month-out mean remained positive:

**{lomo_robust_positive_mean}**

Temporal classification:

**{temporal_classification}**

---

## Ensemble Audit

Classification:

**{blend_classification}**

AP gain relative to best base model:

`{ensemble_ap_gain:.12f}`

Brier gap relative to best base model:

`{ensemble_brier_gap:.12f}`

LogLoss gap relative to best base model:

`{ensemble_logloss_gap:.12f}`

Guardrails passed:

**{guardrails_pass}**

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
"""


FINAL_MD.write_text(
    report,
    encoding="utf-8",
)


# ============================================================================================
# 7. FINAL CHECKS
# ============================================================================================

for path in [
    FINAL_JSON,
    FINAL_MD,
]:

    require(path)


print()
print("=" * 108)
print("FINAL MODEL SELECTION")
print("=" * 108)

print(
    json.dumps(
        final,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("=" * 108)
print("[PASS 25N] FINAL MODEL SELECTION COMPLETE")
print("RECOMMENDED_MODEL = GN_EQ0")
print("BENCHMARK_MODEL = LOGIT_MLE")
print("ENSEMBLE = REJECTED")
print("UNIVERSAL_SUPERIORITY = false")
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("=" * 108)
