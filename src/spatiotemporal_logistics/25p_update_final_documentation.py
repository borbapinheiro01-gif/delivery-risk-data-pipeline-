#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25P — UPDATE FINAL PROJECT DOCUMENTATION

Atualiza o FINAL_PROJECT_DOCUMENTATION.md completo com os resultados finais:

- 25F: robustez temporal GN_EQ0 vs LOGIT_MLE
- 25I: ensemble prequential
- 25N: seleção final do modelo

Não:
- retreina modelos;
- altera RAW;
- recalcula features;
- muda resultados científicos;
- cria matriz N x N.

Este script SOMENTE atualiza a documentação com resultados
já congelados em arquivos machine-readable.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

DOC = ROOT / "FINAL_PROJECT_DOCUMENTATION.md"

OUT = ROOT / "reports" / "final_closeout"


TEMPORAL = (
    OUT
    / "25h_TEMPORAL_ROBUSTNESS_DECISION.json"
)

BLEND = (
    OUT
    / "25l_FINAL_BLEND_DECISION.json"
)

MODEL_SELECTION = (
    OUT
    / "25n_FINAL_MODEL_SELECTION.json"
)

BLEND_SUMMARY = (
    OUT
    / "25k_prequential_blend_summary.csv"
)

BACKUP = (
    OUT
    / "FINAL_PROJECT_DOCUMENTATION_BEFORE_25P.md"
)

UPDATE_JSON = (
    OUT
    / "25p_DOCUMENTATION_UPDATE.json"
)


MARKER_START = (
    "<!-- BEGIN FINAL MODEL SELECTION UPDATE 25P -->"
)

MARKER_END = (
    "<!-- END FINAL MODEL SELECTION UPDATE 25P -->"
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

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


print("=" * 108)
print("25P — UPDATE FINAL PROJECT DOCUMENTATION")
print("=" * 108)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("DOCUMENTATION_ONLY = true")
print()


# ================================================================================================
# 1. REQUIRED INPUTS
# ================================================================================================

for path in [
    DOC,
    TEMPORAL,
    BLEND,
    MODEL_SELECTION,
    BLEND_SUMMARY,
]:

    require(path)

    print(
        "[PASS] exists:",
        path.relative_to(ROOT)
    )


# ================================================================================================
# 2. VALIDATE FULL MD
# ================================================================================================

text = DOC.read_text(
    encoding="utf-8"
)


line_count_before = len(
    text.splitlines()
)


print()
print(
    "MD lines before =",
    line_count_before
)


if line_count_before < 500:

    raise RuntimeError(
        "FINAL_PROJECT_DOCUMENTATION.md ainda é a versão "
        f"incompleta/truncada ({line_count_before} linhas). "
        "Substitua pelo MD completo antes de executar 25P."
    )


required_sections = [
    "# Delivery Risk Intelligence Platform",
    "# 1. Executive Summary",
    "# 4. Data Sources",
    "# 7. Cost Structure Audit",
    "# 8. Expected Freight",
    "# 9. Delivery Speed Audit",
    "# 10. Delivery Reliability",
    "# 11. Nonlinear Reliability Audit",
    "# 12. GN-ADMM Methodological Experiment",
    "# 15. Temporal GN-ADMM Validation",
    "# 16. Constraint Ablation",
    "# 18. Calibration Audit",
    "# 21. Governance and Reproducibility",
    "# 25. Limitations",
    "# 27. Final Scientific Interpretation",
    "# 28. Final Status",
    "# 29. Reporting Framework Note",
    "# 30. Project Freeze",
]


for section in required_sections:

    if section not in text:

        raise RuntimeError(
            f"Seção obrigatória ausente no MD: {section}"
        )


print(
    "[PASS] full documentation structure validated"
)


# ================================================================================================
# 3. READ FINAL RESULTS
# ================================================================================================

temporal = load_json(
    TEMPORAL
)

blend = load_json(
    BLEND
)

selection = load_json(
    MODEL_SELECTION
)


if temporal.get("status") != "PASS":

    raise RuntimeError(
        "25F temporal robustness não está PASS."
    )


if blend.get("status") != "PASS":

    raise RuntimeError(
        "25I blend não está PASS."
    )


if (
    selection.get("status")
    !=
    "FINAL_MODEL_SELECTION_COMPLETE"
):

    raise RuntimeError(
        "25N final model selection não está completo."
    )


if (
    selection.get("recommended_model")
    !=
    "GN_EQ0"
):

    raise RuntimeError(
        "Modelo final congelado não é GN_EQ0."
    )


if (
    selection.get("benchmark_model")
    !=
    "LOGIT_MLE"
):

    raise RuntimeError(
        "Benchmark congelado não é LOGIT_MLE."
    )


# ================================================================================================
# 4. VALUES
# ================================================================================================

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

n_months = int(
    temporal[
        "n_months"
    ]
)

gn_better = int(
    temporal[
        "months_GN_EQ0_better"
    ]
)

logit_better = int(
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

ensemble_months_vs_gn = int(
    blend[
        "ensemble_months_AP_better_than_GN_EQ0"
    ]
)

ensemble_months_vs_logit = int(
    blend[
        "ensemble_months_AP_better_than_LOGIT"
    ]
)

ensemble_classification = str(
    blend[
        "classification"
    ]
)

recommended_model = str(
    selection[
        "recommended_model"
    ]
)

benchmark_model = str(
    selection[
        "benchmark_model"
    ]
)


# ================================================================================================
# 5. READ SAME-PERIOD MODEL SUMMARY
# ================================================================================================

import csv


rows = []

with BLEND_SUMMARY.open(
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)


by_model = {
    row["model"]:
        row
    for row in rows
}


for required_model in [
    "GN_EQ0",
    "LOGIT_MLE",
    "ENSEMBLE",
]:

    if required_model not in by_model:

        raise RuntimeError(
            f"Modelo ausente no 25k: {required_model}"
        )


gn = by_model[
    "GN_EQ0"
]

logit = by_model[
    "LOGIT_MLE"
]

ensemble = by_model[
    "ENSEMBLE"
]


# ================================================================================================
# 6. BUILD NEW DOCUMENTATION SECTION
# ================================================================================================

update_section = f"""
---

{MARKER_START}

# 29A. Final Model Robustness and Selection Update

This section records the final modeling evidence generated after the initial
scientific evidence freeze. It does not replace the earlier GN-ADMM,
nonlinear, calibration or governance audits. It adds the final temporal
robustness check, the low-cost prequential blend experiment and the final
model-selection decision.

## 29A.1 Final temporal robustness: GN_EQ0 versus LOGIT_MLE

The final paired temporal audit compared `GN_EQ0` with `LOGIT_MLE` over
**{n_months} OOT months** using Average Precision as the frozen primary metric.

The mean monthly difference was:

`mean(AP_GN_EQ0 - AP_LOGIT_MLE) = {mean_ap_delta:.12f}`

The median monthly difference was:

`median(AP_GN_EQ0 - AP_LOGIT_MLE) = {median_ap_delta:.12f}`

`GN_EQ0` produced higher AP in:

**{gn_better}/{n_months} months**

while `LOGIT_MLE` produced higher AP in:

**{logit_better}/{n_months} months**

The paired Wilcoxon p-value was:

`{wilcoxon_p:.12f}`

and the two-sided sign-test p-value was:

`{sign_test_p:.12f}`

These tests do not support a statistically decisive temporal superiority claim.

However, the leave-one-month-out mean AP difference remained positive under
every single-month exclusion:

`LOMO mean delta range = [{lomo_min:.12f}, {lomo_max:.12f}]`

Therefore the positive mean AP difference is not attributable to one isolated
month.

Final temporal classification:

**{temporal_classification}**

The appropriate interpretation is that `GN_EQ0` shows a favorable and
leave-one-month-out robust mean AP advantage over `LOGIT_MLE`, while paired
temporal inference remains non-decisive.

## 29A.2 Low-cost prequential probability blend

A final low-complexity ensemble experiment combined:

`GN_EQ0`

and:

`LOGIT_MLE`

using:

`p_blend = alpha * p_GN_EQ0 + (1 - alpha) * p_LOGIT_MLE`

with:

`alpha in {{0, 0.25, 0.50, 0.75, 1.00}}`

For each evaluation month, the value of `alpha` was selected using only prior
OOT months and historical Brier score. The evaluation therefore preserved the
temporal ordering of the experiment.

The first OOT month was used as warm-up. The final same-period comparison used
**13 evaluation months** and **78,009 predictions**.

Final same-period pooled results were:

| Model | AP | Brier | LogLoss | ROC-AUC | ECE10 quantile |
|---|---:|---:|---:|---:|---:|
| GN_EQ0 | {float(gn["pooled_AP"]):.6f} | {float(gn["pooled_Brier"]):.6f} | {float(gn["pooled_LogLoss"]):.6f} | {float(gn["pooled_AUC"]):.6f} | {float(gn["pooled_ECE10_quantile"]):.6f} |
| LOGIT_MLE | {float(logit["pooled_AP"]):.6f} | {float(logit["pooled_Brier"]):.6f} | {float(logit["pooled_LogLoss"]):.6f} | {float(logit["pooled_AUC"]):.6f} | {float(logit["pooled_ECE10_quantile"]):.6f} |
| ENSEMBLE | {float(ensemble["pooled_AP"]):.6f} | {float(ensemble["pooled_Brier"]):.6f} | {float(ensemble["pooled_LogLoss"]):.6f} | {float(ensemble["pooled_AUC"]):.6f} | {float(ensemble["pooled_ECE10_quantile"]):.6f} |

Relative to the best base model, the ensemble produced:

`AP difference = {ensemble_ap_gain:.12f}`

`Brier difference = {ensemble_brier_gap:.12f}`

`LogLoss difference = {ensemble_logloss_gap:.12f}`

The ensemble produced higher monthly AP than `GN_EQ0` in only:

**{ensemble_months_vs_gn} of 13 evaluation months**

and higher monthly AP than `LOGIT_MLE` in:

**{ensemble_months_vs_logit} of 13 evaluation months**

Final ensemble classification:

**{ensemble_classification}**

The ensemble is therefore not retained.

The experiment provides useful negative evidence: combining the two base
probability forecasts did not improve the frozen primary metric while
preserving the probability-quality guardrails.

## 29A.3 Final model selection

The final recommended model for this evaluated case is:

**{recommended_model}**

The principal benchmark remains:

**{benchmark_model}**

The final decision is based on the complete evidence chain rather than on a
single metric:

- `GN_EQ0` achieved the favorable mean AP difference in the 14-month temporal audit;
- the mean AP difference remained positive under every leave-one-month-out exclusion;
- the paired temporal tests did not establish statistically decisive superiority;
- `GN_EQ0` achieved the strongest final compromise across AP, Brier and LogLoss;
- `LOGIT_MLE` remained competitive and retained advantages on some discrimination/calibration diagnostics;
- the prequential ensemble did not improve the best base model and was rejected.

Therefore:

**RECOMMENDED_MODEL = GN_EQ0**

**BENCHMARK_MODEL = LOGIT_MLE**

**ENSEMBLE = REJECTED**

**UNIVERSAL_MODEL_SUPERIORITY = FALSE**

**STATISTICALLY_DECISIVE_TEMPORAL_SUPERIORITY = FALSE**

**CAUSAL_CLAIM = FALSE**

No additional model family is required for the frozen analytical case.

{MARKER_END}
"""


# ================================================================================================
# 7. IDEMPOTENT UPDATE
# ================================================================================================

# Se já existir uma atualização 25P anterior, removê-la antes de inserir a nova.
if (
    MARKER_START in text
    and
    MARKER_END in text
):

    before = text.split(
        MARKER_START,
        1
    )[0]

    after = text.split(
        MARKER_END,
        1
    )[1]

    text = (
        before.rstrip()
        +
        "\n"
        +
        after.lstrip()
    )

    print(
        "[INFO] previous 25P section removed before replacement"
    )


freeze_anchor = "# 30. Project Freeze"


if freeze_anchor not in text:

    raise RuntimeError(
        "Anchor '# 30. Project Freeze' não encontrado."
    )


new_text = text.replace(
    freeze_anchor,
    update_section
    +
    "\n---\n\n"
    +
    freeze_anchor,
    1,
)


# ================================================================================================
# 8. BACKUP + WRITE
# ================================================================================================

shutil.copy2(
    DOC,
    BACKUP
)


DOC.write_text(
    new_text,
    encoding="utf-8"
)


# ================================================================================================
# 9. VALIDATE RESULT
# ================================================================================================

updated = DOC.read_text(
    encoding="utf-8"
)


line_count_after = len(
    updated.splitlines()
)


checks = {
    "full_md_preserved":
        line_count_after > 500,

    "marker_start":
        MARKER_START in updated,

    "marker_end":
        MARKER_END in updated,

    "final_model_GN_EQ0":
        "**RECOMMENDED_MODEL = GN_EQ0**"
        in updated,

    "benchmark_LOGIT":
        "**BENCHMARK_MODEL = LOGIT_MLE**"
        in updated,

    "ensemble_rejected":
        "**ENSEMBLE = REJECTED**"
        in updated,

    "universal_superiority_false":
        "**UNIVERSAL_MODEL_SUPERIORITY = FALSE**"
        in updated,

    "project_freeze_preserved":
        "# 30. Project Freeze"
        in updated,

    "final_end_preserved":
        "**End of final scientific documentation.**"
        in updated,
}


failed = [
    key
    for key, value
    in checks.items()
    if not value
]


if failed:

    shutil.copy2(
        BACKUP,
        DOC
    )

    raise RuntimeError(
        "Falha na validação do MD atualizado. "
        "Backup restaurado. "
        f"Checks falhos: {failed}"
    )


# ================================================================================================
# 10. UPDATE AUDIT
# ================================================================================================

audit = {
    "status":
        "PASS",

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "documentation_file":
        str(
            DOC.relative_to(ROOT)
        ),

    "backup_file":
        str(
            BACKUP.relative_to(ROOT)
        ),

    "lines_before":
        line_count_before,

    "lines_after":
        line_count_after,

    "recommended_model":
        recommended_model,

    "benchmark_model":
        benchmark_model,

    "ensemble":
        "REJECTED",

    "temporal_classification":
        temporal_classification,

    "universal_model_superiority":
        False,

    "statistically_decisive_temporal_superiority":
        False,

    "causal_claim":
        False,

    "model_refit":
        False,

    "raw_modified":
        False,

    "documentation_sha256":
        sha256(
            DOC
        ),

    "validation_checks":
        checks,
}


UPDATE_JSON.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print("=" * 108)
print("FINAL DOCUMENTATION UPDATE")
print("=" * 108)

print(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("[PASS 25P] FINAL DOCUMENTATION UPDATED")
print("RECOMMENDED_MODEL = GN_EQ0")
print("BENCHMARK_MODEL = LOGIT_MLE")
print("ENSEMBLE = REJECTED")
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
