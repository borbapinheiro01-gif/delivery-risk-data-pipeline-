#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE 26 — FINAL PROJECT CLOSEOUT
Delivery Risk Intelligence

Este módulo:
- NÃO retreina modelos;
- NÃO altera RAW;
- NÃO carrega CSVs grandes;
- NÃO cria matrizes N x N;
- valida o Markdown final;
- valida os módulos 23, 24 e 25;
- gera o freeze definitivo do projeto.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

FINAL = (
    ROOT
    / "reports"
    / "final_closeout"
)

DOC = (
    ROOT
    / "FINAL_PROJECT_DOCUMENTATION.md"
)

F23 = (
    FINAL
    / "23d_SCIENTIFIC_EVIDENCE_FREEZE.json"
)

F24 = (
    FINAL
    / "24d_PREDICTIVE_MODEL_AUDIT.json"
)

F25 = (
    FINAL
    / "25e_FINAL_GOVERNANCE_AUDIT.json"
)

CLAIMS = (
    FINAL
    / "23c_claim_registry.json"
)

GATES = (
    FINAL
    / "25d_governance_gates.csv"
)

FREEZE_OUT = (
    ROOT
    / "FINAL_PROJECT_FREEZE.json"
)

REPORT_OUT = (
    ROOT
    / "FINAL_PROJECT_REPORT.md"
)

MANIFEST_OUT = (
    FINAL
    / "26a_final_closeout_manifest.json"
)

DECISION_OUT = (
    FINAL
    / "26b_FINAL_PROJECT_CLOSEOUT.json"
)


# ==================================================================================================
# HELPERS
# ==================================================================================================

def require_file(path, min_bytes=1):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório ausente: {path}"
        )

    if path.stat().st_size < min_bytes:
        raise RuntimeError(
            f"Arquivo vazio ou pequeno demais: {path}"
        )


def load_json(path):
    require_file(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def require_text(text, needle):
    if needle not in text:
        raise RuntimeError(
            "DOCUMENTAÇÃO INCOMPLETA — "
            f"trecho obrigatório ausente: {needle}"
        )


# ==================================================================================================
# START
# ==================================================================================================

print("=" * 110)
print("MODULE 26 — FINAL PROJECT CLOSEOUT")
print("=" * 110)
print("MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("N_X_N_MATRIX_CREATED = false")
print()


# ==================================================================================================
# 1. ARQUIVOS OBRIGATÓRIOS
# ==================================================================================================

required_files = [
    DOC,
    F23,
    F24,
    F25,
    CLAIMS,
    GATES,
]

for path in required_files:
    require_file(path)

    print(
        "[PASS] exists:",
        path.relative_to(ROOT)
    )


# ==================================================================================================
# 2. LOAD 23 / 24 / 25
# ==================================================================================================

f23 = load_json(F23)
f24 = load_json(F24)
f25 = load_json(F25)


if (
    f23.get("status")
    !=
    "SCIENTIFIC_EVIDENCE_FREEZE_COMPLETE"
):
    raise RuntimeError(
        "Module 23 não está congelado: "
        f"{f23.get('status')}"
    )


if (
    f24.get("status")
    !=
    "PASS"
):
    raise RuntimeError(
        "Module 24 não está PASS: "
        f"{f24.get('status')}"
    )


if (
    f25.get("status")
    !=
    "PASS"
):
    raise RuntimeError(
        "Module 25 não está PASS: "
        f"{f25.get('status')}"
    )


print()
print("[PASS] MODULE 23")
print("[PASS] MODULE 24")
print("[PASS] MODULE 25")


# ==================================================================================================
# 3. VALIDAR DOCUMENTAÇÃO FINAL
# ==================================================================================================

doc_text = DOC.read_text(
    encoding="utf-8"
)

doc_lines = len(
    doc_text.splitlines()
)

print()
print(
    "FINAL_PROJECT_DOCUMENTATION lines =",
    doc_lines
)


if doc_lines < 500:
    raise RuntimeError(
        "FINAL_PROJECT_DOCUMENTATION.md está incompleto. "
        f"Foram encontradas apenas {doc_lines} linhas. "
        "Use o MD completo antes de executar o freeze."
    )


required_sections = [
    "# Delivery Risk Intelligence Platform",
    "# 1. Executive Summary",
    "# 4. Data Sources",
    "# 5. Data Engineering and Spatiotemporal Integration",
    "# 7. Cost Structure Audit",
    "# 8. Expected Freight",
    "# 9. Delivery Speed Audit",
    "# 10. Delivery Reliability",
    "# 11. Nonlinear Reliability Audit",
    "# 12. GN-ADMM Methodological Experiment",
    "# 15. Temporal GN-ADMM Validation",
    "# 16. Constraint Ablation",
    "# 17. Curvature Hypothesis",
    "# 18. Calibration Audit",
    "# 19. Final Scientific Hypothesis Matrix",
    "# 21. Governance and Reproducibility",
    "# 22. Resource and Memory Governance",
    "# 23. Allowed Scientific Claims",
    "# 24. Prohibited or Unsupported Claims",
    "# 25. Limitations",
    "# 26. Reproducibility Artifacts",
    "# 27. Final Scientific Interpretation",
    "# 28. Final Status",
    "# 30. Project Freeze",
]


for section in required_sections:
    require_text(
        doc_text,
        section
    )


required_scientific_guardrails = [
    "No causal identification is claimed.",
    "NO_ROBUST_OOT_GAIN",
    "FAVORABLE_NOT_DECISIVE",
    "POSITIVE QUADRATIC CURVATURE = NOT_SUPPORTED",
    "GN-ADMM CALIBRATION SUPERIORITY = NOT_SUPPORTED",
]


for item in required_scientific_guardrails:
    require_text(
        doc_text,
        item
    )


print(
    "[PASS] FINAL_PROJECT_DOCUMENTATION.md completo"
)

print(
    "[PASS] scientific guardrails presentes"
)


# ==================================================================================================
# 4. CONSISTÊNCIA DO FREEZE
# ==================================================================================================

checks = {
    "context_core_n_96211":
        f23.get(
            "context_core_n"
        )
        ==
        96211,

    "expected_freight_n_96211":
        f23.get(
            "expected_freight_n"
        )
        ==
        96211,

    "nonlinear_not_supported":
        f23.get(
            "nonlinear_predictive_gain"
        )
        ==
        "NOT_SUPPORTED",

    "gnadmm_temporal_partially_supported":
        f23.get(
            "gnadmm_temporal_AP"
        )
        ==
        "PARTIALLY_SUPPORTED",

    "positive_curvature_not_supported":
        f23.get(
            "positive_quadratic_curvature"
        )
        ==
        "NOT_SUPPORTED",

    "calibration_superiority_not_supported":
        f23.get(
            "gnadmm_calibration_superiority"
        )
        ==
        "NOT_SUPPORTED",

    "causal_claim_false":
        f23.get(
            "causal_claim"
        )
        is False,

    "universal_superiority_false":
        f23.get(
            "universal_model_superiority"
        )
        is False,

    "governance_failed_gates_zero":
        f25.get(
            "failed_gates"
        )
        ==
        0,

    "governance_caveat_gates_zero":
        f25.get(
            "caveat_gates"
        )
        ==
        0,

    "temporal_oot_pass":
        f25.get(
            "temporal_oot_gate"
        )
        ==
        "PASS",

    "allow_pickle_false":
        f25.get(
            "gn_design_allow_pickle_false"
        )
        is True,
}


failed_checks = [
    key
    for key, value
    in checks.items()
    if not value
]


if failed_checks:
    raise RuntimeError(
        "Falharam checks finais: "
        +
        ", ".join(
            failed_checks
        )
    )


print()

for key in checks:
    print(
        "[PASS]",
        key
    )


# ==================================================================================================
# 5. MANIFESTO FINAL
# ==================================================================================================

manifest_files = [
    DOC,
    F23,
    F24,
    F25,
    CLAIMS,
    GATES,
]


manifest_rows = []

for path in manifest_files:

    manifest_rows.append(
        {
            "path":
                str(
                    path.relative_to(ROOT)
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256(path),
        }
    )


manifest = {
    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "files":
        manifest_rows,
}


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print(
    "[PASS] 26a_final_closeout_manifest.json"
)


# ==================================================================================================
# 6. FREEZE DEFINITIVO
# ==================================================================================================

generated = datetime.now(
    timezone.utc
).isoformat()


freeze = {
    "status":
        "PROJECT_SCIENTIFICALLY_CLOSED",

    "generated_utc":
        generated,

    "scope":
        "DELIVERY_RISK_INTELLIGENCE",

    "model_refit":
        False,

    "raw_modified":
        False,

    "production_deployment":
        False,

    "scientific_case_study_closed":
        True,

    "scientific_freeze":
        f23.get(
            "status"
        ),

    "predictive_audit":
        f24.get(
            "status"
        ),

    "governance_audit":
        f25.get(
            "status"
        ),

    "context_core_n":
        f23.get(
            "context_core_n"
        ),

    "expected_freight_n":
        f23.get(
            "expected_freight_n"
        ),

    "nonlinear_predictive_gain":
        f23.get(
            "nonlinear_predictive_gain"
        ),

    "gnadmm_temporal_AP":
        f23.get(
            "gnadmm_temporal_AP"
        ),

    "positive_quadratic_curvature":
        f23.get(
            "positive_quadratic_curvature"
        ),

    "gnadmm_calibration_superiority":
        f23.get(
            "gnadmm_calibration_superiority"
        ),

    "causal_claim":
        False,

    "universal_model_superiority":
        False,

    "global_external_GN_convergence_certified":
        False,

    "final_documentation":
        "FINAL_PROJECT_DOCUMENTATION.md",

    "closeout_manifest":
        "reports/final_closeout/26a_final_closeout_manifest.json",
}


FREEZE_OUT.write_text(
    json.dumps(
        freeze,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(
    "[PASS] FINAL_PROJECT_FREEZE.json"
)


# ==================================================================================================
# 7. RELATÓRIO FINAL
# ==================================================================================================

metric_winners = f24.get(
    "metric_winners",
    []
)


winner_lines = []

for item in metric_winners:

    metric = item.get(
        "metric"
    )

    winner = item.get(
        "winner"
    )

    value = item.get(
        "value"
    )

    winner_lines.append(
        f"- {metric}: `{winner}` = {value}"
    )


if not winner_lines:
    winner_lines = [
        "- Metric winner records unavailable."
    ]


winner_text = "\n".join(
    winner_lines
)


report = f"""# Delivery Risk Intelligence — Final Project Closeout Report

Generated UTC: {generated}

## Final Status

**PROJECT_SCIENTIFICALLY_CLOSED**

## Final Analytical Population

- Consolidated spatiotemporal core: **{f23.get("context_core_n"):,} orders**
- Expected-freight artifact: **{f23.get("expected_freight_n"):,} orders**

## Scientific Freeze

- Nonlinear predictive gain: **{f23.get("nonlinear_predictive_gain")}**
- GN-ADMM temporal AP evidence: **{f23.get("gnadmm_temporal_AP")}**
- Positive quadratic curvature: **{f23.get("positive_quadratic_curvature")}**
- GN-ADMM calibration superiority: **{f23.get("gnadmm_calibration_superiority")}**

## Final Predictive Metric Winners

{winner_text}

## Governance

- Governance status: **{f25.get("status")}**
- Failed gates: **{f25.get("failed_gates")}**
- Caveat gates: **{f25.get("caveat_gates")}**
- Temporal OOT gate: **{f25.get("temporal_oot_gate")}**
- GN design allow_pickle=False: **{f25.get("gn_design_allow_pickle_false")}**

## Final Scientific Interpretation

The project is closed as an analytical and scientific case study.

Flexible nonlinear reliability specifications did not establish robust
out-of-time predictive superiority.

The GN-ADMM Brier-NLS experiment demonstrated numerical applicability on the
evaluated real-world logistics data.

Its mean Average Precision advantage was favorable but not decisive under
paired temporal inference.

The externally imposed positive-quadratic constraint was not supported as a
structural empirical conclusion.

Different metrics favored different models.

No universal model superiority is claimed.

## Claim Boundaries

The project does not establish:

- causal effects of freight deviations on delivery delay;
- universal superiority of GN-ADMM over logistic regression;
- global convergence of the external nonlinear Gauss-Newton sequence;
- a true U-shaped mechanism implied by beta_z2 >= 0;
- equivalence between Haversine distance and road-network distance;
- point-in-time operational availability of retrospective GDP.

## Final Documentation

`FINAL_PROJECT_DOCUMENTATION.md`

## Machine-readable Freeze

`FINAL_PROJECT_FREEZE.json`

## Final Manifest

`reports/final_closeout/26a_final_closeout_manifest.json`
"""


REPORT_OUT.write_text(
    report,
    encoding="utf-8"
)


print(
    "[PASS] FINAL_PROJECT_REPORT.md"
)


# ==================================================================================================
# 8. DECISÃO MACHINE-READABLE DO 26
# ==================================================================================================

decision = {
    "status":
        "PASS",

    "project_status":
        "PROJECT_SCIENTIFICALLY_CLOSED",

    "module_23":
        "PASS",

    "module_24":
        "PASS",

    "module_25":
        "PASS",

    "module_26":
        "PASS",

    "documentation_validated":
        True,

    "documentation_lines":
        doc_lines,

    "freeze_created":
        True,

    "final_report_created":
        True,

    "manifest_created":
        True,

    "raw_modified":
        False,

    "model_refit":
        False,

    "n_x_n_matrix_created":
        False,
}


DECISION_OUT.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(
    "[PASS] 26b_FINAL_PROJECT_CLOSEOUT.json"
)


# ==================================================================================================
# 9. VALIDAÇÃO DOS ARQUIVOS GERADOS
# ==================================================================================================

generated_files = [
    FREEZE_OUT,
    REPORT_OUT,
    MANIFEST_OUT,
    DECISION_OUT,
]


for path in generated_files:
    require_file(
        path
    )


print()
print("=" * 110)
print("FINAL PROJECT FREEZE")
print("=" * 110)

print(
    json.dumps(
        freeze,
        indent=2,
        ensure_ascii=False
    )
)

print()
print("=" * 110)
print("[PASS 26] PROJECT SCIENTIFICALLY CLOSED.")
print("RAW_MODIFIED = false")
print("MODEL_REFIT = false")
print("N_X_N_MATRIX_CREATED = false")
print("=" * 110)
