#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.6-C
FORMAL FREEZE OF FUNCTIONAL COMPONENT DECISION
===============================================================================

OBJETIVO
--------
Congelar formalmente a decisão de representação para MODEL 01:

    K* = 0

Isto significa:

    ORDER_CORE_V1
    sem componentes funcionais RAW-30D

ESCOPO DA DECISÃO
-----------------
A decisão é válida para:

    MODEL_01_ORDER_LOGISTIC
    protocolo temporal supervisionado congelado
    métrica primária Average Precision
    representação funcional RAW-30D já auditada

NÃO significa:

    "a série temporal é inútil"

NÃO significa:

    "nenhum modelo futuro pode se beneficiar dela"

NÃO significa:

    "K=0 foi provado universalmente superior"

A conclusão correta é:

    não foi encontrada evidência robusta de ganho preditivo
    para K > 0 neste modelo e neste protocolo.

NÃO:
- treina modelo;
- executa PCA;
- usa curvas NPY;
- seleciona threshold;
- modifica RAW;
- cria Silver.
===============================================================================
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

DIR = (
    ROOT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

CONFIG_DIR = ROOT / "configs"
DOCS = ROOT / "docs"


C36A = DIR / "07e_supervised_k_summary.json"
C36B = DIR / "08g_c36b_summary.json"
C36B_VALID = DIR / "08f_c36b_validation.csv"
C36B_PAIRED = DIR / "08b_ap_paired_summary.csv"

CONTRACT = (
    CONFIG_DIR
    / "model_01_supervised_temporal_contract_v1.json"
)

OUT_DECISION = (
    DIR
    / "09a_model01_k0_formal_decision.json"
)

OUT_VALIDATION = (
    DIR
    / "09b_model01_k0_freeze_validation.csv"
)

OUT_REPORT = (
    DIR
    / "09c_model01_k0_freeze_report.txt"
)

OUT_DOC = (
    DOCS
    / "MODEL_01_FUNCTIONAL_K_DECISION.md"
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_json(path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# =============================================================================
# START
# =============================================================================

print()
print("=" * 118)
print("MODEL 01.0-C3.6-C — FORMAL FREEZE K*=0")
print("=" * 118)


required = [
    C36A,
    C36B,
    C36B_VALID,
    C36B_PAIRED,
    CONTRACT,
]

for p in required:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Arquivo ausente: {p}"
        )

    print(
        f"[PASS] {p.name}"
    )


a = load_json(C36A)
b = load_json(C36B)
contract = load_json(CONTRACT)

paired = pd.read_csv(
    C36B_PAIRED
)

valid_b = pd.read_csv(
    C36B_VALID
)


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


def add(name, ok, observed, expected):

    checks.append({
        "check": name,
        "status": "PASS" if ok else "FAIL",
        "observed": observed,
        "expected": expected,
    })


add(
    "c36a_pass",
    a.get("status") == "PASS",
    a.get("status"),
    "PASS",
)

add(
    "c36b_pass",
    b.get("status") == "PASS",
    b.get("status"),
    "PASS",
)

add(
    "c36b_validation_zero_failures",
    int(
        valid_b["status"]
        .eq("FAIL")
        .sum()
    ) == 0,
    int(
        valid_b["status"]
        .eq("FAIL")
        .sum()
    ),
    0,
)

add(
    "baseline_k0_highest_mean_ap",
    a.get(
        "descriptive_highest_mean_ap_k"
    ) == 0,
    a.get(
        "descriptive_highest_mean_ap_k"
    ),
    0,
)

add(
    "all_mean_deltas_negative",
    b.get(
        "all_observed_mean_deltas_negative"
    ) is True,
    b.get(
        "all_observed_mean_deltas_negative"
    ),
    True,
)

add(
    "all_median_deltas_negative",
    b.get(
        "all_observed_median_deltas_negative"
    ) is True,
    b.get(
        "all_observed_median_deltas_negative"
    ),
    True,
)

add(
    "contiguous_deletion_robust",
    b.get(
        "all_k_remain_negative_after_all_contiguous_deletions"
    ) is True,
    b.get(
        "all_k_remain_negative_after_all_contiguous_deletions"
    ),
    True,
)

add(
    "no_functional_simultaneous_superiority",
    b.get(
        "any_bootstrap_simultaneous_functional_superiority"
    ) is False,
    b.get(
        "any_bootstrap_simultaneous_functional_superiority"
    ),
    False,
)

add(
    "no_robust_functional_k",
    b.get(
        "robust_functional_superiority_k"
    ) == [],
    b.get(
        "robust_functional_superiority_k"
    ),
    [],
)

add(
    "no_claim_of_robust_k0_superiority",
    b.get(
        "robust_k0_superiority_k"
    ) == [],
    b.get(
        "robust_k0_superiority_k"
    ),
    [],
)

add(
    "secondary_metrics_nonpositive",
    b.get(
        "all_secondary_mean_benefits_nonpositive"
    ) is True,
    b.get(
        "all_secondary_mean_benefits_nonpositive"
    ),
    True,
)

add(
    "decision_candidate_ready",
    b.get(
        "decision_candidate"
    )
    ==
    "K0_READY_FOR_FORMAL_FREEZE_MODEL01",
    b.get(
        "decision_candidate"
    ),
    "K0_READY_FOR_FORMAL_FREEZE_MODEL01",
)

add(
    "paired_rows_13",
    len(paired) == 13,
    len(paired),
    13,
)

add(
    "all_paired_mean_delta_negative",
    (
        paired[
            "mean_delta_ap"
        ]
        <
        0
    ).all(),
    int(
        (
            paired[
                "mean_delta_ap"
            ]
            >=
            0
        ).sum()
    ),
    0,
)

add(
    "contract_frozen_before_performance",
    contract.get(
        "status"
    )
    ==
    "FROZEN_BEFORE_MODEL_PERFORMANCE",
    contract.get(
        "status"
    ),
    "FROZEN_BEFORE_MODEL_PERFORMANCE",
)


validation = pd.DataFrame(
    checks
)

validation.to_csv(
    OUT_VALIDATION,
    index=False,
)


failures = int(
    validation[
        "status"
    ]
    .eq("FAIL")
    .sum()
)


if failures:

    print()
    print(
        validation.to_string(
            index=False
        )
    )

    raise SystemExit(
        f"[FAIL] {failures} validações."
    )


# =============================================================================
# FORMAL DECISION
# =============================================================================

decision = {

    "decision_id":
        "MODEL_01_FUNCTIONAL_K_FREEZE_V1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN",

    "model":
        "MODEL_01_ORDER_LOGISTIC",

    "primary_metric":
        "AVERAGE_PRECISION",

    "functional_representation":
        "RAW_30D",

    "selected_k":
        0,

    "selected_functional_dimensions":
        0,

    "selected_model_input":
        "ORDER_CORE_V1_ONLY",

    "decision":
        "REJECT_FUNCTIONAL_COMPONENTS_FOR_PRIMARY_MODEL_01",

    "scientific_conclusion":
        (
            "No robust predictive improvement was observed "
            "for any K>0 relative to K=0 under the frozen "
            "temporal evaluation protocol."
        ),

    "important_scope_limit":
        (
            "This decision applies only to MODEL_01_ORDER_LOGISTIC "
            "under the frozen temporal protocol. It does not establish "
            "that temporal functional information is universally useless."
        ),

    "evidence": {

        "temporal_months":
            17,

        "alternative_k_values":
            13,

        "model_fits":
            238,

        "baseline_highest_mean_ap":
            True,

        "all_mean_delta_ap_negative":
            True,

        "all_median_delta_ap_negative":
            True,

        "all_contiguous_deletion_scenarios_negative":
            True,

        "functional_simultaneous_superiority_found":
            False,

        "robust_functional_superiority_k":
            [],

        "robust_k0_superiority_k":
            [],

        "secondary_mean_benefits_nonpositive":
            True,
    },

    "inferential_language": {

        "allowed":
            (
                "No robust evidence of predictive gain from "
                "functional components was found."
            ),

        "not_allowed":
            (
                "K=0 is universally or statistically superior "
                "to every K under all possible conditions."
            ),
    },

    "next_stage": {

        "representation":
            "ORDER_CORE_V1_ONLY",

        "functional_k":
            0,

        "task":
            "MODEL_01_BASELINE_PERFORMANCE_AND_PROBABILITY_EVALUATION",
    },

    "governance": {

        "k_final_frozen":
            True,

        "threshold_selected":
            False,

        "new_model_trained":
            False,

        "pca_refit":
            False,

        "raw_modified":
            False,

        "silver_created":
            False,
    },

    "provenance_sha256": {

        "07e_supervised_k_summary.json":
            sha256(C36A),

        "08g_c36b_summary.json":
            sha256(C36B),

        "08b_ap_paired_summary.csv":
            sha256(C36B_PAIRED),

        "model_01_supervised_temporal_contract_v1.json":
            sha256(CONTRACT),
    },
}


OUT_DECISION.write_text(
    json.dumps(
        decision,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

report = f"""
==============================================================================================================
MODEL 01.0-C3.6-C — FORMAL FUNCTIONAL K DECISION
==============================================================================================================

STATUS
--------------------------------------------------------------------------------------------------------------
Decision status                  : FROZEN
Model                            : MODEL_01_ORDER_LOGISTIC
Selected K                       : 0
Functional dimensions            : 0
Selected input                   : ORDER_CORE_V1_ONLY

SCIENTIFIC CONCLUSION
--------------------------------------------------------------------------------------------------------------
No robust predictive gain from K>0 was found under the frozen temporal protocol.

EVIDENCE
--------------------------------------------------------------------------------------------------------------
Temporal months                  : 17
Alternative K values             : 13
Total model fits                 : 238

All mean Delta AP < 0            : YES
All median Delta AP < 0          : YES
Contiguous deletion robust       : YES
Functional simultaneous win      : NO
Robust functional-superior K     : NONE
Robust K0-superior K             : NONE
Secondary metrics concordant     : YES

INTERPRETATION
--------------------------------------------------------------------------------------------------------------
K=0 is selected for MODEL 01 because no K>0 produced robust predictive improvement.

This does NOT establish universal statistical superiority of K=0.

The functional temporal representation remains scientifically informative and may be reconsidered in a different
supervised model family, nonlinear model, interaction model, or explicit regime-dependent challenger.

NEXT STAGE
--------------------------------------------------------------------------------------------------------------
ORDER_CORE_V1_ONLY

MODEL 01 baseline probability/ranking evaluation

GOVERNANCE
--------------------------------------------------------------------------------------------------------------
K final frozen                    : YES
Threshold selected               : NO
New model trained                : NO
PCA refit                        : NO
Silver created                   : NO
RAW modified                     : NO

==============================================================================================================
""".strip()


OUT_REPORT.write_text(
    report,
    encoding="utf-8",
)


# =============================================================================
# SCIENTIFIC DOC
# =============================================================================

doc = r"""
# MODEL 01 — Decisão formal do número de componentes funcionais

## Decisão

Para o modelo logístico primário, adotou-se

\[
\boxed{K^\star=0}.
\]

Consequentemente, a especificação que avança para a próxima etapa é

\[
\boxed{
X_i^{(01)}
=
X_{i,\mathrm{ORDER\_CORE\_V1}}
}
\]

sem inclusão dos escores funcionais derivados das curvas RAW-30D.

## Evidência supervisionada

Para cada mês futuro \(m\) e para cada \(K>0\), foi calculada

\[
\Delta_{m,K}
=
AP_{m,K}
-
AP_{m,0},
\]

onde \(AP\) representa Average Precision.

Todas as alternativas apresentaram

\[
\overline{\Delta}_{K}<0.
\]

As medianas das diferenças também foram negativas para todos os valores de
\(K>0\).

A direção da conclusão permaneceu inalterada quando foram removidos blocos
contíguos de um, dois ou três meses.

Assim, nenhum pequeno subconjunto temporal isolado foi responsável pela
vantagem média do baseline.

## Inferência dependente e comparação simultânea

Foi utilizado circular block bootstrap com comprimentos de bloco de 2, 3 e 4
meses e intervalos simultâneos max-t considerando conjuntamente os 13 valores
de \(K>0\).

Nenhuma alternativa funcional apresentou superioridade simultânea robusta.

Entretanto, os intervalos simultâneos também não permitiram classificar
individualmente os 13 valores de \(K\) como estatisticamente inferiores ao
baseline em todos os esquemas.

Por essa razão, a conclusão adotada não é:

> K=0 é universalmente superior.

A conclusão correta é:

> Não foi encontrada evidência robusta de ganho preditivo decorrente da
> inclusão das componentes funcionais RAW-30D no MODEL 01.

## Métricas secundárias

A direção média observada em ROC-AUC, Recall@10%, Log Loss e Brier Score foi
consistente com a métrica primária.

## Escopo da decisão

A escolha

\[
K^\star=0
\]

é específica ao modelo logístico primário e ao protocolo temporal congelado.

Ela não implica que a informação temporal funcional seja inútil em outros
modelos, particularmente modelos não lineares, supervisionados, com
interações ou dependentes de regime.
""".strip()


OUT_DOC.write_text(
    doc,
    encoding="utf-8",
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 118)
print("VALIDATION")
print("=" * 118)

print(
    validation.to_string(
        index=False
    )
)

print()
print("=" * 118)
print("DECISÃO FINAL")
print("=" * 118)

print()
print("MODEL                    = MODEL_01_ORDER_LOGISTIC")
print("K*                       = 0")
print("FUNCTIONAL DIMENSIONS    = 0")
print("INPUT                    = ORDER_CORE_V1_ONLY")

print()
print("INTERPRETAÇÃO:")
print(
    "Nenhum K>0 apresentou ganho preditivo robusto "
    "sob o protocolo temporal congelado."
)

print()
print(
    "NÃO afirmamos superioridade estatística universal de K=0."
)

print()
print("K FINAL                  = CONGELADO")
print("THRESHOLD                = NÃO")
print("NOVO MODELO              = NÃO TREINADO")
print("PCA                      = NÃO EXECUTADA")
print("SILVER                   = NÃO CRIADA")
print("RAW                      = INTACTO")

print()
print("[PASS] C3.6-C concluído.")
print("[PASS] K*=0 formalmente congelado para MODEL 01.")
print("[PASS] Próxima etapa: avaliação do baseline ORDER_CORE_V1.")

