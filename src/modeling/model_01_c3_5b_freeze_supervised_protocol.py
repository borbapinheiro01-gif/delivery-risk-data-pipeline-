#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.5-B
FREEZE SUPERVISED TEMPORAL EVALUATION PROTOCOL
===============================================================================

Somente GOVERNANÇA.

NÃO:
- treina modelo;
- executa PCA;
- usa probabilidades preditas;
- seleciona K;
- seleciona threshold;
- altera RAW;
- cria Silver.

Congela ANTES de observar performance:

1. folds temporais;
2. regra de disponibilidade do label;
3. família do primeiro classificador;
4. grade de K;
5. métricas primárias/secundárias;
6. regra de PCA/scaling;
7. comparação obrigatória K=0.
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

FOLDS = DIR / "06a_supervised_temporal_fold_audit.csv"
SUMMARY_A = DIR / "06d_supervised_temporal_protocol_summary.json"
ROBUST = DIR / "05e_robust_rank_evidence_summary.json"

CONTRACT = (
    ROOT
    / "configs"
    / "model_01_supervised_temporal_contract_v1.json"
)

VALIDATION = DIR / "06f_supervised_protocol_freeze_validation.csv"
SUMMARY = DIR / "06g_supervised_protocol_freeze_summary.json"
REPORT = DIR / "06h_supervised_protocol_freeze_report.txt"


print()
print("=" * 120)
print("MODEL 01.0-C3.5-B — FREEZE SUPERVISED TEMPORAL PROTOCOL")
print("=" * 120)


# =============================================================================
# PREREQUISITES
# =============================================================================

for p in [
    FOLDS,
    SUMMARY_A,
    ROBUST,
]:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Ausente: {p}"
        )

    print(f"[PASS] {p.name}")


with SUMMARY_A.open(encoding="utf-8") as f:
    c35a = json.load(f)

with ROBUST.open(encoding="utf-8") as f:
    structural = json.load(f)


if c35a.get("status") != "PASS":
    raise RuntimeError("C3.5-A não está PASS.")

if structural.get("status") != "PASS":
    raise RuntimeError("C3.4-D não está PASS.")


folds = pd.read_csv(FOLDS)


# =============================================================================
# CANDIDATE K GRID
# =============================================================================

# K=0 é obrigatório:
# baseline ORDER_CORE_V1 sem módulo funcional.
#
# 1,3,6 vêm da evidência estrutural modal.
#
# Pontos maiores impedem que a análise estrutural não supervisionada
# impeça componentes potencialmente úteis ao target.
#
# 30 representa a curva inteira no espaço PCA (sem truncamento funcional).

K_GRID = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    10,
    15,
    20,
    23,
    25,
    27,
    30,
]


# =============================================================================
# METRICS
# =============================================================================

METRICS = {
    "primary": {
        "average_precision": (
            "Primary ranking metric for delayed-order class."
        ),
    },

    "secondary_ranking": {
        "roc_auc": (
            "Threshold-free ranking metric."
        ),

        "ap_lift_over_prevalence": (
            "Average precision divided by test-month "
            "positive prevalence."
        ),
    },

    "probabilistic": {
        "log_loss": (
            "Proper scoring rule."
        ),

        "brier_score": (
            "Mean squared probability error."
        ),
    },

    "operational": {
        "recall_at_05pct": (
            "Fraction of true delays captured among "
            "the highest-risk 5% of orders."
        ),

        "recall_at_10pct": (
            "Fraction of true delays captured among "
            "the highest-risk 10% of orders."
        ),

        "recall_at_20pct": (
            "Fraction of true delays captured among "
            "the highest-risk 20% of orders."
        ),
    },
}


# =============================================================================
# CONTRACT
# =============================================================================

contract = {
    "contract_id":
        "MODEL_01_SUPERVISED_TEMPORAL_CONTRACT_V1",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN_BEFORE_MODEL_PERFORMANCE",

    "model":
        "MODEL_01_ORDER_LOGISTIC",

    "prediction_time":
        "order_purchase_timestamp",

    "target":
        "late_delivery_calendar_day",

    "target_positive_class":
        1,

    "representation":
        "RAW_30D",

    "temporal_protocol": {
        "type":
            "EXPANDING_HISTORY_MONTHLY_OUT_OF_TIME",

        "test_months":
            folds["test_month"].tolist(),

        "first_test_month":
            str(
                folds.iloc[0]["test_month"]
            ),

        "last_test_month":
            str(
                folds.iloc[-1]["test_month"]
            ),

        "training_purchase_rule":
            "order_purchase_timestamp < fit_cutoff",

        "training_label_availability_rule":
            "order_delivered_customer_date < fit_cutoff",

        "test_rule":
            "purchase belongs to current test month",
    },

    "preprocessing_contract": {
        "fit_scaler_on_train_only":
            True,

        "transform_test_with_train_scaler_only":
            True,

        "fit_functional_pca_on_train_only":
            True,

        "transform_test_with_train_pca_only":
            True,

        "pca_channels": [
            "purchase_volume",
            "purchase_freight",
        ],

        "functional_window_days":
            30,

        "smoothing":
            False,
    },

    "classifier_contract": {
        "family":
            "LOGISTIC_REGRESSION",

        "penalty":
            "L2",

        "solver":
            "lbfgs",

        "max_iter":
            5000,

        "random_state":
            20260830,

        "class_weight":
            None,

        "note":
            (
                "Class weighting is not tuned in the first "
                "K-utility experiment to avoid introducing "
                "another selection dimension."
            ),
    },

    "k_grid":
        K_GRID,

    "mandatory_baseline": {
        "k":
            0,

        "meaning":
            "ORDER_CORE_V1 only; no functional components.",
    },

    "full_functional_control": {
        "k":
            30,

        "meaning":
            (
                "No PCA truncation within each 30-day "
                "functional channel."
            ),
    },

    "metrics":
        METRICS,

    "threshold_policy": {
        "classification_threshold_selected":
            False,

        "reason":
            (
                "This stage evaluates ranking/probability "
                "performance before operational threshold selection."
            ),
    },

    "selection_scope": {
        "k_final_selected":
            False,

        "final_holdout_selected":
            False,

        "adaptive_evidence_used":
            False,

        "production_claim_allowed":
            False,
    },

    "governance": {
        "raw_modified":
            False,

        "silver_created":
            False,

        "classifier_trained_at_contract_creation":
            False,
    },
}


CONTRACT.write_text(
    json.dumps(
        contract,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


def add(name, ok, observed, expected):

    checks.append({
        "check":
            name,

        "status":
            "PASS" if ok else "FAIL",

        "observed":
            observed,

        "expected":
            expected,
    })


add(
    "c35a_pass",
    c35a.get("status") == "PASS",
    c35a.get("status"),
    "PASS",
)

add(
    "c34d_pass",
    structural.get("status") == "PASS",
    structural.get("status"),
    "PASS",
)

add(
    "temporal_months_17",
    len(folds) == 17,
    len(folds),
    17,
)

add(
    "k_grid_unique",
    len(K_GRID) == len(set(K_GRID)),
    len(K_GRID) - len(set(K_GRID)),
    0,
)

add(
    "k_grid_within_0_30",
    all(
        0 <= k <= 30
        for k in K_GRID
    ),
    [min(K_GRID), max(K_GRID)],
    [0, 30],
)

add(
    "baseline_k0_present",
    0 in K_GRID,
    0 in K_GRID,
    True,
)

add(
    "structural_modes_present",
    all(
        k in K_GRID
        for k in [1, 3, 6]
    ),
    [
        k
        for k in [1, 3, 6]
        if k in K_GRID
    ],
    [1, 3, 6],
)

add(
    "full_rank_control_present",
    30 in K_GRID,
    30 in K_GRID,
    True,
)

add(
    "primary_metric_average_precision",
    "average_precision"
    in METRICS["primary"],
    list(
        METRICS["primary"]
    ),
    ["average_precision"],
)

add(
    "pca_train_only",
    contract[
        "preprocessing_contract"
    ][
        "fit_functional_pca_on_train_only"
    ],
    True,
    True,
)

add(
    "scaler_train_only",
    contract[
        "preprocessing_contract"
    ][
        "fit_scaler_on_train_only"
    ],
    True,
    True,
)

add(
    "threshold_not_selected",
    not contract[
        "threshold_policy"
    ][
        "classification_threshold_selected"
    ],
    False,
    False,
)

add(
    "classifier_not_trained",
    not contract[
        "governance"
    ][
        "classifier_trained_at_contract_creation"
    ],
    False,
    False,
)


validation = pd.DataFrame(checks)

validation.to_csv(
    VALIDATION,
    index=False,
)

failures = int(
    validation[
        "status"
    ].eq("FAIL").sum()
)


# =============================================================================
# HASH
# =============================================================================

sha256 = hashlib.sha256(
    CONTRACT.read_bytes()
).hexdigest()


summary = {
    "step":
        "MODEL_01_0_C3_5B_FREEZE_SUPERVISED_PROTOCOL",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "contract_status":
        contract["status"],

    "contract_file":
        str(
            CONTRACT.relative_to(ROOT)
        ),

    "contract_sha256":
        sha256,

    "temporal_months":
        17,

    "k_candidates":
        len(K_GRID),

    "k_grid":
        K_GRID,

    "primary_metric":
        "average_precision",

    "mandatory_baseline_k":
        0,

    "full_rank_control_k":
        30,

    "target_performance_observed":
        False,

    "k_final_selected":
        False,

    "threshold_selected":
        False,

    "classifier_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures,
}


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


report = f"""
========================================================================================================================
MODEL 01.0-C3.5-B — SUPERVISED TEMPORAL PROTOCOL FREEZE
========================================================================================================================

STATUS
------------------------------------------------------------------------------------------------------------------------
{summary["status"]}

CONTRACT STATUS
------------------------------------------------------------------------------------------------------------------------
FROZEN BEFORE MODEL PERFORMANCE

TEMPORAL DESIGN
------------------------------------------------------------------------------------------------------------------------
Months                          : 17
First                           : 2017-04
Last                            : 2018-08

Training:
    purchase_j < fit_cutoff
    delivery_j < fit_cutoff

Preprocessing:
    scaler fit on TRAIN only
    PCA fit on TRAIN only
    TEST only transformed by TRAIN objects

MODEL
------------------------------------------------------------------------------------------------------------------------
Family                          : Logistic Regression
Penalty                         : L2
Solver                          : lbfgs
Class weight                    : None

K GRID
------------------------------------------------------------------------------------------------------------------------
{K_GRID}

K=0                             : ORDER_CORE_V1 baseline
K=30                            : full 30-day functional control

PRIMARY METRIC
------------------------------------------------------------------------------------------------------------------------
Average Precision

SECONDARY
------------------------------------------------------------------------------------------------------------------------
ROC-AUC
AP lift over prevalence
Log Loss
Brier Score
Recall@5%
Recall@10%
Recall@20%

IMPORTANT
------------------------------------------------------------------------------------------------------------------------
Target performance observed     : NO
K final selected                : NO
Threshold selected              : NO
Final holdout selected          : NO
Classifier trained              : NO
Silver created                  : NO
RAW modified                    : NO

Contract SHA256:
{sha256}

Validation failures             : {failures}
========================================================================================================================
""".strip()


REPORT.write_text(
    report,
    encoding="utf-8",
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 120)
print("FROZEN CONTRACT")
print("=" * 120)

print(
    json.dumps(
        contract,
        indent=4,
        ensure_ascii=False,
    )
)


print()
print("=" * 120)
print("VALIDATION")
print("=" * 120)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("RESULTADO C3.5-B")
print("=" * 120)

print(
    "STATUS                  =",
    summary["status"]
)

print(
    "CONTRACT                =",
    summary["contract_status"]
)

print(
    "K CANDIDATES            =",
    len(K_GRID)
)

print(
    "K GRID                  =",
    K_GRID
)

print(
    "PRIMARY METRIC          = average_precision"
)

print()
print(
    "PERFORMANCE OBSERVED    = NÃO"
)
print(
    "K FINAL                 = NÃO"
)
print(
    "THRESHOLD               = NÃO"
)
print(
    "CLASSIFIER              = NÃO TREINADO"
)
print(
    "SILVER                  = NÃO CRIADA"
)
print(
    "RAW                     = INTACTO"
)


if failures:
    raise SystemExit(2)


print()
print("[PASS] C3.5-B concluído.")
print("[PASS] Protocolo congelado antes de observar performance.")
print("[PASS] Próximo estágio poderá executar a comparação supervisionada.")

