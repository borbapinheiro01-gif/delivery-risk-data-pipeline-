#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.5-A
SUPERVISED TEMPORAL PROTOCOL — LABEL AVAILABILITY AUDIT
===============================================================================

OBJETIVO
--------
Antes de usar o target em qualquer modelo, construir os folds temporais
supervisionados respeitando DUAS condições:

    purchase_j < cutoff_m

e

    delivery_j < cutoff_m

onde cutoff_m é o início do mês futuro m.

Isso garante que o rótulo do pedido histórico j já seria conhecido
quando o modelo do mês m fosse ajustado.

NÃO:
- executa PCA;
- seleciona K;
- treina regressão logística;
- escolhe threshold;
- congela holdout final;
- altera RAW;
- cria Silver.

O target aparece apenas para auditar suporte de classe nos períodos
de teste. Não é usado para selecionar modelo nesta etapa.
===============================================================================
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

MODEL_DIR = (
    ROOT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

MATRIX = (
    ROOT
    / "artifacts"
    / "model_01_order_logistic"
    / "pretraining"
    / "ORDER_CORE_V1_AUDIT_MATRIX.csv"
)

ORDERS = (
    ROOT
    / "data"
    / "raw"
    / "olist"
    / "olist_orders_dataset.csv"
)

ROBUST_JSON = (
    MODEL_DIR
    / "05e_robust_rank_evidence_summary.json"
)

OUT_FOLDS = (
    MODEL_DIR
    / "06a_supervised_temporal_fold_audit.csv"
)

OUT_EXCLUDED = (
    MODEL_DIR
    / "06b_label_availability_exclusions.csv"
)

OUT_VALIDATION = (
    MODEL_DIR
    / "06c_supervised_temporal_protocol_validation.csv"
)

OUT_SUMMARY = (
    MODEL_DIR
    / "06d_supervised_temporal_protocol_summary.json"
)

OUT_REPORT = (
    MODEL_DIR
    / "06e_supervised_temporal_protocol_report.txt"
)


# =============================================================================
# START
# =============================================================================

print()
print("=" * 120)
print("MODEL 01.0-C3.5-A — SUPERVISED TEMPORAL LABEL-AVAILABILITY PROTOCOL")
print("=" * 120)


# =============================================================================
# PREREQUISITES
# =============================================================================

for p in [
    MATRIX,
    ORDERS,
    ROBUST_JSON,
]:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Arquivo ausente: {p}"
        )

    print(
        f"[PASS] {p.name}"
    )


with ROBUST_JSON.open(
    encoding="utf-8"
) as f:

    robust = json.load(f)


if robust.get("status") != "PASS":

    raise RuntimeError(
        "C3.4-D não está PASS."
    )


if robust.get(
    "final_k_selected",
    True
):

    raise RuntimeError(
        "C3.4-D registra K final, "
        "mas esta etapa pressupõe K ainda aberto."
    )


print()
print("[PASS] C3.4-D validado.")
print("[PASS] K continua aberto.")


# =============================================================================
# LOAD
# =============================================================================

m = pd.read_csv(
    MATRIX,
    parse_dates=[
        "order_purchase_timestamp",
    ],
)


orders = pd.read_csv(
    ORDERS,
    usecols=[
        "order_id",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ],
    parse_dates=[
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ],
)


if m["order_id"].duplicated().any():

    raise RuntimeError(
        "ORDER_CORE contém order_id duplicado."
    )


if orders["order_id"].duplicated().any():

    raise RuntimeError(
        "orders RAW contém order_id duplicado."
    )


# =============================================================================
# REATTACH OUTCOME AVAILABILITY TIME
# =============================================================================

x = m.merge(
    orders[
        [
            "order_id",
            "order_delivered_customer_date",
        ]
    ],
    on="order_id",
    how="left",
    validate="one_to_one",
)


if x[
    "order_delivered_customer_date"
].isna().any():

    raise RuntimeError(
        "Há pedido supervisionado sem delivery timestamp."
    )


# =============================================================================
# FULL 30-DAY LEFT HISTORY SUPPORT
# =============================================================================

raw_start = orders[
    "order_purchase_timestamp"
].min()

support_start = (
    raw_start
    +
    pd.Timedelta(
        days=30
    )
)


x[
    "full_support_30d"
] = (
    x[
        "order_purchase_timestamp"
    ]
    >=
    support_start
)


full = x.loc[
    x[
        "full_support_30d"
    ]
].copy()


print()
print("=" * 120)
print("30D SUPPORT")
print("=" * 120)

print(
    "RAW start          :",
    raw_start
)

print(
    "30D support start  :",
    support_start
)

print(
    "Task rows          :",
    f"{len(x):,}"
)

print(
    "Full support rows  :",
    f"{len(full):,}"
)


# =============================================================================
# TEST MONTHS
# =============================================================================

full[
    "purchase_month"
] = (
    full[
        "order_purchase_timestamp"
    ]
    .dt
    .to_period("M")
    .astype(str)
)


months = [
    str(p)
    for p in pd.period_range(
        "2017-04",
        "2018-08",
        freq="M",
    )
]


fold_rows = []
excluded_rows = []


# =============================================================================
# CONSTRUCT STRICT SUPERVISED FOLDS
# =============================================================================

for month in months:

    period = pd.Period(
        month,
        freq="M"
    )

    cutoff = period.start_time


    test = full.loc[
        full[
            "purchase_month"
        ]
        ==
        month
    ].copy()


    purchase_past = full.loc[
        full[
            "order_purchase_timestamp"
        ]
        <
        cutoff
    ].copy()


    train = purchase_past.loc[
        purchase_past[
            "order_delivered_customer_date"
        ]
        <
        cutoff
    ].copy()


    immature = purchase_past.loc[
        purchase_past[
            "order_delivered_customer_date"
        ]
        >=
        cutoff
    ].copy()


    train_pos = int(
        train[
            "late_delivery_calendar_day"
        ].sum()
    )

    train_neg = int(
        (
            train[
                "late_delivery_calendar_day"
            ]
            ==
            0
        ).sum()
    )

    test_pos = int(
        test[
            "late_delivery_calendar_day"
        ].sum()
    )

    test_neg = int(
        (
            test[
                "late_delivery_calendar_day"
            ]
            ==
            0
        ).sum()
    )


    if len(train):

        max_train_purchase = (
            train[
                "order_purchase_timestamp"
            ].max()
        )

        max_train_delivery = (
            train[
                "order_delivered_customer_date"
            ].max()
        )

    else:

        max_train_purchase = pd.NaT
        max_train_delivery = pd.NaT


    fold_rows.append({
        "test_month":
            month,

        "fit_cutoff":
            cutoff,

        "purchase_past_rows":
            len(
                purchase_past
            ),

        "label_available_train_rows":
            len(
                train
            ),

        "excluded_unavailable_label_rows":
            len(
                immature
            ),

        "excluded_unavailable_label_pct":
            (
                100.0
                *
                len(immature)
                /
                len(purchase_past)
                if len(purchase_past)
                else np.nan
            ),

        "train_positive":
            train_pos,

        "train_negative":
            train_neg,

        "train_positive_pct":
            (
                100.0
                *
                train_pos
                /
                len(train)
                if len(train)
                else np.nan
            ),

        "test_rows":
            len(
                test
            ),

        "test_positive":
            test_pos,

        "test_negative":
            test_neg,

        "test_positive_pct":
            (
                100.0
                *
                test_pos
                /
                len(test)
                if len(test)
                else np.nan
            ),

        "max_train_purchase_time":
            max_train_purchase,

        "max_train_delivery_time":
            max_train_delivery,

        "purchase_boundary_valid":
            bool(
                pd.notna(
                    max_train_purchase
                )
                and
                max_train_purchase
                <
                cutoff
            ),

        "label_boundary_valid":
            bool(
                pd.notna(
                    max_train_delivery
                )
                and
                max_train_delivery
                <
                cutoff
            ),
    })


    excluded_rows.append({
        "test_month":
            month,

        "fit_cutoff":
            cutoff,

        "purchase_past_rows":
            len(
                purchase_past
            ),

        "label_available_rows":
            len(
                train
            ),

        "label_not_yet_available_rows":
            len(
                immature
            ),

        "label_not_yet_available_pct":
            (
                100.0
                *
                len(immature)
                /
                len(purchase_past)
                if len(purchase_past)
                else np.nan
            ),
    })


folds = pd.DataFrame(
    fold_rows
)

excluded = pd.DataFrame(
    excluded_rows
)


folds.to_csv(
    OUT_FOLDS,
    index=False
)

excluded.to_csv(
    OUT_EXCLUDED,
    index=False
)


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


def add(
    name,
    ok,
    observed,
    expected,
):

    checks.append({
        "check":
            name,

        "status":
            "PASS"
            if ok
            else "FAIL",

        "observed":
            observed,

        "expected":
            expected,
    })


add(
    "c34d_pass",
    robust.get(
        "status"
    )
    ==
    "PASS",
    robust.get(
        "status"
    ),
    "PASS",
)


add(
    "task_rows",
    len(x)
    ==
    96470,
    len(x),
    96470,
)


add(
    "full_support_30d",
    len(full)
    ==
    96421,
    len(full),
    96421,
)


add(
    "temporal_months",
    len(folds)
    ==
    17,
    len(folds),
    17,
)


add(
    "first_month",
    folds.iloc[0][
        "test_month"
    ]
    ==
    "2017-04",
    folds.iloc[0][
        "test_month"
    ],
    "2017-04",
)


add(
    "last_month",
    folds.iloc[-1][
        "test_month"
    ]
    ==
    "2018-08",
    folds.iloc[-1][
        "test_month"
    ],
    "2018-08",
)


add(
    "all_purchase_boundaries_strict",
    folds[
        "purchase_boundary_valid"
    ].all(),
    int(
        (
            ~folds[
                "purchase_boundary_valid"
            ]
        ).sum()
    ),
    0,
)


add(
    "all_label_boundaries_strict",
    folds[
        "label_boundary_valid"
    ].all(),
    int(
        (
            ~folds[
                "label_boundary_valid"
            ]
        ).sum()
    ),
    0,
)


add(
    "all_train_sets_have_both_classes",
    (
        (
            folds[
                "train_positive"
            ]
            >
            0
        )
        &
        (
            folds[
                "train_negative"
            ]
            >
            0
        )
    ).all(),
    int(
        (
            (
                folds[
                    "train_positive"
                ]
                ==
                0
            )
            |
            (
                folds[
                    "train_negative"
                ]
                ==
                0
            )
        ).sum()
    ),
    0,
)


add(
    "all_test_sets_have_both_classes",
    (
        (
            folds[
                "test_positive"
            ]
            >
            0
        )
        &
        (
            folds[
                "test_negative"
            ]
            >
            0
        )
    ).all(),
    int(
        (
            (
                folds[
                    "test_positive"
                ]
                ==
                0
            )
            |
            (
                folds[
                    "test_negative"
                ]
                ==
                0
            )
        ).sum()
    ),
    0,
)


add(
    "no_final_holdout_frozen",
    True,
    False,
    False,
)


add(
    "no_model_trained",
    True,
    False,
    False,
)


validation = pd.DataFrame(
    checks
)

validation.to_csv(
    OUT_VALIDATION,
    index=False
)


failures = int(
    validation[
        "status"
    ].eq(
        "FAIL"
    ).sum()
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = {
    "step":
        "MODEL_01_0_C3_5A_SUPERVISED_TEMPORAL_PROTOCOL",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "representation":
        "RAW_30D",

    "prediction_time":
        "order_purchase_timestamp",

    "fit_policy":
        "MONTH_START_EXPANDING_HISTORY",

    "training_purchase_rule":
        "order_purchase_timestamp < fit_cutoff",

    "training_label_availability_rule":
        "order_delivered_customer_date < fit_cutoff",

    "test_policy":
        "CURRENT_CALENDAR_MONTH",

    "task_rows":
        len(x),

    "full_support_30d_rows":
        len(full),

    "temporal_test_months":
        len(folds),

    "first_test_month":
        folds.iloc[0][
            "test_month"
        ],

    "last_test_month":
        folds.iloc[-1][
            "test_month"
        ],

    "total_label_unavailable_exclusions_across_folds":
        int(
            excluded[
                "label_not_yet_available_rows"
            ].sum()
        ),

    "max_label_unavailable_pct":
        float(
            excluded[
                "label_not_yet_available_pct"
            ].max()
        ),

    "target_used_for_model_selection":
        False,

    "final_holdout_selected":
        False,

    "final_k_selected":
        False,

    "pca_executed":
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


OUT_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

report = f"""
========================================================================================================================
MODEL 01.0-C3.5-A — SUPERVISED TEMPORAL PROTOCOL
========================================================================================================================

STATUS
------------------------------------------------------------------------------------------------------------------------
{summary["status"]}

TEMPORAL CONTRACT
------------------------------------------------------------------------------------------------------------------------
Prediction time:
    order_purchase_timestamp

For test month m:

    fit_cutoff = beginning of month m

Training purchase requirement:

    purchase_j < fit_cutoff

Training label-availability requirement:

    delivery_j < fit_cutoff

Thus the training procedure cannot use a historical order whose final
delivery outcome was still unknown at model-fit time.

DATA
------------------------------------------------------------------------------------------------------------------------
Task rows                       : {len(x):,}
Full 30D support rows           : {len(full):,}
Temporal test months            : {len(folds)}
First test month                : {folds.iloc[0]["test_month"]}
Last test month                 : {folds.iloc[-1]["test_month"]}

GOVERNANCE
------------------------------------------------------------------------------------------------------------------------
Final holdout frozen            : NO
Final K selected                : NO
PCA executed                    : NO
Classifier trained              : NO
Silver created                  : NO
RAW modified                    : NO

Validation failures             : {failures}
========================================================================================================================
""".strip()


OUT_REPORT.write_text(
    report,
    encoding="utf-8",
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 120)
print("SUPERVISED TEMPORAL FOLDS")
print("=" * 120)

display_cols = [
    "test_month",
    "fit_cutoff",
    "purchase_past_rows",
    "label_available_train_rows",
    "excluded_unavailable_label_rows",
    "excluded_unavailable_label_pct",
    "train_positive",
    "train_negative",
    "train_positive_pct",
    "test_rows",
    "test_positive",
    "test_negative",
    "test_positive_pct",
]

print(
    folds[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda z: f"{z:.6f}",
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
print("RESULTADO C3.5-A")
print("=" * 120)

print(
    "STATUS                    =",
    summary[
        "status"
    ]
)

print(
    "TEMPORAL MONTHS           =",
    len(folds)
)

print(
    "LABEL AVAILABILITY RULE   = delivery_j < fit_cutoff"
)

print(
    "MAX EXCLUSION PCT         =",
    f'{summary["max_label_unavailable_pct"]:.6f}%'
)

print()
print(
    "FINAL HOLDOUT             = NÃO"
)
print(
    "K FINAL                   = NÃO"
)
print(
    "PCA                       = NÃO"
)
print(
    "CLASSIFIER                = NÃO"
)
print(
    "SILVER                    = NÃO"
)
print(
    "RAW                       = INTACTO"
)


if failures:

    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.5-A concluído."
)
print(
    "[PASS] Folds supervisionados possuem label disponível no passado."
)
print(
    "[PASS] Parar antes de qualquer treinamento."
)

