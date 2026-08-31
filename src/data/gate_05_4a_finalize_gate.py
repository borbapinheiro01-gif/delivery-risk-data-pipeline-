#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DQ GATE 05.4-A — FINAL POINT-IN-TIME / LEAKAGE AUDIT
Delivery Risk Intelligence Platform
===============================================================================

Consolida:

05.1   Raw Point-in-Time Registry
05.2-A ORDER_CORE_V1
05.2-B Modular Feature Contract
05.3   Seller History Point-in-Time

PASS significa:
- contrato temporal validado;
- future/target bloqueados;
- HOLD preservado;
- seller history validado materialmente;
- MODEL_01 pode ser construído e avaliado temporalmente.

PASS NÃO significa:
- MODEL_01 promovido;
- MODEL_02 liberado;
- Silver criada;
- payments liberados;
- shipping_limit_date liberado;
- geo resolvida.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

OUT = (
    ROOT
    / "reports"
    / "data_quality"
    / "gate_05_point_in_time"
)

FILES = {
    "05.1":
        OUT / "dq_gate_05_1_summary.json",

    "05.2-A":
        OUT / "dq_gate_05_2a_summary.json",

    "05.2-B":
        OUT / "dq_gate_05_2b_summary.json",

    "05.3":
        OUT / "dq_gate_05_3_summary.json",
}

LABEL_CONTRACT = (
    ROOT
    / "contracts"
    / "DELIVERY_RISK_LABEL_CONTRACT_V1.json"
)

ORDER_CORE = (
    ROOT
    / "configs"
    / "order_core_v1_feature_contract.json"
)

PROTOCOL = (
    ROOT
    / "configs"
    / "model_validation_protocol_v1.json"
)

PIT_REGISTRY = (
    OUT
    / "01_raw_feature_point_in_time_registry.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path):

    if not path.exists():
        raise SystemExit(
            f"[ERRO] Arquivo obrigatório ausente:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def add_check(
    rows,
    check_id,
    dimension,
    condition,
    observed,
    expected,
    severity="CRITICAL",
    details=""
):

    rows.append({
        "check_id":
            check_id,

        "dimension":
            dimension,

        "severity":
            severity,

        "status":
            "PASS" if condition else "FAIL",

        "observed":
            observed,

        "expected":
            expected,

        "details":
            details
    })


# =============================================================================
# LOAD
# =============================================================================

summaries = {
    name:
        load_json(path)
    for name, path
    in FILES.items()
}

label = load_json(
    LABEL_CONTRACT
)

order_core = load_json(
    ORDER_CORE
)

protocol = load_json(
    PROTOCOL
)


if not PIT_REGISTRY.exists():
    raise SystemExit(
        f"[ERRO] PIT Registry ausente:\n{PIT_REGISTRY}"
    )


pit = pd.read_csv(
    PIT_REGISTRY,
    low_memory=False
)


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


# -----------------------------------------------------------------------------
# A. ETAPAS 05.1–05.3
# -----------------------------------------------------------------------------

for name in [
    "05.1",
    "05.2-A",
    "05.2-B",
    "05.3",
]:

    status = summaries[
        name
    ].get(
        "step_status"
    )

    add_check(
        checks,
        f"PIT-STEP-{name}",
        "prerequisite",
        status == "PASS",
        status,
        "PASS"
    )


# -----------------------------------------------------------------------------
# B. LABEL / t0
# -----------------------------------------------------------------------------

add_check(
    checks,
    "PIT-LABEL-001",
    "label_contract",
    label.get(
        "status"
    ) == "FROZEN_FOR_MODELING",
    label.get(
        "status"
    ),
    "FROZEN_FOR_MODELING"
)


add_check(
    checks,
    "PIT-T0-001",
    "prediction_time",
    label.get(
        "prediction_time"
    ) == "order_purchase_timestamp",
    label.get(
        "prediction_time"
    ),
    "order_purchase_timestamp"
)


# -----------------------------------------------------------------------------
# C. RAW REGISTRY
# -----------------------------------------------------------------------------

s051 = summaries[
    "05.1"
]


add_check(
    checks,
    "PIT-RAW-001",
    "raw_registry",
    s051.get(
        "raw_columns_audited"
    ) == 52,
    s051.get(
        "raw_columns_audited"
    ),
    52
)


add_check(
    checks,
    "PIT-RAW-002",
    "raw_registry",
    s051.get(
        "classified_columns"
    ) == 52,
    s051.get(
        "classified_columns"
    ),
    52
)


add_check(
    checks,
    "PIT-RAW-003",
    "raw_registry",
    s051.get(
        "review_required"
    ) == 0,
    s051.get(
        "review_required"
    ),
    0
)


# -----------------------------------------------------------------------------
# D. FUTURE / TARGET NÃO PODE SER FEATURE DIRETA
# -----------------------------------------------------------------------------

future_target = pit[
    pit[
        "point_in_time_status"
    ]
    .isin([
        "FORBIDDEN_FUTURE",
        "TARGET_ONLY"
    ])
]


future_target_direct = int(
    future_target[
        "direct_model_feature"
    ]
    .fillna(False)
    .astype(bool)
    .sum()
)


add_check(
    checks,
    "PIT-LEAK-001",
    "future_target_policy",
    future_target_direct == 0,
    future_target_direct,
    0
)


# -----------------------------------------------------------------------------
# E. HOLD NÃO PODE SER FEATURE DIRETA
# -----------------------------------------------------------------------------

holds = pit[
    pit[
        "point_in_time_status"
    ]
    .isin([
        "HOLD_PROVENANCE",
        "HOLD_DATA_QUALITY"
    ])
]


hold_direct = int(
    holds[
        "direct_model_feature"
    ]
    .fillna(False)
    .astype(bool)
    .sum()
)


add_check(
    checks,
    "PIT-HOLD-001",
    "hold_policy",
    hold_direct == 0,
    hold_direct,
    0
)


# -----------------------------------------------------------------------------
# F. MODEL 01
# -----------------------------------------------------------------------------

s052a = summaries[
    "05.2-A"
]


feature_count = len(
    order_core.get(
        "features",
        []
    )
)


add_check(
    checks,
    "PIT-M01-001",
    "model01_contract",
    feature_count == 13,
    feature_count,
    13
)


for field, cid in [
    (
        "missing_sources",
        "PIT-M01-002"
    ),
    (
        "unsafe_dependencies",
        "PIT-M01-003"
    ),
    (
        "forbidden_target_or_hold_dependencies",
        "PIT-M01-004"
    ),
    (
        "validation_failures",
        "PIT-M01-005"
    ),
]:

    value = s052a.get(
        field
    )

    add_check(
        checks,
        cid,
        "model01_contract",
        value == 0,
        value,
        0
    )


# -----------------------------------------------------------------------------
# G. MODULAR CONTRACT
# -----------------------------------------------------------------------------

s052b = summaries[
    "05.2-B"
]


for field, cid in [
    (
        "missing_sources",
        "PIT-MOD-001"
    ),
    (
        "hold_marked_safe",
        "PIT-MOD-002"
    ),
    (
        "target_current_feature_misuse",
        "PIT-MOD-003"
    ),
    (
        "historical_without_availability_rule",
        "PIT-MOD-004"
    ),
    (
        "payment_not_hold",
        "PIT-MOD-005"
    ),
    (
        "unlocked_future_modules",
        "PIT-MOD-006"
    ),
    (
        "validation_failures",
        "PIT-MOD-007"
    ),
]:

    value = s052b.get(
        field
    )

    add_check(
        checks,
        cid,
        "modular_contract",
        value == 0,
        value,
        0
    )


# -----------------------------------------------------------------------------
# H. SELLER HISTORY
# -----------------------------------------------------------------------------

s053 = summaries[
    "05.3"
]


expected_supervised = int(
    label[
        "eligible_supervised_population"
    ][
        "orders"
    ]
)


add_check(
    checks,
    "PIT-SELLER-001",
    "seller_history",
    s053.get(
        "supervised_orders"
    ) == expected_supervised,
    s053.get(
        "supervised_orders"
    ),
    expected_supervised
)


for field, cid in [
    (
        "purchase_time_leakage",
        "PIT-SELLER-002"
    ),
    (
        "outcome_time_leakage",
        "PIT-SELLER-003"
    ),
    (
        "bruteforce_mismatches",
        "PIT-SELLER-004"
    ),
    (
        "validation_failures",
        "PIT-SELLER-005"
    ),
]:

    value = s053.get(
        field
    )

    add_check(
        checks,
        cid,
        "seller_history",
        value == 0,
        value,
        0
    )


# -----------------------------------------------------------------------------
# I. MODEL SEQUENCE
# -----------------------------------------------------------------------------

sequence = protocol.get(
    "sequence",
    []
)


try:
    gate_idx = sequence.index(
        "GATE_05_POINT_IN_TIME"
    )
except ValueError:
    gate_idx = -1


try:
    model01_idx = sequence.index(
        "MODEL_01_ORDER_LOGISTIC"
    )
except ValueError:
    model01_idx = -1


sequence_ok = (
    gate_idx >= 0
    and
    model01_idx == gate_idx + 1
)


add_check(
    checks,
    "PIT-PROTOCOL-001",
    "sequential_validation",
    sequence_ok,
    f"GATE={gate_idx}; MODEL01={model01_idx}",
    "MODEL_01 immediately after GATE_05"
)


add_check(
    checks,
    "PIT-PROTOCOL-002",
    "sequential_validation",
    protocol.get(
        "principle"
    ) == "ONE_MODEL_AT_A_TIME",
    protocol.get(
        "principle"
    ),
    "ONE_MODEL_AT_A_TIME"
)


# =============================================================================
# RESULT
# =============================================================================

validation = pd.DataFrame(
    checks
)


critical_failures = int(
    (
        validation[
            "severity"
        ]
        .eq(
            "CRITICAL"
        )
        &
        validation[
            "status"
        ]
        .eq(
            "FAIL"
        )
    )
    .sum()
)


gate_status = (
    "PASS"
    if critical_failures == 0
    else "FAIL"
)


model01_unlocked = (
    gate_status == "PASS"
)


# =============================================================================
# MODEL STATUS
# =============================================================================

model_status = pd.DataFrame([
    {
        "component":
            "MODEL_01_ORDER_LOGISTIC",

        "status":
            (
                "UNLOCKED_FOR_TEMPORAL_VALIDATION"
                if model01_unlocked
                else "BLOCKED"
            ),

        "reason":
            "ORDER_CORE_V1 passed Gate 05."
    },

    {
        "component":
            "MODEL_02_ORDER_CATBOOST",

        "status":
            "LOCKED",

        "reason":
            "Wait for MODEL_01 PROMOTE/HOLD/REJECT."
    },

    {
        "component":
            "MODEL_03_SELLER_EXPERT",

        "status":
            "LOCKED",

        "reason":
            (
                "PIT infrastructure validated, "
                "predictive utility not yet tested."
            )
    },

    {
        "component":
            "MODEL_04_GEO_PROMISE_EXPERT",

        "status":
            "LOCKED",

        "reason":
            "Geolocation treatment still unresolved."
    },

    {
        "component":
            "MODEL_05_TEMPORAL_EXPERT",

        "status":
            "LOCKED",

        "reason":
            "Sequential validation protocol."
    },

    {
        "component":
            "PAYMENT_FEATURES",

        "status":
            "BLOCKED",

        "reason":
            "Point-in-time provenance unresolved."
    },

    {
        "component":
            "SHIPPING_LIMIT_FEATURES",

        "status":
            "BLOCKED",

        "reason":
            "shipping_limit_date provenance unresolved."
    },

    {
        "component":
            "GEO_COORDINATE_FEATURES",

        "status":
            "BLOCKED",

        "reason":
            "ZIP/geolocation consolidation unresolved."
    }
])


# =============================================================================
# SAVE
# =============================================================================

validation.to_csv(
    OUT
    / "04a_gate_05_final_validation.csv",
    index=False
)


model_status.to_csv(
    OUT
    / "04b_model_unlock_status.csv",
    index=False
)


summary = {
    "gate":
        "DQ_GATE_05_POINT_IN_TIME_AVAILABILITY_AND_LEAKAGE",

    "status":
        gate_status,

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "prediction_time":
        "order_purchase_timestamp",

    "steps": {
        "05.1":
            summaries[
                "05.1"
            ].get(
                "step_status"
            ),

        "05.2-A":
            summaries[
                "05.2-A"
            ].get(
                "step_status"
            ),

        "05.2-B":
            summaries[
                "05.2-B"
            ].get(
                "step_status"
            ),

        "05.3":
            summaries[
                "05.3"
            ].get(
                "step_status"
            )
    },

    "raw_columns_audited":
        s051.get(
            "raw_columns_audited"
        ),

    "hold_columns":
        int(
            len(
                holds
            )
        ),

    "future_target_columns":
        int(
            len(
                future_target
            )
        ),

    "model_01_feature_set":
        "ORDER_CORE_V1",

    "model_01_features":
        feature_count,

    "seller_history_order_seller_rows":
        s053.get(
            "order_seller_rows"
        ),

    "seller_history_unique_sellers":
        s053.get(
            "unique_current_sellers"
        ),

    "seller_history_purchase_leakage":
        s053.get(
            "purchase_time_leakage"
        ),

    "seller_history_outcome_leakage":
        s053.get(
            "outcome_time_leakage"
        ),

    "bruteforce_mismatches":
        s053.get(
            "bruteforce_mismatches"
        ),

    "critical_failures":
        critical_failures,

    "model_01_unlocked":
        model01_unlocked,

    "model_01_unlock_scope":
        (
            "FEATURE_MATRIX_BUILD_AND_TEMPORAL_VALIDATION_ONLY"
            if model01_unlocked
            else "NONE"
        ),

    "model_02_unlocked":
        False,

    "seller_expert_unlocked":
        False,

    "payment_features_allowed":
        False,

    "shipping_limit_allowed":
        False,

    "geo_coordinates_allowed":
        False,

    "raw_modified":
        False,

    "silver_created":
        False,

    "model_trained":
        False
}


with (
    OUT
    / "dq_gate_05_summary.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# =============================================================================
# REPORT
# =============================================================================

report = f"""
========================================================================================================
DQ GATE 05 — POINT-IN-TIME AVAILABILITY & LEAKAGE
========================================================================================================

STATUS                                  : {gate_status}

PREDICTION TIME                         : order_purchase_timestamp

STEPS
--------------------------------------------------------------------------------------------------------
05.1 Raw PIT Registry                   : {summaries['05.1'].get('step_status')}
05.2-A ORDER_CORE_V1                    : {summaries['05.2-A'].get('step_status')}
05.2-B Modular Contract                 : {summaries['05.2-B'].get('step_status')}
05.3 Seller History PIT                 : {summaries['05.3'].get('step_status')}

RAW COLUMNS AUDITED                     : {s051.get('raw_columns_audited')}
FUTURE / TARGET COLUMNS                 : {len(future_target)}
HOLD COLUMNS                            : {len(holds)}

MODEL 01
--------------------------------------------------------------------------------------------------------
FEATURE SET                             : ORDER_CORE_V1
FEATURES                                : {feature_count}

SELLER HISTORY
--------------------------------------------------------------------------------------------------------
ORDER-SELLER ROWS                       : {s053.get('order_seller_rows')}
UNIQUE SELLERS                          : {s053.get('unique_current_sellers')}
PURCHASE-TIME LEAKAGE                   : {s053.get('purchase_time_leakage')}
OUTCOME-TIME LEAKAGE                    : {s053.get('outcome_time_leakage')}
BRUTE-FORCE MISMATCHES                  : {s053.get('bruteforce_mismatches')}

FINAL VALIDATION
--------------------------------------------------------------------------------------------------------
CRITICAL FAILURES                       : {critical_failures}

MODEL STATUS
--------------------------------------------------------------------------------------------------------
MODEL_01_ORDER_LOGISTIC                 : {"UNLOCKED FOR TEMPORAL VALIDATION" if model01_unlocked else "BLOCKED"}
MODEL_02_ORDER_CATBOOST                 : LOCKED
MODEL_03_SELLER_EXPERT                  : LOCKED
MODEL_04_GEO_PROMISE_EXPERT             : LOCKED
MODEL_05_TEMPORAL_EXPERT                : LOCKED

PAYMENT                                 : BLOCKED
SHIPPING_LIMIT_DATE                     : BLOCKED
GEO COORDINATES                         : BLOCKED

IMPORTANT
--------------------------------------------------------------------------------------------------------
Gate 05 PASS validates the point-in-time/leakage contract.

It does NOT prove MODEL_01 is useful.

MODEL_01 must next undergo temporal out-of-sample evaluation and receive:

PROMOTE
HOLD
or
REJECT

before MODEL_02 can be released.

RAW modified                            : NO
Silver created                          : NO
Model trained                           : NO

========================================================================================================
"""


(
    OUT
    / "DQ_GATE_05_POINT_IN_TIME_REPORT.txt"
).write_text(
    report.strip()
    +
    "\n",
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()
print("=" * 108)
print("DQ GATE 05 — FINAL VALIDATION")
print("=" * 108)

print(
    validation.to_string(
        index=False,
        max_rows=None,
        max_cols=None
    )
)


print()
print("=" * 108)
print("MODEL / MODULE STATUS")
print("=" * 108)

print(
    model_status.to_string(
        index=False,
        max_rows=None,
        max_cols=None
    )
)


print()
print("=" * 108)
print("RESULTADO")
print("=" * 108)

print(
    f"GATE STATUS              : {gate_status}"
)

print(
    f"CRITICAL FAILURES        : {critical_failures}"
)

print(
    f"FUTURE/TARGET COLUMNS    : {len(future_target)}"
)

print(
    f"HOLD COLUMNS             : {len(holds)}"
)

print(
    f"MODEL 01 FEATURES        : {feature_count}"
)

print(
    "MODEL 01 UNLOCKED        : "
    +
    (
        "SIM — TEMPORAL VALIDATION ONLY"
        if model01_unlocked
        else "NÃO"
    )
)

print(
    "MODEL 02 UNLOCKED        : NÃO"
)

print(
    "RAW MODIFIED             : NÃO"
)

print(
    "SILVER CREATED           : NÃO"
)

print(
    "MODEL TRAINED            : NÃO"
)


if gate_status != "PASS":
    sys.exit(2)


print()
print(
    "[PASS] DQ GATE 05 APROVADO."
)

print(
    "MODEL_01 liberado somente para construção "
    "da matriz e validação temporal."
)

print(
    "MODEL_02 continua bloqueado."
)

