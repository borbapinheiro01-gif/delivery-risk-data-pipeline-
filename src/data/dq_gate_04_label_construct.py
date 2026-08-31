#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DQ GATE 04 — LABEL / CONSTRUCT VALIDITY
Delivery Risk Intelligence Platform
===============================================================================

OBJETIVO
--------
Auditar a definição matemática do target de atraso.

COMPARA
-------
Y_ts =
    1[delivery_timestamp > estimated_delivery_timestamp]

Y_day =
    1[delivery_calendar_date > estimated_delivery_calendar_date]

Também calcula uma definição de sensibilidade:

Y_day_grace1 =
    1[delivery_calendar_date >
      estimated_delivery_calendar_date + 1 dia]

VALIDA
------
1. observabilidade do outcome;
2. coerência temporal;
3. semântica temporal da data prometida;
4. divergência entre definições do label;
5. concentração das divergências na fronteira;
6. distribuição da margem de atraso;
7. prevalência da classe;
8. estabilidade temporal;
9. janela de observação;
10. possíveis observações censuradas/incompletas.

NÃO FAZ
-------
- não altera RAW;
- não cria Silver;
- não imputa;
- não remove;
- não treina modelo;
- não decide automaticamente um SLA empresarial inexistente.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

RAW = (
    PROJECT
    / "data"
    / "raw"
    / "olist"
)

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG = (
    PROJECT
    / "configs"
    / "dq_gate_04_label_construct.json"
)


# =============================================================================
# PREREQUISITES
# =============================================================================

PREREQUISITES = {
    "DQ_GATE_01":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_01_structural"
        / "dq_gate_01_summary.json",

    "DQ_GATE_02":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_02_semantic"
        / "dq_gate_02_summary.json",

    "DQ_GATE_03":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_03_statistical"
        / "dq_gate_03_summary.json",

    "DQ_GATE_03B":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_03b_conditional_task"
        / "dq_gate_03b_summary.json",

    "REGISTRY_1_1":
        PROJECT
        / "metadata"
        / "registry_v1_1"
        / "registry_v1_1_manifest.json",
}


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


for name, path in PREREQUISITES.items():

    if not path.exists():

        print(
            f"[BLOQUEADO] Pré-requisito ausente: {name}"
        )

        print(path)

        sys.exit(2)


gate1 = load_json(
    PREREQUISITES["DQ_GATE_01"]
)

gate2 = load_json(
    PREREQUISITES["DQ_GATE_02"]
)

gate3 = load_json(
    PREREQUISITES["DQ_GATE_03"]
)

gate3b = load_json(
    PREREQUISITES["DQ_GATE_03B"]
)

registry = load_json(
    PREREQUISITES["REGISTRY_1_1"]
)


for name, obj in [
    ("DQ_GATE_01", gate1),
    ("DQ_GATE_02", gate2),
    ("DQ_GATE_03", gate3),
    ("DQ_GATE_03B", gate3b),
]:

    if obj.get(
        "status"
    ) != "PASS":

        print(
            f"[BLOQUEADO] {name} != PASS"
        )

        sys.exit(2)


if registry.get(
    "internal_registry_status"
) != "PASS":

    print(
        "[BLOQUEADO] Registry 1.1 != PASS"
    )

    sys.exit(2)


# =============================================================================
# CONFIG
# =============================================================================

with CONFIG.open(
    "r",
    encoding="utf-8"
) as f:

    CONTRACT = json.load(f)


# =============================================================================
# LOAD ORDERS
# =============================================================================

ORDERS_FILE = (
    RAW
    / "olist_orders_dataset.csv"
)


if not ORDERS_FILE.exists():

    raise SystemExit(
        f"[ERRO] Arquivo não encontrado: {ORDERS_FILE}"
    )


orders = pd.read_csv(
    ORDERS_FILE,
    low_memory=False
)


# =============================================================================
# HELPERS
# =============================================================================

def pct(
    numerator,
    denominator
):

    if denominator == 0:

        return np.nan

    return (
        100.0
        *
        numerator
        /
        denominator
    )


def wilson_interval(
    k,
    n,
    z=1.959963984540054
):

    if n == 0:

        return (
            np.nan,
            np.nan
        )

    p = k / n

    denom = (
        1
        +
        z * z / n
    )

    center = (
        p
        +
        z * z / (2 * n)
    ) / denom

    half = (
        z
        *
        math.sqrt(
            p * (1 - p) / n
            +
            z * z / (4 * n * n)
        )
        /
        denom
    )

    return (
        center - half,
        center + half
    )


results = []


def add_result(
    check_id,
    dimension,
    severity,
    status,
    affected,
    denominator,
    observed,
    expected,
    details=""
):

    results.append(
        {
            "gate":
                "DQ_GATE_04_LABEL_CONSTRUCT_VALIDITY",

            "check_id":
                check_id,

            "dimension":
                dimension,

            "severity":
                severity,

            "status":
                status,

            "affected":
                int(
                    affected
                ),

            "denominator":
                int(
                    denominator
                ),

            "affected_pct":
                pct(
                    affected,
                    denominator
                ),

            "observed":
                str(
                    observed
                ),

            "expected":
                str(
                    expected
                ),

            "details":
                str(
                    details
                )
        }
    )


# =============================================================================
# PARSE TIMESTAMPS
# =============================================================================

TS_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


parsed = {}


for col in TS_COLUMNS:

    parsed[
        col
    ] = pd.to_datetime(
        orders[
            col
        ],
        errors="coerce"
    )


parse_failures = {}


for col in TS_COLUMNS:

    parse_failures[
        col
    ] = int(
        (
            orders[
                col
            ]
            .notna()
            &
            parsed[
                col
            ]
            .isna()
        )
        .sum()
    )


purchase = parsed[
    "order_purchase_timestamp"
]

estimated = parsed[
    "order_estimated_delivery_date"
]

actual = parsed[
    "order_delivered_customer_date"
]

carrier = parsed[
    "order_delivered_carrier_date"
]


# =============================================================================
# TASK COHORT
# =============================================================================

task_mask = (
    orders[
        "order_status"
    ]
    .eq(
        "delivered"
    )
    &
    purchase.notna()
    &
    estimated.notna()
    &
    actual.notna()
)


task = orders.loc[
    task_mask
].copy()


task[
    "purchase_dt"
] = purchase.loc[
    task_mask
]


task[
    "estimated_dt"
] = estimated.loc[
    task_mask
]


task[
    "actual_dt"
] = actual.loc[
    task_mask
]


task[
    "purchase_month"
] = (
    task[
        "purchase_dt"
    ]
    .dt
    .to_period(
        "M"
    )
    .astype(str)
)


N_SOURCE = len(
    orders
)

N_TASK = len(
    task
)


# =============================================================================
# TEMPORAL CONSISTENCY
# =============================================================================

estimated_before_purchase = int(
    (
        task[
            "estimated_dt"
        ]
        <
        task[
            "purchase_dt"
        ]
    )
    .sum()
)


actual_before_purchase = int(
    (
        task[
            "actual_dt"
        ]
        <
        task[
            "purchase_dt"
        ]
    )
    .sum()
)


delivered_status_missing_actual = int(
    (
        orders[
            "order_status"
        ]
        .eq(
            "delivered"
        )
        &
        actual.isna()
    )
    .sum()
)


non_delivered_with_actual = int(
    (
        ~orders[
            "order_status"
        ]
        .eq(
            "delivered"
        )
        &
        actual.notna()
    )
    .sum()
)


# =============================================================================
# LABELS
# =============================================================================

task[
    "estimated_calendar_date"
] = (
    task[
        "estimated_dt"
    ]
    .dt
    .normalize()
)


task[
    "actual_calendar_date"
] = (
    task[
        "actual_dt"
    ]
    .dt
    .normalize()
)


task[
    "label_timestamp_strict"
] = (
    task[
        "actual_dt"
    ]
    >
    task[
        "estimated_dt"
    ]
).astype(
    "int8"
)


task[
    "label_calendar_day"
] = (
    task[
        "actual_calendar_date"
    ]
    >
    task[
        "estimated_calendar_date"
    ]
).astype(
    "int8"
)


task[
    "label_calendar_day_plus_1_grace"
] = (
    task[
        "actual_calendar_date"
    ]
    >
    (
        task[
            "estimated_calendar_date"
        ]
        +
        pd.Timedelta(
            days=1
        )
    )
).astype(
    "int8"
)


# =============================================================================
# LATENESS MARGINS
# =============================================================================

task[
    "lateness_hours"
] = (
    task[
        "actual_dt"
    ]
    -
    task[
        "estimated_dt"
    ]
).dt.total_seconds() / 3600.0


task[
    "lateness_calendar_days"
] = (
    task[
        "actual_calendar_date"
    ]
    -
    task[
        "estimated_calendar_date"
    ]
).dt.days


task[
    "same_calendar_day"
] = (
    task[
        "actual_calendar_date"
    ]
    ==
    task[
        "estimated_calendar_date"
    ]
)


task[
    "label_disagreement"
] = (
    task[
        "label_timestamp_strict"
    ]
    !=
    task[
        "label_calendar_day"
    ]
)


# =============================================================================
# SEMÂNTICA DO TIMESTAMP PROMETIDO
# =============================================================================

estimated_time = (
    task[
        "estimated_dt"
    ]
    .dt
    .strftime(
        "%H:%M:%S"
    )
)


estimated_midnight_count = int(
    (
        estimated_time
        ==
        "00:00:00"
    )
    .sum()
)


estimated_midnight_pct = pct(
    estimated_midnight_count,
    N_TASK
)


estimated_unique_times = int(
    estimated_time.nunique()
)


actual_time = (
    task[
        "actual_dt"
    ]
    .dt
    .strftime(
        "%H:%M:%S"
    )
)


actual_midnight_pct = pct(
    int(
        (
            actual_time
            ==
            "00:00:00"
        )
        .sum()
    ),
    N_TASK
)


time_semantics = pd.DataFrame(
    [
        {
            "variable":
                "order_estimated_delivery_date",

            "task_rows":
                N_TASK,

            "unique_time_of_day_values":
                estimated_unique_times,

            "midnight_count":
                estimated_midnight_count,

            "midnight_pct":
                estimated_midnight_pct,

            "interpretation":
                (
                    "If nearly all values are midnight, "
                    "the field behaves as a calendar-date promise "
                    "rather than an exact time-of-day SLA."
                )
        },

        {
            "variable":
                "order_delivered_customer_date",

            "task_rows":
                N_TASK,

            "unique_time_of_day_values":
                int(
                    actual_time.nunique()
                ),

            "midnight_count":
                int(
                    (
                        actual_time
                        ==
                        "00:00:00"
                    )
                    .sum()
                ),

            "midnight_pct":
                actual_midnight_pct,

            "interpretation":
                "Actual delivery contains operational timestamp information."
        }
    ]
)


time_semantics.to_csv(
    OUT
    /
    "01_timestamp_semantics.csv",
    index=False
)


# =============================================================================
# LABEL COMPARISON
# =============================================================================

label_specs = [
    (
        "TIMESTAMP_STRICT",
        "label_timestamp_strict",
        "candidate"
    ),

    (
        "CALENDAR_DAY",
        "label_calendar_day",
        "candidate"
    ),

    (
        "CALENDAR_DAY_PLUS_1_GRACE",
        "label_calendar_day_plus_1_grace",
        "sensitivity_only"
    )
]


label_rows = []


for name, column, role in label_specs:

    positives = int(
        task[
            column
        ]
        .sum()
    )

    negatives = (
        N_TASK
        -
        positives
    )

    rate = (
        positives
        /
        N_TASK
        if N_TASK
        else np.nan
    )

    lower, upper = wilson_interval(
        positives,
        N_TASK
    )

    majority_accuracy = (
        max(
            positives,
            negatives
        )
        /
        N_TASK
        if N_TASK
        else np.nan
    )


    label_rows.append(
        {
            "label_definition":
                name,

            "role":
                role,

            "orders":
                N_TASK,

            "late_orders":
                positives,

            "on_time_orders":
                negatives,

            "late_rate_pct":
                100 * rate,

            "late_rate_ci95_low_pct":
                100 * lower,

            "late_rate_ci95_high_pct":
                100 * upper,

            "no_skill_pr_auc":
                rate,

            "majority_class_accuracy":
                majority_accuracy
        }
    )


label_comparison = pd.DataFrame(
    label_rows
)


label_comparison.to_csv(
    OUT
    /
    "02_label_definition_comparison.csv",
    index=False
)


# =============================================================================
# CONFUSION BETWEEN LABEL DEFINITIONS
# =============================================================================

ct = pd.crosstab(
    task[
        "label_timestamp_strict"
    ],
    task[
        "label_calendar_day"
    ]
).reindex(
    index=[
        0,
        1
    ],
    columns=[
        0,
        1
    ],
    fill_value=0
)


confusion_rows = []


for strict_value in [
    0,
    1
]:

    for calendar_value in [
        0,
        1
    ]:

        count = int(
            ct.loc[
                strict_value,
                calendar_value
            ]
        )

        confusion_rows.append(
            {
                "timestamp_strict_label":
                    strict_value,

                "calendar_day_label":
                    calendar_value,

                "count":
                    count,

                "pct_task":
                    pct(
                        count,
                        N_TASK
                    )
            }
        )


confusion_df = pd.DataFrame(
    confusion_rows
)


confusion_df.to_csv(
    OUT
    /
    "03_label_confusion_matrix.csv",
    index=False
)


# =============================================================================
# DISAGREEMENT
# =============================================================================

disagreement = task[
    task[
        "label_disagreement"
    ]
].copy()


N_DISAGREE = len(
    disagreement
)


same_day_disagreements = int(
    disagreement[
        "same_calendar_day"
    ]
    .sum()
)


same_day_disagreement_pct = pct(
    same_day_disagreements,
    N_DISAGREE
)


strict_late_calendar_ontime = int(
    (
        (
            disagreement[
                "label_timestamp_strict"
            ]
            ==
            1
        )
        &
        (
            disagreement[
                "label_calendar_day"
            ]
            ==
            0
        )
    )
    .sum()
)


calendar_late_strict_ontime = int(
    (
        (
            disagreement[
                "label_timestamp_strict"
            ]
            ==
            0
        )
        &
        (
            disagreement[
                "label_calendar_day"
            ]
            ==
            1
        )
    )
    .sum()
)


disagreement[
    [
        "order_id",
        "purchase_dt",
        "estimated_dt",
        "actual_dt",
        "lateness_hours",
        "lateness_calendar_days",
        "same_calendar_day",
        "label_timestamp_strict",
        "label_calendar_day",
    ]
].to_csv(
    OUT
    /
    "04_label_disagreement_cases.csv",
    index=False
)


# =============================================================================
# MARGIN PROFILE
# =============================================================================

def profile_series(
    name,
    series
):

    s = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    quantiles = {
        "min":
            s.min(),

        "p01":
            s.quantile(
                0.01
            ),

        "p05":
            s.quantile(
                0.05
            ),

        "p25":
            s.quantile(
                0.25
            ),

        "median":
            s.median(),

        "mean":
            s.mean(),

        "p75":
            s.quantile(
                0.75
            ),

        "p95":
            s.quantile(
                0.95
            ),

        "p99":
            s.quantile(
                0.99
            ),

        "max":
            s.max(),

        "std":
            s.std()
    }

    return [
        {
            "variable":
                name,

            "metric":
                metric,

            "value":
                float(
                    value
                )
        }
        for metric, value
        in quantiles.items()
    ]


margin_rows = []


margin_rows.extend(
    profile_series(
        "lateness_hours",
        task[
            "lateness_hours"
        ]
    )
)


margin_rows.extend(
    profile_series(
        "lateness_calendar_days",
        task[
            "lateness_calendar_days"
        ]
    )
)


margin_profile = pd.DataFrame(
    margin_rows
)


margin_profile.to_csv(
    OUT
    /
    "05_lateness_margin_profile.csv",
    index=False
)


# =============================================================================
# MARGIN BINS
# =============================================================================

def margin_bucket(
    d
):

    if d <= -8:
        return "<= -8 days"

    if -7 <= d <= -2:
        return "-7 to -2 days"

    if d == -1:
        return "-1 day"

    if d == 0:
        return "0 days / same promised day"

    if d == 1:
        return "+1 day"

    if 2 <= d <= 3:
        return "+2 to +3 days"

    if 4 <= d <= 7:
        return "+4 to +7 days"

    if 8 <= d <= 14:
        return "+8 to +14 days"

    return ">= +15 days"


task[
    "lateness_bucket"
] = task[
    "lateness_calendar_days"
].map(
    margin_bucket
)


bucket_order = [
    "<= -8 days",
    "-7 to -2 days",
    "-1 day",
    "0 days / same promised day",
    "+1 day",
    "+2 to +3 days",
    "+4 to +7 days",
    "+8 to +14 days",
    ">= +15 days",
]


margin_bins = (
    task[
        "lateness_bucket"
    ]
    .value_counts()
    .reindex(
        bucket_order,
        fill_value=0
    )
    .rename_axis(
        "lateness_bucket"
    )
    .reset_index(
        name="orders"
    )
)


margin_bins[
    "pct_task"
] = (
    100
    *
    margin_bins[
        "orders"
    ]
    /
    N_TASK
)


margin_bins.to_csv(
    OUT
    /
    "06_lateness_margin_bins.csv",
    index=False
)


# =============================================================================
# MONTHLY LABEL STABILITY
# =============================================================================

monthly = (
    task
    .groupby(
        "purchase_month"
    )
    .agg(
        task_orders=(
            "order_id",
            "count"
        ),

        timestamp_strict_late=(
            "label_timestamp_strict",
            "sum"
        ),

        calendar_day_late=(
            "label_calendar_day",
            "sum"
        ),

        grace1_late=(
            "label_calendar_day_plus_1_grace",
            "sum"
        ),

        disagreements=(
            "label_disagreement",
            "sum"
        ),

        median_lateness_days=(
            "lateness_calendar_days",
            "median"
        ),

        mean_lateness_days=(
            "lateness_calendar_days",
            "mean"
        )
    )
    .reset_index()
)


for count_col, rate_col in [
    (
        "timestamp_strict_late",
        "timestamp_strict_late_rate_pct"
    ),

    (
        "calendar_day_late",
        "calendar_day_late_rate_pct"
    ),

    (
        "grace1_late",
        "grace1_late_rate_pct"
    ),

    (
        "disagreements",
        "disagreement_rate_pct"
    ),
]:

    monthly[
        rate_col
    ] = (
        100
        *
        monthly[
            count_col
        ]
        /
        monthly[
            "task_orders"
        ]
    )


monthly.to_csv(
    OUT
    /
    "07_label_by_month.csv",
    index=False
)


# =============================================================================
# OBSERVATION WINDOW
# =============================================================================

observation_end = (
    actual
    .dropna()
    .max()
)


status = (
    orders[
        "order_status"
    ]
    .astype(str)
)


observability = pd.Series(
    "UNCLASSIFIED",
    index=orders.index,
    dtype="object"
)


observability.loc[
    task_mask
] = "OBSERVED_DELIVERED"


terminal_non_delivery = (
    status.isin(
        [
            "canceled",
            "unavailable"
        ]
    )
    &
    ~task_mask
)


observability.loc[
    terminal_non_delivery
] = "NOT_APPLICABLE_TERMINAL_NON_DELIVERY"


outcome_present_status_mismatch = (
    actual.notna()
    &
    ~status.eq(
        "delivered"
    )
)


observability.loc[
    outcome_present_status_mismatch
] = "OUTCOME_PRESENT_STATUS_MISMATCH"


right_censor_candidate = (
    actual.isna()
    &
    estimated.notna()
    &
    (
        estimated
        >
        observation_end
    )
    &
    ~terminal_non_delivery
)


observability.loc[
    right_censor_candidate
] = "RIGHT_CENSORING_CANDIDATE"


unobserved_before_end = (
    actual.isna()
    &
    estimated.notna()
    &
    (
        estimated
        <=
        observation_end
    )
    &
    ~terminal_non_delivery
    &
    ~right_censor_candidate
)


observability.loc[
    unobserved_before_end
] = "OUTCOME_UNOBSERVED_BY_DATASET_END"


still_unclassified = (
    observability
    ==
    "UNCLASSIFIED"
)


observability.loc[
    still_unclassified
] = "OTHER_REVIEW"


orders_observation = pd.DataFrame(
    {
        "order_id":
            orders[
                "order_id"
            ],

        "order_status":
            orders[
                "order_status"
            ],

        "purchase_dt":
            purchase,

        "estimated_dt":
            estimated,

        "actual_dt":
            actual,

        "observation_class":
            observability
    }
)


observation_summary = (
    orders_observation[
        "observation_class"
    ]
    .value_counts()
    .rename_axis(
        "observation_class"
    )
    .reset_index(
        name="orders"
    )
)


observation_summary[
    "pct_source"
] = (
    100
    *
    observation_summary[
        "orders"
    ]
    /
    N_SOURCE
)


observation_summary[
    "observation_end"
] = str(
    observation_end
)


observation_summary.to_csv(
    OUT
    /
    "08_observation_window_audit.csv",
    index=False
)


orders_observation[
    orders_observation[
        "observation_class"
    ]
    !=
    "OBSERVED_DELIVERED"
].to_csv(
    OUT
    /
    "09_non_observable_orders.csv",
    index=False
)


# =============================================================================
# SOURCE LABEL COVERAGE BY MONTH
# =============================================================================

source_month = pd.DataFrame(
    {
        "order_id":
            orders[
                "order_id"
            ],

        "purchase_month":
            purchase
            .dt
            .to_period(
                "M"
            )
            .astype(str),

        "task_observable":
            task_mask,

        "right_censor_candidate":
            right_censor_candidate,

        "terminal_non_delivery":
            terminal_non_delivery,

        "unobserved_before_end":
            unobserved_before_end
    }
)


coverage_month = (
    source_month[
        source_month[
            "purchase_month"
        ]
        !=
        "NaT"
    ]
    .groupby(
        "purchase_month"
    )
    .agg(
        source_orders=(
            "order_id",
            "count"
        ),

        task_observable=(
            "task_observable",
            "sum"
        ),

        right_censor_candidates=(
            "right_censor_candidate",
            "sum"
        ),

        terminal_non_delivery=(
            "terminal_non_delivery",
            "sum"
        ),

        unobserved_before_end=(
            "unobserved_before_end",
            "sum"
        )
    )
    .reset_index()
)


coverage_month[
    "task_coverage_pct"
] = (
    100
    *
    coverage_month[
        "task_observable"
    ]
    /
    coverage_month[
        "source_orders"
    ]
)


coverage_month.to_csv(
    OUT
    /
    "10_label_observability_by_month.csv",
    index=False
)


# =============================================================================
# CONSTRUCT RECOMMENDATION
# =============================================================================

rule = CONTRACT[
    "recommendation_rule"
]


if (
    estimated_midnight_pct
    >=
    float(
        rule[
            "estimated_midnight_pct_min"
        ]
    )
    and
    N_DISAGREE
    >
    0
    and
    same_day_disagreement_pct
    >=
    float(
        rule[
            "disagreement_same_calendar_day_pct_min"
        ]
    )
):

    recommendation = (
        "CALENDAR_DAY_RECOMMENDED_PENDING_BUSINESS_CONFIRMATION"
    )

    rationale = (
        "The estimated-delivery field behaves as a calendar-date promise "
        "and the strict timestamp definition primarily changes orders "
        "delivered on the promised calendar day."
    )

else:

    recommendation = (
        "BUSINESS_RULE_REVIEW_REQUIRED"
    )

    rationale = (
        "Observed timestamp structure is insufficient for automatically "
        "preferring calendar-day semantics."
    )


candidate_contract = pd.DataFrame(
    [
        {
            "candidate":
                "TIMESTAMP_STRICT",

            "formula":
                "1[actual_timestamp > estimated_timestamp]",

            "recommended_role":
                "sensitivity / comparison",

            "construct_status":
                (
                    "NOT_PRIMARY_RECOMMENDATION"
                    if recommendation.startswith(
                        "CALENDAR_DAY"
                    )
                    else
                    "REVIEW"
                )
        },

        {
            "candidate":
                "CALENDAR_DAY",

            "formula":
                "1[date(actual) > date(estimated)]",

            "recommended_role":
                (
                    "primary candidate"
                    if recommendation.startswith(
                        "CALENDAR_DAY"
                    )
                    else
                    "candidate"
                ),

            "construct_status":
                recommendation
        },

        {
            "candidate":
                "CALENDAR_DAY_PLUS_1_GRACE",

            "formula":
                "1[date(actual) > date(estimated) + 1 day]",

            "recommended_role":
                "sensitivity only",

            "construct_status":
                "NO BUSINESS JUSTIFICATION FOR PRIMARY USE"
        }
    ]
)


candidate_contract.to_csv(
    OUT
    /
    "11_label_contract_candidates.csv",
    index=False
)


# =============================================================================
# SCORECARD
# =============================================================================

add_result(
    "LBL-001",
    "task_observability",
    "CRITICAL",
    (
        "PASS"
        if N_TASK > 0
        else "FAIL"
    ),
    (
        0
        if N_TASK > 0
        else 1
    ),
    max(
        N_SOURCE,
        1
    ),
    f"{N_TASK} observable delivered orders",
    "> 0",
    "Candidate supervised cohort exists."
)


for idx, col in enumerate(
    [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ],
    start=2
):

    failures = parse_failures[
        col
    ]

    add_result(
        f"LBL-{idx:03d}",
        "timestamp_parseability",
        "CRITICAL",
        (
            "PASS"
            if failures == 0
            else "FAIL"
        ),
        failures,
        int(
            orders[
                col
            ]
            .notna()
            .sum()
        ),
        f"{failures} parse failures",
        "0",
        col
    )


add_result(
    "LBL-005",
    "temporal_validity",
    "CRITICAL",
    (
        "PASS"
        if estimated_before_purchase == 0
        else "FAIL"
    ),
    estimated_before_purchase,
    N_TASK,
    f"{estimated_before_purchase} estimated dates before purchase",
    "0",
    ""
)


add_result(
    "LBL-006",
    "temporal_validity",
    "CRITICAL",
    (
        "PASS"
        if actual_before_purchase == 0
        else "FAIL"
    ),
    actual_before_purchase,
    N_TASK,
    f"{actual_before_purchase} actual deliveries before purchase",
    "0",
    ""
)


add_result(
    "LBL-007",
    "status_outcome_consistency",
    "WARNING",
    (
        "PASS"
        if delivered_status_missing_actual == 0
        else "WARN"
    ),
    delivered_status_missing_actual,
    int(
        orders[
            "order_status"
        ]
        .eq(
            "delivered"
        )
        .sum()
    ),
    f"{delivered_status_missing_actual} delivered statuses without actual delivery timestamp",
    "0 preferred",
    ""
)


add_result(
    "LBL-008",
    "status_outcome_consistency",
    "WARNING",
    (
        "PASS"
        if non_delivered_with_actual == 0
        else "WARN"
    ),
    non_delivered_with_actual,
    N_SOURCE,
    f"{non_delivered_with_actual} non-delivered statuses with actual timestamp",
    "0 preferred",
    ""
)


add_result(
    "LBL-009",
    "promise_time_semantics",
    "INFO",
    "INFO",
    estimated_midnight_count,
    N_TASK,
    f"{estimated_midnight_pct:.6f}% estimated timestamps at midnight",
    "diagnostic",
    (
        f"unique_time_of_day_values={estimated_unique_times}"
    )
)


add_result(
    "LBL-010",
    "construct_sensitivity",
    "INFO",
    "INFO",
    N_DISAGREE,
    N_TASK,
    f"{N_DISAGREE} timestamp/calendar disagreements",
    "diagnostic",
    ""
)


add_result(
    "LBL-011",
    "construct_boundary",
    "INFO",
    "INFO",
    same_day_disagreements,
    max(
        N_DISAGREE,
        1
    ),
    (
        f"{same_day_disagreement_pct:.6f}% of disagreements "
        "are same-calendar-day cases"
    ),
    "diagnostic",
    ""
)


add_result(
    "LBL-012",
    "observation_window",
    "INFO",
    "INFO",
    int(
        right_censor_candidate.sum()
    ),
    N_SOURCE,
    (
        f"{int(right_censor_candidate.sum())} "
        "right-censoring candidates"
    ),
    "diagnostic",
    (
        f"observation_end={observation_end}"
    )
)


scorecard = pd.DataFrame(
    results
)


scorecard.to_csv(
    OUT
    /
    "dq_gate_04_scorecard.csv",
    index=False
)


exceptions = scorecard[
    scorecard[
        "status"
    ]
    .isin(
        [
            "WARN",
            "FAIL"
        ]
    )
].copy()


exceptions.to_csv(
    OUT
    /
    "dq_gate_04_exceptions.csv",
    index=False
)


critical_failures = int(
    (
        (
            scorecard[
                "severity"
            ]
            ==
            "CRITICAL"
        )
        &
        (
            scorecard[
                "status"
            ]
            ==
            "FAIL"
        )
    )
    .sum()
)


warnings = int(
    (
        scorecard[
            "status"
        ]
        ==
        "WARN"
    )
    .sum()
)


gate_status = (
    "PASS"
    if critical_failures == 0
    else
    "FAIL"
)


# =============================================================================
# ISSUE EVIDENCE
# =============================================================================

issue_evidence = pd.DataFrame(
    [
        {
            "canonical_issue_id":
                "ISSUE-TARGET-001",

            "gate":
                "DQ_GATE_04",

            "evidence_status":
                "ASSESSED",

            "recommendation":
                recommendation,

            "evidence":
                (
                    f"estimated_midnight_pct={estimated_midnight_pct:.6f}; "
                    f"label_disagreements={N_DISAGREE}; "
                    f"same_day_disagreement_pct={same_day_disagreement_pct:.6f}"
                ),

            "final_business_confirmation_required":
                True
        },

        {
            "canonical_issue_id":
                "ISSUE-OBSERVATION-WINDOW-001",

            "gate":
                "DQ_GATE_04",

            "evidence_status":
                "ASSESSED",

            "recommendation":
                "DO_NOT_LABEL_UNOBSERVED_ORDERS_AS_ON_TIME",

            "evidence":
                (
                    f"observation_end={observation_end}; "
                    f"right_censor_candidates={int(right_censor_candidate.sum())}"
                ),

            "final_business_confirmation_required":
                False
        }
    ]
)


issue_evidence.to_csv(
    OUT
    /
    "12_issue_evidence_gate_04.csv",
    index=False
)


# =============================================================================
# SUMMARY JSON
# =============================================================================

summary = {
    "gate":
        "DQ_GATE_04_LABEL_CONSTRUCT_VALIDITY",

    "status":
        gate_status,

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_orders":
        N_SOURCE,

    "task_candidate_orders":
        N_TASK,

    "task_candidate_pct":
        pct(
            N_TASK,
            N_SOURCE
        ),

    "observation_end":
        str(
            observation_end
        ),

    "estimated_midnight_pct":
        estimated_midnight_pct,

    "estimated_unique_time_values":
        estimated_unique_times,

    "timestamp_strict_late_orders":
        int(
            task[
                "label_timestamp_strict"
            ]
            .sum()
        ),

    "timestamp_strict_late_rate_pct":
        pct(
            int(
                task[
                    "label_timestamp_strict"
                ]
                .sum()
            ),
            N_TASK
        ),

    "calendar_day_late_orders":
        int(
            task[
                "label_calendar_day"
            ]
            .sum()
        ),

    "calendar_day_late_rate_pct":
        pct(
            int(
                task[
                    "label_calendar_day"
                ]
                .sum()
            ),
            N_TASK
        ),

    "calendar_day_plus_1_grace_late_orders":
        int(
            task[
                "label_calendar_day_plus_1_grace"
            ]
            .sum()
        ),

    "label_disagreements":
        N_DISAGREE,

    "label_disagreement_pct":
        pct(
            N_DISAGREE,
            N_TASK
        ),

    "same_day_disagreements":
        same_day_disagreements,

    "same_day_disagreement_pct":
        same_day_disagreement_pct,

    "strict_late_calendar_ontime":
        strict_late_calendar_ontime,

    "calendar_late_strict_ontime":
        calendar_late_strict_ontime,

    "right_censoring_candidates":
        int(
            right_censor_candidate.sum()
        ),

    "delivered_status_missing_actual":
        delivered_status_missing_actual,

    "non_delivered_with_actual":
        non_delivered_with_actual,

    "construct_recommendation":
        recommendation,

    "construct_rationale":
        rationale,

    "label_finalized":
        False,

    "business_confirmation_required":
        True,

    "critical_failures":
        critical_failures,

    "warnings":
        warnings,

    "raw_modified":
        False,

    "silver_created":
        False
}


with (
    OUT
    /
    "dq_gate_04_summary.json"
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

calendar_row = label_comparison[
    label_comparison[
        "label_definition"
    ]
    ==
    "CALENDAR_DAY"
].iloc[
    0
]


strict_row = label_comparison[
    label_comparison[
        "label_definition"
    ]
    ==
    "TIMESTAMP_STRICT"
].iloc[
    0
]


report = []


report.append(
    "=" * 108
)

report.append(
    "DQ GATE 04 — LABEL / CONSTRUCT VALIDITY"
)

report.append(
    "=" * 108
)

report.append("")

report.append(
    f"STATUS                           : {gate_status}"
)

report.append(
    f"SOURCE ORDERS                    : {N_SOURCE:,}"
)

report.append(
    f"TASK CANDIDATE                   : {N_TASK:,}"
)

report.append(
    f"TASK / SOURCE                    : {pct(N_TASK, N_SOURCE):.4f}%"
)

report.append("")

report.append(
    "LABEL — TIMESTAMP STRICT"
)

report.append(
    f"  late_orders                    : {int(strict_row['late_orders']):,}"
)

report.append(
    f"  late_rate                      : {strict_row['late_rate_pct']:.6f}%"
)

report.append("")

report.append(
    "LABEL — CALENDAR DAY"
)

report.append(
    f"  late_orders                    : {int(calendar_row['late_orders']):,}"
)

report.append(
    f"  late_rate                      : {calendar_row['late_rate_pct']:.6f}%"
)

report.append("")

report.append(
    "CONSTRUCT COMPARISON"
)

report.append(
    f"  disagreements                  : {N_DISAGREE:,}"
)

report.append(
    f"  disagreement_rate              : {pct(N_DISAGREE, N_TASK):.6f}%"
)

report.append(
    f"  same-day disagreements         : {same_day_disagreements:,}"
)

report.append(
    f"  same-day share                 : {same_day_disagreement_pct:.6f}%"
)

report.append(
    f"  estimated timestamps midnight  : {estimated_midnight_pct:.6f}%"
)

report.append(
    f"  unique estimate times          : {estimated_unique_times}"
)

report.append("")

report.append(
    "OBSERVATION WINDOW"
)

report.append(
    f"  observation_end                : {observation_end}"
)

report.append(
    f"  right-censoring candidates     : {int(right_censor_candidate.sum()):,}"
)

report.append("")

report.append(
    "RECOMMENDATION"
)

report.append(
    f"  {recommendation}"
)

report.append("")

report.append(
    rationale
)

report.append("")

report.append(
    "IMPORTANT:"
)

report.append(
    "- This is a construct-validity recommendation."
)

report.append(
    "- The final business label is NOT automatically finalized."
)

report.append(
    "- Unobserved outcomes are NOT converted to on-time."
)

report.append(
    "- RAW was not modified."
)

report.append(
    "- Silver was not created."
)

report.append("")

report.append(
    f"CRITICAL FAILURES                : {critical_failures}"
)

report.append(
    f"WARNINGS                         : {warnings}"
)

report.append(
    "=" * 108
)


REPORT_PATH = (
    OUT
    /
    "DQ_GATE_04_LABEL_CONSTRUCT_REPORT.txt"
)


REPORT_PATH.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()
print("=" * 108)
print("DQ GATE 04 — LABEL / CONSTRUCT VALIDITY")
print("=" * 108)

print()

print(
    f"STATUS                           : {gate_status}"
)

print(
    f"SOURCE ORDERS                    : {N_SOURCE:,}"
)

print(
    f"TASK CANDIDATE                   : {N_TASK:,}"
)

print(
    f"TASK / SOURCE                    : {pct(N_TASK, N_SOURCE):.4f}%"
)


print()
print("=" * 108)
print("1. SEMÂNTICA TEMPORAL DA PROMESSA")
print("=" * 108)

print(
    time_semantics.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("2. COMPARAÇÃO DAS DEFINIÇÕES DE LABEL")
print("=" * 108)

print(
    label_comparison.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("3. CONFUSION MATRIX — LABEL vs LABEL")
print("=" * 108)

print(
    confusion_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("4. DISAGREEMENT / BOUNDARY")
print("=" * 108)

print(
    f"Disagreements                  : {N_DISAGREE:,}"
)

print(
    f"Rate                           : {pct(N_DISAGREE, N_TASK):.6f}%"
)

print(
    f"Same-calendar-day disagreements: {same_day_disagreements:,}"
)

print(
    f"Same-day share                 : {same_day_disagreement_pct:.6f}%"
)

print(
    f"Strict late / calendar on-time : {strict_late_calendar_ontime:,}"
)

print(
    f"Calendar late / strict on-time : {calendar_late_strict_ontime:,}"
)


print()
print("=" * 108)
print("5. MARGEM DE ATRASO")
print("=" * 108)

print(
    margin_profile.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("6. DISTRIBUIÇÃO EM FAIXAS")
print("=" * 108)

print(
    margin_bins.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("7. OBSERVATION WINDOW")
print("=" * 108)

print(
    observation_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("8. LABEL CONTRACT CANDIDATES")
print("=" * 108)

print(
    candidate_contract.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("DQ GATE 04 — RESULTADO")
print("=" * 108)

print(
    f"STATUS                    : {gate_status}"
)

print(
    f"CRITICAL FAILURES         : {critical_failures}"
)

print(
    f"WARNINGS                  : {warnings}"
)

print(
    f"CONSTRUCT RECOMMENDATION  : {recommendation}"
)

print(
    "LABEL FINALIZED           : NÃO"
)

print(
    "RAW MODIFIED              : NÃO"
)

print(
    "SILVER CREATED            : NÃO"
)

print()

print(
    "Arquivos gerados:"
)

for path in sorted(
    OUT.iterdir()
):

    if path.is_file():

        print(
            f"  - {path}"
        )


if gate_status == "PASS":

    print()

    print(
        "[PASS] DQ GATE 04 EXECUTADO."
    )

    print(
        "Construct validity foi auditada."
    )

    print(
        "A recomendação de label permanece sujeita "
        "à confirmação explícita do contrato de negócio."
    )

    sys.exit(0)

else:

    print()

    print(
        "[FAIL] DQ GATE 04 REPROVADO."
    )

    sys.exit(2)

