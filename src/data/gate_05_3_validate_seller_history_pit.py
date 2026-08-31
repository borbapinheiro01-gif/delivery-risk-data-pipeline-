#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DQ GATE 05.3
Seller History — Point-in-Time Material Validation

Não cria Silver.
Não treina modelo.

Este script prova materialmente que históricos de seller podem ser
reconstruídos sem usar eventos posteriores ao instante da previsão.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

RAW = (
    ROOT
    / "data"
    / "raw"
    / "olist"
)

OUT = (
    ROOT
    / "reports"
    / "data_quality"
    / "gate_05_point_in_time"
)

LABEL_PATH = (
    ROOT
    / "contracts"
    / "DELIVERY_RISK_LABEL_CONTRACT_V1.json"
)

PIT_CONTRACT_PATH = (
    ROOT
    / "configs"
    / "seller_history_pit_contract_v1.json"
)

SUMMARY_051 = (
    OUT
    / "dq_gate_05_1_summary.json"
)

SUMMARY_052A = (
    OUT
    / "dq_gate_05_2a_summary.json"
)

SUMMARY_052B = (
    OUT
    / "dq_gate_05_2b_summary.json"
)

ORDERS_FILE = (
    RAW
    / "olist_orders_dataset.csv"
)

ITEMS_FILE = (
    RAW
    / "olist_order_items_dataset.csv"
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


def pct(n, d):

    return (
        100.0 * n / d
        if d
        else np.nan
    )


# =============================================================================
# PREREQUISITES
# =============================================================================

s051 = load_json(
    SUMMARY_051
)

s052a = load_json(
    SUMMARY_052A
)

s052b = load_json(
    SUMMARY_052B
)

label_contract = load_json(
    LABEL_PATH
)

pit_contract = load_json(
    PIT_CONTRACT_PATH
)


for name, obj in [
    ("05.1", s051),
    ("05.2-A", s052a),
    ("05.2-B", s052b),
]:

    if obj.get(
        "step_status"
    ) != "PASS":

        raise SystemExit(
            f"[BLOQUEADO] {name} não está PASS."
        )


if label_contract.get(
    "status"
) != "FROZEN_FOR_MODELING":

    raise SystemExit(
        "[BLOQUEADO] Label Contract não está congelado."
    )


print("=" * 108)
print("DQ GATE 05.3 — SELLER HISTORY POINT-IN-TIME")
print("=" * 108)
print()

print("[PASS] Gate 05.1")
print("[PASS] Gate 05.2-A")
print("[PASS] Gate 05.2-B")
print("[PASS] Label Contract V1")
print()


# =============================================================================
# LOAD
# =============================================================================

orders = pd.read_csv(
    ORDERS_FILE,
    usecols=[
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ],
    low_memory=False
)


items = pd.read_csv(
    ITEMS_FILE,
    usecols=[
        "order_id",
        "seller_id",
    ],
    low_memory=False
)


# One association per order/seller.
seller_order = (
    items[
        [
            "order_id",
            "seller_id"
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


# =============================================================================
# TIMESTAMPS
# =============================================================================

orders[
    "purchase_dt"
] = pd.to_datetime(
    orders[
        "order_purchase_timestamp"
    ],
    errors="coerce"
)


orders[
    "estimated_dt"
] = pd.to_datetime(
    orders[
        "order_estimated_delivery_date"
    ],
    errors="coerce"
)


orders[
    "delivery_dt"
] = pd.to_datetime(
    orders[
        "order_delivered_customer_date"
    ],
    errors="coerce"
)


# =============================================================================
# SUPERVISED COHORT
# =============================================================================

task_mask = (
    orders[
        "order_status"
    ]
    .eq(
        "delivered"
    )
    &
    orders[
        "purchase_dt"
    ]
    .notna()
    &
    orders[
        "estimated_dt"
    ]
    .notna()
    &
    orders[
        "delivery_dt"
    ]
    .notna()
)


task_orders = orders.loc[
    task_mask,
    [
        "order_id",
        "purchase_dt",
        "estimated_dt",
        "delivery_dt",
    ]
].copy()


task_orders[
    "late_label"
] = (
    task_orders[
        "delivery_dt"
    ]
    .dt
    .normalize()
    >
    task_orders[
        "estimated_dt"
    ]
    .dt
    .normalize()
).astype(
    "int8"
)


expected_task_orders = int(
    label_contract[
        "eligible_supervised_population"
    ][
        "orders"
    ]
)


# =============================================================================
# CURRENT PREDICTION ENTITY TABLE
# =============================================================================

current = (
    seller_order
    .merge(
        task_orders[
            [
                "order_id",
                "purchase_dt"
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
    .rename(
        columns={
            "purchase_dt":
                "prediction_time"
        }
    )
    .drop_duplicates(
        [
            "order_id",
            "seller_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# PURCHASE EVENT HISTORY
#
# Known as soon as order purchase occurs.
# =============================================================================

purchase_events = (
    seller_order
    .merge(
        orders[
            [
                "order_id",
                "purchase_dt"
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


purchase_events = (
    purchase_events[
        purchase_events[
            "purchase_dt"
        ]
        .notna()
    ]
    .copy()
)


# =============================================================================
# OUTCOME EVENT HISTORY
#
# Outcome becomes usable only at delivery_dt.
# =============================================================================

outcome_events = (
    seller_order
    .merge(
        task_orders[
            [
                "order_id",
                "delivery_dt",
                "late_label"
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


# =============================================================================
# OWN-TIME VALIDITY OF HISTORICAL OUTCOMES
# =============================================================================

own_time = (
    seller_order
    .merge(
        task_orders[
            [
                "order_id",
                "purchase_dt",
                "delivery_dt"
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one"
    )
)


historical_outcome_before_own_purchase = int(
    (
        own_time[
            "delivery_dt"
        ]
        <
        own_time[
            "purchase_dt"
        ]
    )
    .sum()
)


# =============================================================================
# SORTED EVENT ARRAYS
# =============================================================================

purchase_arrays = {}


for seller, g in (
    purchase_events
    .sort_values(
        [
            "seller_id",
            "purchase_dt"
        ]
    )
    .groupby(
        "seller_id",
        sort=False
    )
):

    purchase_arrays[
        seller
    ] = (
        g[
            "purchase_dt"
        ]
        .astype(
            "int64"
        )
        .to_numpy()
    )


outcome_arrays = {}


for seller, g in (
    outcome_events
    .sort_values(
        [
            "seller_id",
            "delivery_dt"
        ]
    )
    .groupby(
        "seller_id",
        sort=False
    )
):

    times = (
        g[
            "delivery_dt"
        ]
        .astype(
            "int64"
        )
        .to_numpy()
    )

    labels = (
        g[
            "late_label"
        ]
        .astype(
            "int64"
        )
        .to_numpy()
    )

    prefix_late = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64
            ),
            np.cumsum(
                labels,
                dtype=np.int64
            )
        ]
    )

    outcome_arrays[
        seller
    ] = (
        times,
        labels,
        prefix_late
    )


# =============================================================================
# OUTPUT ARRAYS
# =============================================================================

N = len(
    current
)

NAT_INT = np.iinfo(
    np.int64
).min

NS_DAY = (
    24
    *
    60
    *
    60
    *
    1_000_000_000
)

NS_30D = (
    30
    *
    NS_DAY
)

NS_90D = (
    90
    *
    NS_DAY
)


prior_orders = np.zeros(
    N,
    dtype=np.int64
)

prior_orders_30d = np.zeros(
    N,
    dtype=np.int64
)

prior_outcomes = np.zeros(
    N,
    dtype=np.int64
)

prior_late = np.zeros(
    N,
    dtype=np.int64
)

outcomes_30d = np.zeros(
    N,
    dtype=np.int64
)

late_30d = np.zeros(
    N,
    dtype=np.int64
)

outcomes_90d = np.zeros(
    N,
    dtype=np.int64
)

late_90d = np.zeros(
    N,
    dtype=np.int64
)

latest_purchase = np.full(
    N,
    NAT_INT,
    dtype=np.int64
)

latest_outcome = np.full(
    N,
    NAT_INT,
    dtype=np.int64
)


# =============================================================================
# FAST POINT-IN-TIME BUILDER
# =============================================================================

for seller, idx in current.groupby(
    "seller_id",
    sort=False
).groups.items():

    idx = np.asarray(
        list(idx),
        dtype=np.int64
    )

    t = (
        current.loc[
            idx,
            "prediction_time"
        ]
        .astype(
            "int64"
        )
        .to_numpy()
    )


    # -------------------------------------------------------------------------
    # PURCHASE HISTORY
    # -------------------------------------------------------------------------

    parr = purchase_arrays.get(
        seller
    )


    if parr is not None:

        right = np.searchsorted(
            parr,
            t,
            side="left"
        )

        left30 = np.searchsorted(
            parr,
            t - NS_30D,
            side="left"
        )

        prior_orders[
            idx
        ] = right

        prior_orders_30d[
            idx
        ] = (
            right
            -
            left30
        )

        has_prev = (
            right
            >
            0
        )

        latest_purchase[
            idx[
                has_prev
            ]
        ] = (
            parr[
                right[
                    has_prev
                ]
                -
                1
            ]
        )


    # -------------------------------------------------------------------------
    # OUTCOME HISTORY
    # -------------------------------------------------------------------------

    outcome_pack = outcome_arrays.get(
        seller
    )


    if outcome_pack is not None:

        (
            otime,
            olabel,
            prefix
        ) = outcome_pack


        right = np.searchsorted(
            otime,
            t,
            side="left"
        )

        left30 = np.searchsorted(
            otime,
            t - NS_30D,
            side="left"
        )

        left90 = np.searchsorted(
            otime,
            t - NS_90D,
            side="left"
        )


        prior_outcomes[
            idx
        ] = right


        prior_late[
            idx
        ] = prefix[
            right
        ]


        outcomes_30d[
            idx
        ] = (
            right
            -
            left30
        )


        late_30d[
            idx
        ] = (
            prefix[
                right
            ]
            -
            prefix[
                left30
            ]
        )


        outcomes_90d[
            idx
        ] = (
            right
            -
            left90
        )


        late_90d[
            idx
        ] = (
            prefix[
                right
            ]
            -
            prefix[
                left90
            ]
        )


        has_prev = (
            right
            >
            0
        )


        latest_outcome[
            idx[
                has_prev
            ]
        ] = (
            otime[
                right[
                    has_prev
                ]
                -
                1
            ]
        )


# =============================================================================
# ATTACH FEATURES
# =============================================================================

current[
    "seller_prior_order_count"
] = prior_orders


current[
    "seller_order_volume_30d"
] = prior_orders_30d


current[
    "seller_prior_observed_delivery_count"
] = prior_outcomes


current[
    "seller_prior_late_count"
] = prior_late


current[
    "seller_observed_delivery_count_30d"
] = outcomes_30d


current[
    "seller_late_count_30d"
] = late_30d


current[
    "seller_observed_delivery_count_90d"
] = outcomes_90d


current[
    "seller_late_count_90d"
] = late_90d


current[
    "latest_prior_purchase_ts"
] = pd.Series(
    latest_purchase
).astype(
    "datetime64[ns]"
)


current[
    "latest_prior_outcome_ts"
] = pd.Series(
    latest_outcome
).astype(
    "datetime64[ns]"
)


# =============================================================================
# RAW RATES — AUDIT ONLY
#
# NO smoothing here.
# =============================================================================

current[
    "seller_late_rate_prior_raw"
] = np.divide(
    prior_late,
    prior_outcomes,
    out=np.full(
        N,
        np.nan,
        dtype=float
    ),
    where=(
        prior_outcomes
        >
        0
    )
)


current[
    "seller_late_rate_30d_raw"
] = np.divide(
    late_30d,
    outcomes_30d,
    out=np.full(
        N,
        np.nan,
        dtype=float
    ),
    where=(
        outcomes_30d
        >
        0
    )
)


current[
    "seller_late_rate_90d_raw"
] = np.divide(
    late_90d,
    outcomes_90d,
    out=np.full(
        N,
        np.nan,
        dtype=float
    ),
    where=(
        outcomes_90d
        >
        0
    )
)


# =============================================================================
# MATERIAL LEAKAGE CHECKS
# =============================================================================

purchase_time_leakage = int(
    (
        current[
            "latest_prior_purchase_ts"
        ]
        .notna()
        &
        (
            current[
                "latest_prior_purchase_ts"
            ]
            >=
            current[
                "prediction_time"
            ]
        )
    )
    .sum()
)


outcome_time_leakage = int(
    (
        current[
            "latest_prior_outcome_ts"
        ]
        .notna()
        &
        (
            current[
                "latest_prior_outcome_ts"
            ]
            >=
            current[
                "prediction_time"
            ]
        )
    )
    .sum()
)


negative_count_violations = int(
    (
        current[
            [
                "seller_prior_order_count",
                "seller_order_volume_30d",
                "seller_prior_observed_delivery_count",
                "seller_prior_late_count",
                "seller_observed_delivery_count_30d",
                "seller_late_count_30d",
                "seller_observed_delivery_count_90d",
                "seller_late_count_90d",
            ]
        ]
        <
        0
    )
    .any(
        axis=1
    )
    .sum()
)


late_gt_observed = int(
    (
        (
            current[
                "seller_prior_late_count"
            ]
            >
            current[
                "seller_prior_observed_delivery_count"
            ]
        )
        |
        (
            current[
                "seller_late_count_30d"
            ]
            >
            current[
                "seller_observed_delivery_count_30d"
            ]
        )
        |
        (
            current[
                "seller_late_count_90d"
            ]
            >
            current[
                "seller_observed_delivery_count_90d"
            ]
        )
    )
    .sum()
)


duplicate_current_entity_rows = int(
    current[
        [
            "order_id",
            "seller_id"
        ]
    ]
    .duplicated()
    .sum()
)


task_orders_without_seller = int(
    expected_task_orders
    -
    current[
        "order_id"
    ]
    .nunique()
)


# =============================================================================
# INDEPENDENT BRUTE-FORCE VALIDATION
# =============================================================================

purchase_groups = {
    seller:
        g.copy()
    for seller, g
    in purchase_events.groupby(
        "seller_id",
        sort=False
    )
}


outcome_groups = {
    seller:
        g.copy()
    for seller, g
    in outcome_events.groupby(
        "seller_id",
        sort=False
    )
}


sample_n = min(
    300,
    len(
        current
    )
)


sample = current.sample(
    n=sample_n,
    random_state=20260830
)


brute_rows = []


for idx, row in sample.iterrows():

    seller = row[
        "seller_id"
    ]

    t = row[
        "prediction_time"
    ]


    pg = purchase_groups.get(
        seller
    )


    if pg is None:

        exp_prior_orders = 0
        exp_volume30 = 0

    else:

        exp_prior_orders = int(
            (
                pg[
                    "purchase_dt"
                ]
                <
                t
            )
            .sum()
        )

        exp_volume30 = int(
            (
                (
                    pg[
                        "purchase_dt"
                    ]
                    >=
                    t
                    -
                    pd.Timedelta(
                        days=30
                    )
                )
                &
                (
                    pg[
                        "purchase_dt"
                    ]
                    <
                    t
                )
            )
            .sum()
        )


    og = outcome_groups.get(
        seller
    )


    if og is None:

        exp_outcomes = 0
        exp_late = 0
        exp_outcomes30 = 0
        exp_late30 = 0
        exp_outcomes90 = 0
        exp_late90 = 0

    else:

        prior = og[
            og[
                "delivery_dt"
            ]
            <
            t
        ]


        exp_outcomes = len(
            prior
        )


        exp_late = int(
            prior[
                "late_label"
            ]
            .sum()
        )


        w30 = prior[
            prior[
                "delivery_dt"
            ]
            >=
            t
            -
            pd.Timedelta(
                days=30
            )
        ]


        exp_outcomes30 = len(
            w30
        )


        exp_late30 = int(
            w30[
                "late_label"
            ]
            .sum()
        )


        w90 = prior[
            prior[
                "delivery_dt"
            ]
            >=
            t
            -
            pd.Timedelta(
                days=90
            )
        ]


        exp_outcomes90 = len(
            w90
        )


        exp_late90 = int(
            w90[
                "late_label"
            ]
            .sum()
        )


    actual = [
        int(
            row[
                "seller_prior_order_count"
            ]
        ),
        int(
            row[
                "seller_order_volume_30d"
            ]
        ),
        int(
            row[
                "seller_prior_observed_delivery_count"
            ]
        ),
        int(
            row[
                "seller_prior_late_count"
            ]
        ),
        int(
            row[
                "seller_observed_delivery_count_30d"
            ]
        ),
        int(
            row[
                "seller_late_count_30d"
            ]
        ),
        int(
            row[
                "seller_observed_delivery_count_90d"
            ]
        ),
        int(
            row[
                "seller_late_count_90d"
            ]
        ),
    ]


    expected = [
        exp_prior_orders,
        exp_volume30,
        exp_outcomes,
        exp_late,
        exp_outcomes30,
        exp_late30,
        exp_outcomes90,
        exp_late90,
    ]


    match = (
        actual
        ==
        expected
    )


    brute_rows.append(
        {
            "order_id":
                row[
                    "order_id"
                ],

            "seller_id":
                seller,

            "prediction_time":
                t,

            "match":
                match
        }
    )


brute = pd.DataFrame(
    brute_rows
)


bruteforce_mismatches = int(
    (
        ~brute[
            "match"
        ]
    )
    .sum()
)


# =============================================================================
# COVERAGE / COLD START
# =============================================================================

current[
    "has_prior_purchase_history"
] = (
    current[
        "seller_prior_order_count"
    ]
    >
    0
)


current[
    "has_prior_outcome_history"
] = (
    current[
        "seller_prior_observed_delivery_count"
    ]
    >
    0
)


current[
    "has_prior_outcome_30d"
] = (
    current[
        "seller_observed_delivery_count_30d"
    ]
    >
    0
)


current[
    "has_prior_outcome_90d"
] = (
    current[
        "seller_observed_delivery_count_90d"
    ]
    >
    0
)


order_coverage = (
    current
    .groupby(
        "order_id"
    )
    .agg(
        sellers=(
            "seller_id",
            "nunique"
        ),

        any_seller_purchase_history=(
            "has_prior_purchase_history",
            "any"
        ),

        all_sellers_purchase_history=(
            "has_prior_purchase_history",
            "all"
        ),

        any_seller_outcome_history=(
            "has_prior_outcome_history",
            "any"
        ),

        all_sellers_outcome_history=(
            "has_prior_outcome_history",
            "all"
        ),

        any_seller_outcome_30d=(
            "has_prior_outcome_30d",
            "any"
        ),

        any_seller_outcome_90d=(
            "has_prior_outcome_90d",
            "any"
        ),
    )
    .reset_index()
)


coverage_rows = []


for col in [
    "has_prior_purchase_history",
    "has_prior_outcome_history",
    "has_prior_outcome_30d",
    "has_prior_outcome_90d",
]:

    n = int(
        current[
            col
        ]
        .sum()
    )

    coverage_rows.append(
        {
            "grain":
                "ORDER_SELLER",

            "metric":
                col,

            "available":
                n,

            "denominator":
                len(
                    current
                ),

            "coverage_pct":
                pct(
                    n,
                    len(
                        current
                    )
                )
        }
    )


for col in [
    "any_seller_purchase_history",
    "all_sellers_purchase_history",
    "any_seller_outcome_history",
    "all_sellers_outcome_history",
    "any_seller_outcome_30d",
    "any_seller_outcome_90d",
]:

    n = int(
        order_coverage[
            col
        ]
        .sum()
    )

    coverage_rows.append(
        {
            "grain":
                "ORDER",

            "metric":
                col,

            "available":
                n,

            "denominator":
                len(
                    order_coverage
                ),

            "coverage_pct":
                pct(
                    n,
                    len(
                        order_coverage
                    )
                )
        }
    )


coverage = pd.DataFrame(
    coverage_rows
)


# =============================================================================
# FEATURE DISTRIBUTION
# =============================================================================

profile_rows = []


for col in [
    "seller_prior_order_count",
    "seller_order_volume_30d",
    "seller_prior_observed_delivery_count",
    "seller_prior_late_count",
    "seller_observed_delivery_count_30d",
    "seller_late_count_30d",
    "seller_observed_delivery_count_90d",
    "seller_late_count_90d",
]:

    s = current[
        col
    ]

    profile_rows.append(
        {
            "feature":
                col,

            "min":
                int(
                    s.min()
                ),

            "median":
                float(
                    s.median()
                ),

            "p95":
                float(
                    s.quantile(
                        0.95
                    )
                ),

            "p99":
                float(
                    s.quantile(
                        0.99
                    )
                ),

            "max":
                int(
                    s.max()
                ),

            "zero_pct":
                pct(
                    int(
                        (
                            s
                            ==
                            0
                        )
                        .sum()
                    ),
                    len(
                        s
                    )
                )
        }
    )


profile = pd.DataFrame(
    profile_rows
)


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


def add_check(
    check,
    condition,
    observed,
    expected
):

    checks.append(
        {
            "check":
                check,

            "status":
                (
                    "PASS"
                    if condition
                    else
                    "FAIL"
                ),

            "observed":
                observed,

            "expected":
                expected
        }
    )


add_check(
    "supervised_order_count_matches_label_contract",
    (
        len(
            task_orders
        )
        ==
        expected_task_orders
    ),
    len(
        task_orders
    ),
    expected_task_orders
)


add_check(
    "all_task_orders_have_seller_entity",
    (
        task_orders_without_seller
        ==
        0
    ),
    task_orders_without_seller,
    0
)


add_check(
    "order_seller_entity_unique",
    (
        duplicate_current_entity_rows
        ==
        0
    ),
    duplicate_current_entity_rows,
    0
)


add_check(
    "historical_outcomes_not_before_own_purchase",
    (
        historical_outcome_before_own_purchase
        ==
        0
    ),
    historical_outcome_before_own_purchase,
    0
)


add_check(
    "latest_purchase_event_strictly_before_prediction",
    (
        purchase_time_leakage
        ==
        0
    ),
    purchase_time_leakage,
    0
)


add_check(
    "latest_outcome_event_strictly_before_prediction",
    (
        outcome_time_leakage
        ==
        0
    ),
    outcome_time_leakage,
    0
)


add_check(
    "no_negative_history_counts",
    (
        negative_count_violations
        ==
        0
    ),
    negative_count_violations,
    0
)


add_check(
    "late_counts_never_exceed_observed_outcomes",
    (
        late_gt_observed
        ==
        0
    ),
    late_gt_observed,
    0
)


add_check(
    "independent_bruteforce_sample_matches",
    (
        bruteforce_mismatches
        ==
        0
    ),
    bruteforce_mismatches,
    0
)


validation = pd.DataFrame(
    checks
)


failures = int(
    validation[
        "status"
    ]
    .eq(
        "FAIL"
    )
    .sum()
)


step_status = (
    "PASS"
    if failures == 0
    else
    "FAIL"
)


# =============================================================================
# OUTPUTS
# =============================================================================

current.to_csv(
    OUT
    / "03a_seller_history_pit_audit_matrix.csv",
    index=False
)


validation.to_csv(
    OUT
    / "03b_seller_history_validation.csv",
    index=False
)


coverage.to_csv(
    OUT
    / "03c_seller_history_coverage.csv",
    index=False
)


profile.to_csv(
    OUT
    / "03d_seller_history_profile.csv",
    index=False
)


brute.to_csv(
    OUT
    / "03e_bruteforce_validation_sample.csv",
    index=False
)


order_coverage.to_csv(
    OUT
    / "03f_order_level_seller_history_coverage.csv",
    index=False
)


summary = {
    "step":
        "DQ_GATE_05_3_SELLER_HISTORY_PIT",

    "step_status":
        step_status,

    "gate_05_status":
        "IN_PROGRESS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "supervised_orders":
        len(
            task_orders
        ),

    "order_seller_rows":
        len(
            current
        ),

    "unique_current_sellers":
        int(
            current[
                "seller_id"
            ]
            .nunique()
        ),

    "purchase_event_rows":
        len(
            purchase_events
        ),

    "outcome_event_rows":
        len(
            outcome_events
        ),

    "purchase_time_leakage":
        purchase_time_leakage,

    "outcome_time_leakage":
        outcome_time_leakage,

    "bruteforce_sample_rows":
        sample_n,

    "bruteforce_mismatches":
        bruteforce_mismatches,

    "validation_failures":
        failures,

    "smoothing_applied":
        False,

    "model_01_unlocked":
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
    / "dq_gate_05_3_summary.json"
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


report = f"""
========================================================================================================
DQ GATE 05.3 — SELLER HISTORY POINT-IN-TIME
========================================================================================================

STEP STATUS                         : {step_status}
GATE 05 STATUS                      : IN_PROGRESS

SUPERVISED ORDERS                   : {len(task_orders):,}
ORDER-SELLER ROWS                   : {len(current):,}
UNIQUE CURRENT SELLERS              : {current['seller_id'].nunique():,}

PURCHASE EVENT ROWS                 : {len(purchase_events):,}
OUTCOME EVENT ROWS                  : {len(outcome_events):,}

PURCHASE-TIME LEAKAGE               : {purchase_time_leakage}
OUTCOME-TIME LEAKAGE                : {outcome_time_leakage}

BRUTE-FORCE SAMPLE                  : {sample_n}
BRUTE-FORCE MISMATCHES              : {bruteforce_mismatches}

VALIDATION FAILURES                 : {failures}

SMOOTHING APPLIED                   : NO

IMPORTANT
--------------------------------------------------------------------------------------------------------
Purchase-history rule:

    purchase_j < purchase_i

Outcome-history rule:

    delivery_j < purchase_i

The label/outcome of the current order is never used as a predictor.

No global full-dataset target prevalence was used to smooth historical rates.

This matrix is an AUDIT MATRIX only.
It is not Silver and it is not a training matrix.

RAW modified                        : NO
Silver created                      : NO
Model trained                       : NO

========================================================================================================
"""


(
    OUT
    / "DQ_GATE_05_3_SELLER_HISTORY_PIT_REPORT.txt"
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
print("VALIDATION")
print("=" * 108)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 108)
print("SELLER HISTORY COVERAGE")
print("=" * 108)

print(
    coverage.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("SELLER HISTORY PROFILE")
print("=" * 108)

print(
    profile.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print()
print("=" * 108)
print("RESULTADO 05.3")
print("=" * 108)

print(
    f"STEP STATUS             : {step_status}"
)

print(
    f"SUPERVISED ORDERS       : {len(task_orders):,}"
)

print(
    f"ORDER-SELLER ROWS       : {len(current):,}"
)

print(
    f"UNIQUE SELLERS          : {current['seller_id'].nunique():,}"
)

print(
    f"PURCHASE LEAKAGE        : {purchase_time_leakage}"
)

print(
    f"OUTCOME LEAKAGE         : {outcome_time_leakage}"
)

print(
    f"BRUTE MISMATCHES        : {bruteforce_mismatches}"
)

print(
    f"FAILURES                : {failures}"
)

print(
    "SMOOTHING APPLIED       : NÃO"
)

print(
    "MODEL 01 UNLOCKED       : NÃO"
)

print(
    "RAW MODIFIED            : NÃO"
)

print(
    "SILVER CREATED          : NÃO"
)

print(
    "MODEL TRAINED           : NÃO"
)


if failures:
    sys.exit(2)


print()
print(
    "[PASS] 05.3 VALIDADO."
)

print(
    "Gate 05 continua IN_PROGRESS."
)

print(
    "Não treinar MODEL_01 ainda."
)

