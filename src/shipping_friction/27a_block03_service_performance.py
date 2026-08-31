#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE 27A — BLOCK 03
SERVICE PERFORMANCE AUDIT

Cinco componentes congelados:

11. Actual delivery time
12. Promised delivery time
13. Delivery gap
14. Positive tardiness
15. Late-delivery reliability

Objetivo
--------
Separar:

A) velocidade física observada;
B) duração prometida;
C) desvio contínuo em relação à promessa;
D) severidade positiva do atraso;
E) falha oficial em dia-calendário.

Importante
----------
O target oficial do projeto é:

    LATE_DELIVERY_CALENDAR_DAY

Por isso NÃO assumimos que:

    continuous_delivery_gap > 0

é matematicamente idêntico ao target.

Auditamos explicitamente a definição calendário.

Não:
- retreina modelo preditivo;
- altera RAW;
- altera expected freight;
- estima causalidade;
- cria matriz N x N.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

MASTER = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

OUT_DIR = ROOT / "artifacts" / "shipping_friction"
REP_DIR = ROOT / "reports" / "shipping_friction"

OUT_CORE = (
    OUT_DIR
    / "27a_03_service_performance_core.csv"
)

OUT_COVERAGE = (
    REP_DIR
    / "27a_03_service_coverage.csv"
)

OUT_QUANTILES = (
    REP_DIR
    / "27a_03_service_quantiles.csv"
)

OUT_MONTHLY = (
    REP_DIR
    / "27a_03_service_monthly.csv"
)

OUT_CONSISTENCY = (
    REP_DIR
    / "27a_03_target_consistency.csv"
)

OUT_DECISION = (
    REP_DIR
    / "27a_block03_decision.json"
)

OUT_REPORT = (
    REP_DIR
    / "27a_block03_report.txt"
)


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

REQUIRED = [
    "order_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_delivered_customer_date",
    "actual_delivery_days",
    "promised_delivery_days",
    "late_delivery_calendar_day",
    "year_month",
]


# =============================================================================
# HELPERS
# =============================================================================

SECONDS_DAY = 86400.0


def safe_float(x):
    if x is None or pd.isna(x):
        return None
    return float(x)


def pct(n, d):
    if d == 0:
        return np.nan
    return 100.0 * n / d


def summarize_numeric(frame, field):

    s = frame[field].dropna()

    if len(s) == 0:
        return None

    return {
        "field": field,
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "q10": float(s.quantile(0.10)),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q90": float(s.quantile(0.90)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


# =============================================================================
# START
# =============================================================================

print("=" * 104)
print("27A — BLOCK 03 — SERVICE PERFORMANCE AUDIT")
print("=" * 104)

print("FINAL_PREDICTIVE_MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("CAUSAL_CLAIM = false")
print("N_X_N_MATRIX = false")
print()


# =============================================================================
# 1. INPUT VALIDATION
# =============================================================================

if not MASTER.exists():
    raise FileNotFoundError(
        f"Arquivo ausente: {MASTER}"
    )


header = pd.read_csv(
    MASTER,
    nrows=0,
)

missing = [
    c
    for c in REQUIRED
    if c not in header.columns
]

if missing:
    raise RuntimeError(
        f"Colunas obrigatórias ausentes: {missing}"
    )


print("[PASS] required service columns found")


# =============================================================================
# 2. LOAD ONLY REQUIRED COLUMNS
# =============================================================================

df = pd.read_csv(
    MASTER,
    usecols=REQUIRED,
    low_memory=False,
)


n_total = len(df)

print(f"[INFO] rows loaded = {n_total:,}")
print(f"[INFO] columns loaded = {len(df.columns)}")


if n_total == 0:
    raise RuntimeError(
        "Base vazia."
    )


duplicates = int(
    df["order_id"]
    .duplicated()
    .sum()
)

if duplicates != 0:
    raise RuntimeError(
        f"order_id duplicado: {duplicates}"
    )


# =============================================================================
# 3. PARSE TIMESTAMPS
# =============================================================================

timestamp_cols = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_delivered_customer_date",
]


for col in timestamp_cols:

    df[col] = pd.to_datetime(
        df[col],
        errors="coerce",
    )


df["year_month"] = (
    df["year_month"]
    .astype("string")
)


# =============================================================================
# 4. RECOMPUTE ACTUAL / PROMISED DURATIONS
# =============================================================================

df["actual_delivery_days_recomputed"] = (
    (
        df["order_delivered_customer_date"]
        -
        df["order_purchase_timestamp"]
    )
    .dt.total_seconds()
    /
    SECONDS_DAY
)


df["promised_delivery_days_recomputed"] = (
    (
        df["order_estimated_delivery_date"]
        -
        df["order_purchase_timestamp"]
    )
    .dt.total_seconds()
    /
    SECONDS_DAY
)


# Existing project fields.
df["actual_delivery_days"] = pd.to_numeric(
    df["actual_delivery_days"],
    errors="coerce",
)


df["promised_delivery_days"] = pd.to_numeric(
    df["promised_delivery_days"],
    errors="coerce",
)


df["late_delivery_calendar_day"] = pd.to_numeric(
    df["late_delivery_calendar_day"],
    errors="coerce",
)


# =============================================================================
# 5. AUDIT EXISTING DURATION FIELDS
# =============================================================================

actual_pair = df[
    [
        "actual_delivery_days",
        "actual_delivery_days_recomputed",
    ]
].dropna()


if len(actual_pair):

    actual_max_error = float(
        np.max(
            np.abs(
                actual_pair[
                    "actual_delivery_days"
                ]
                -
                actual_pair[
                    "actual_delivery_days_recomputed"
                ]
            )
        )
    )

else:

    actual_max_error = np.nan


promise_pair = df[
    [
        "promised_delivery_days",
        "promised_delivery_days_recomputed",
    ]
].dropna()


if len(promise_pair):

    promise_max_error = float(
        np.max(
            np.abs(
                promise_pair[
                    "promised_delivery_days"
                ]
                -
                promise_pair[
                    "promised_delivery_days_recomputed"
                ]
            )
        )
    )

else:

    promise_max_error = np.nan


# =============================================================================
# 6. CONTINUOUS DELIVERY GAP
#
# Positive:
#   delivered after promised timestamp.
#
# Negative:
#   delivered before promised timestamp.
# =============================================================================

df["delivery_gap_days_continuous"] = (
    (
        df["order_delivered_customer_date"]
        -
        df["order_estimated_delivery_date"]
    )
    .dt.total_seconds()
    /
    SECONDS_DAY
)


# Algebraic cross-check:
#
# actual - promised
#
df["delivery_gap_days_from_durations"] = (
    df["actual_delivery_days_recomputed"]
    -
    df["promised_delivery_days_recomputed"]
)


gap_pair = df[
    [
        "delivery_gap_days_continuous",
        "delivery_gap_days_from_durations",
    ]
].dropna()


if len(gap_pair):

    gap_identity_max_error = float(
        np.max(
            np.abs(
                gap_pair[
                    "delivery_gap_days_continuous"
                ]
                -
                gap_pair[
                    "delivery_gap_days_from_durations"
                ]
            )
        )
    )

else:

    gap_identity_max_error = np.nan


# =============================================================================
# 7. POSITIVE TARDINESS
#
# T_i = max(gap_i, 0)
# =============================================================================

df["positive_tardiness_days"] = np.where(
    df["delivery_gap_days_continuous"].notna(),
    np.maximum(
        df["delivery_gap_days_continuous"],
        0.0,
    ),
    np.nan,
)


# Early-delivery amount.
df["early_delivery_days"] = np.where(
    df["delivery_gap_days_continuous"].notna(),
    np.maximum(
        -df["delivery_gap_days_continuous"],
        0.0,
    ),
    np.nan,
)


# =============================================================================
# 8. CALENDAR-DAY DELIVERY GAP
#
# Official target is based on calendar-day lateness.
# =============================================================================

delivered_date = (
    df["order_delivered_customer_date"]
    .dt.normalize()
)

estimated_date = (
    df["order_estimated_delivery_date"]
    .dt.normalize()
)


df["delivery_gap_calendar_days"] = (
    delivered_date
    -
    estimated_date
).dt.days


df["late_from_calendar_gap"] = np.where(
    df["delivery_gap_calendar_days"].notna(),
    (
        df["delivery_gap_calendar_days"]
        > 0
    ).astype("int8"),
    np.nan,
)


# =============================================================================
# 9. CONTINUOUS-TIMESTAMP LATE FLAG
#
# Diagnostic only.
# It is intentionally NOT the official target.
# =============================================================================

df["late_from_continuous_gap"] = np.where(
    df["delivery_gap_days_continuous"].notna(),
    (
        df["delivery_gap_days_continuous"]
        > 0
    ).astype("int8"),
    np.nan,
)


# =============================================================================
# 10. OFFICIAL TARGET CONSISTENCY
# =============================================================================

target_pair = df[
    [
        "late_delivery_calendar_day",
        "late_from_calendar_gap",
    ]
].dropna()


if len(target_pair):

    official_mismatch = int(
        (
            target_pair[
                "late_delivery_calendar_day"
            ]
            !=
            target_pair[
                "late_from_calendar_gap"
            ]
        )
        .sum()
    )

else:

    official_mismatch = 0


continuous_pair = df[
    [
        "late_delivery_calendar_day",
        "late_from_continuous_gap",
    ]
].dropna()


if len(continuous_pair):

    continuous_vs_official_mismatch = int(
        (
            continuous_pair[
                "late_delivery_calendar_day"
            ]
            !=
            continuous_pair[
                "late_from_continuous_gap"
            ]
        )
        .sum()
    )

else:

    continuous_vs_official_mismatch = 0


# Cases where continuous timestamp says "after estimate"
# but calendar-day target says not late.
continuous_after_same_calendar = int(
    (
        (df["late_from_continuous_gap"] == 1)
        &
        (df["late_from_calendar_gap"] == 0)
    )
    .sum()
)


# =============================================================================
# 11. COVERAGE
# =============================================================================

coverage_fields = [
    "actual_delivery_days",
    "promised_delivery_days",
    "actual_delivery_days_recomputed",
    "promised_delivery_days_recomputed",
    "delivery_gap_days_continuous",
    "delivery_gap_calendar_days",
    "positive_tardiness_days",
    "late_delivery_calendar_day",
]


coverage_rows = []


for field in coverage_fields:

    n_non_null = int(
        df[field]
        .notna()
        .sum()
    )

    coverage_rows.append({
        "field":
            field,

        "n_non_null":
            n_non_null,

        "n_total":
            n_total,

        "coverage_pct":
            pct(
                n_non_null,
                n_total,
            ),
    })


coverage = pd.DataFrame(
    coverage_rows
)


coverage.to_csv(
    OUT_COVERAGE,
    index=False,
)


# =============================================================================
# 12. QUANTILES
# =============================================================================

summary_rows = []


for field in [
    "actual_delivery_days_recomputed",
    "promised_delivery_days_recomputed",
    "delivery_gap_days_continuous",
    "delivery_gap_calendar_days",
    "positive_tardiness_days",
    "early_delivery_days",
]:

    row = summarize_numeric(
        df,
        field,
    )

    if row is not None:
        summary_rows.append(row)


quantiles = pd.DataFrame(
    summary_rows
)


quantiles.to_csv(
    OUT_QUANTILES,
    index=False,
)


# =============================================================================
# 13. MONTHLY SERVICE AUDIT
# =============================================================================

monthly_rows = []


for month, grp in df.groupby(
    "year_month",
    dropna=True,
    observed=True,
):

    valid_service = grp[
        grp["delivery_gap_days_continuous"]
        .notna()
    ]


    n_month = len(grp)
    n_service = len(valid_service)


    official_target = grp[
        "late_delivery_calendar_day"
    ].dropna()


    monthly_rows.append({
        "year_month":
            str(month),

        "n_orders":
            int(n_month),

        "n_service_observed":
            int(n_service),

        "service_coverage_pct":
            pct(
                n_service,
                n_month,
            ),

        "actual_days_mean":
            safe_float(
                valid_service[
                    "actual_delivery_days_recomputed"
                ].mean()
            ),

        "actual_days_median":
            safe_float(
                valid_service[
                    "actual_delivery_days_recomputed"
                ].median()
            ),

        "promised_days_mean":
            safe_float(
                valid_service[
                    "promised_delivery_days_recomputed"
                ].mean()
            ),

        "promised_days_median":
            safe_float(
                valid_service[
                    "promised_delivery_days_recomputed"
                ].median()
            ),

        "gap_days_mean":
            safe_float(
                valid_service[
                    "delivery_gap_days_continuous"
                ].mean()
            ),

        "gap_days_median":
            safe_float(
                valid_service[
                    "delivery_gap_days_continuous"
                ].median()
            ),

        "positive_tardiness_mean":
            safe_float(
                valid_service[
                    "positive_tardiness_days"
                ].mean()
            ),

        "late_rate_official":
            (
                safe_float(
                    official_target.mean()
                )
                if len(official_target)
                else None
            ),
    })


monthly = pd.DataFrame(
    monthly_rows
)


monthly.to_csv(
    OUT_MONTHLY,
    index=False,
)


# =============================================================================
# 14. TARGET CONSISTENCY TABLE
# =============================================================================

consistency = pd.DataFrame(
    [
        {
            "comparison":
                "official_target_vs_calendar_gap",

            "n_compared":
                int(len(target_pair)),

            "n_mismatch":
                official_mismatch,

            "mismatch_pct":
                pct(
                    official_mismatch,
                    len(target_pair),
                ),

            "expected_identity":
                True,
        },

        {
            "comparison":
                "official_target_vs_continuous_timestamp_gap",

            "n_compared":
                int(len(continuous_pair)),

            "n_mismatch":
                continuous_vs_official_mismatch,

            "mismatch_pct":
                pct(
                    continuous_vs_official_mismatch,
                    len(continuous_pair),
                ),

            "expected_identity":
                False,
        },
    ]
)


consistency.to_csv(
    OUT_CONSISTENCY,
    index=False,
)


# =============================================================================
# 15. MATERIALIZE SERVICE CORE
# =============================================================================

out_cols = [
    "order_id",
    "year_month",

    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_delivered_customer_date",

    "actual_delivery_days",
    "actual_delivery_days_recomputed",

    "promised_delivery_days",
    "promised_delivery_days_recomputed",

    "delivery_gap_days_continuous",
    "delivery_gap_calendar_days",

    "positive_tardiness_days",
    "early_delivery_days",

    "late_delivery_calendar_day",
    "late_from_calendar_gap",
    "late_from_continuous_gap",
]


df[
    out_cols
].to_csv(
    OUT_CORE,
    index=False,
)


# =============================================================================
# 16. CORE RESULTS
# =============================================================================

service_observed = df[
    "delivery_gap_days_continuous"
].notna()


n_service = int(
    service_observed.sum()
)


official = df[
    "late_delivery_calendar_day"
].dropna()


late_rate = (
    float(
        official.mean()
    )
    if len(official)
    else np.nan
)


gap_valid = df.loc[
    service_observed,
    "delivery_gap_days_continuous",
]


actual_valid = df.loc[
    service_observed,
    "actual_delivery_days_recomputed",
]


promise_valid = df.loc[
    service_observed,
    "promised_delivery_days_recomputed",
]


tardiness_valid = df.loc[
    service_observed,
    "positive_tardiness_days",
]


n_continuous_late = int(
    (
        df["late_from_continuous_gap"]
        == 1
    ).sum()
)


n_calendar_late = int(
    (
        df["late_from_calendar_gap"]
        == 1
    ).sum()
)


# =============================================================================
# 17. GATES
# =============================================================================

gates = {
    "B03_G01_input_exists":
        MASTER.exists(),

    "B03_G02_order_id_unique":
        duplicates == 0,

    "B03_G03_population_preserved":
        len(df) == n_total,

    "B03_G04_service_population_nontrivial":
        n_service >= 1000,

    "B03_G05_actual_duration_recomputation_consistent":
        (
            np.isnan(actual_max_error)
            or actual_max_error <= 1e-8
        ),

    "B03_G06_promised_duration_recomputation_consistent":
        (
            np.isnan(promise_max_error)
            or promise_max_error <= 1e-8
        ),

    "B03_G07_gap_identity_valid":
        (
            np.isnan(gap_identity_max_error)
            or gap_identity_max_error <= 1e-10
        ),

    "B03_G08_official_target_matches_calendar_definition":
        official_mismatch == 0,

    "B03_G09_positive_tardiness_nonnegative":
        bool(
            df[
                "positive_tardiness_days"
            ]
            .dropna()
            .ge(0)
            .all()
        ),

    "B03_G10_late_target_binary":
        bool(
            set(
                df[
                    "late_delivery_calendar_day"
                ]
                .dropna()
                .unique()
            )
            .issubset(
                {0, 1, 0.0, 1.0}
            )
        ),

    "B03_G11_final_predictive_model_not_refit":
        True,

    "B03_G12_raw_untouched":
        True,
}


failed = [
    key
    for key, ok
    in gates.items()
    if not ok
]


status = (
    "PASS"
    if not failed
    else "FAIL"
)


# =============================================================================
# 18. DECISION JSON
# =============================================================================

decision = {
    "status":
        status,

    "generated_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "module":
        "27A_BLOCK03_SERVICE_PERFORMANCE",

    "population": {
        "n_orders":
            int(n_total),

        "n_service_observed":
            n_service,

        "service_coverage_pct":
            float(
                100.0
                *
                n_service
                /
                n_total
            ),

        "n_official_late":
            int(
                (
                    df[
                        "late_delivery_calendar_day"
                    ]
                    == 1
                )
                .sum()
            ),

        "official_late_rate":
            safe_float(
                late_rate
            ),

        "n_calendar_late_recomputed":
            n_calendar_late,

        "n_continuous_timestamp_late":
            n_continuous_late,

        "continuous_after_estimate_but_same_calendar_day":
            continuous_after_same_calendar,
    },

    "service_summary": {
        "actual_delivery_days_mean":
            safe_float(
                actual_valid.mean()
            ),

        "actual_delivery_days_median":
            safe_float(
                actual_valid.median()
            ),

        "promised_delivery_days_mean":
            safe_float(
                promise_valid.mean()
            ),

        "promised_delivery_days_median":
            safe_float(
                promise_valid.median()
            ),

        "delivery_gap_days_mean":
            safe_float(
                gap_valid.mean()
            ),

        "delivery_gap_days_median":
            safe_float(
                gap_valid.median()
            ),

        "positive_tardiness_days_mean":
            safe_float(
                tardiness_valid.mean()
            ),

        "positive_tardiness_days_median":
            safe_float(
                tardiness_valid.median()
            ),
    },

    "definition_audit": {
        "actual_duration_max_abs_error":
            safe_float(
                actual_max_error
            ),

        "promised_duration_max_abs_error":
            safe_float(
                promise_max_error
            ),

        "delivery_gap_identity_max_abs_error":
            safe_float(
                gap_identity_max_error
            ),

        "official_vs_calendar_mismatch":
            official_mismatch,

        "official_vs_continuous_mismatch":
            continuous_vs_official_mismatch,
    },

    "primary_service_definitions": {
        "speed":
            "actual delivery elapsed days",

        "promise":
            "estimated delivery timestamp minus purchase timestamp",

        "continuous_delivery_gap":
            "delivered timestamp minus estimated delivery timestamp",

        "positive_tardiness":
            "max(continuous delivery gap, 0)",

        "official_reliability_failure":
            "LATE_DELIVERY_CALENDAR_DAY",

        "calendar_gap":
            "delivered calendar date minus estimated calendar date",
    },

    "interpretation_guardrails": {
        "actual_speed_equals_reliability":
            False,

        "continuous_gap_positive_equals_official_late_by_definition":
            False,

        "official_target_is_calendar_day_definition":
            True,

        "post_outcome_service_fields_valid_as_purchase_time_predictors":
            False,

        "service_audit_is_causal":
            False,

        "final_predictive_model_refit":
            False,

        "raw_modified":
            False,
    },

    "gates":
        gates,

    "failed_gates":
        failed,

    "outputs": [
        str(
            OUT_CORE.relative_to(ROOT)
        ),
        str(
            OUT_COVERAGE.relative_to(ROOT)
        ),
        str(
            OUT_QUANTILES.relative_to(ROOT)
        ),
        str(
            OUT_MONTHLY.relative_to(ROOT)
        ),
        str(
            OUT_CONSISTENCY.relative_to(ROOT)
        ),
    ],
}


OUT_DECISION.write_text(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# 19. HUMAN-READABLE REPORT
# =============================================================================

report = f"""
====================================================================================================
27A — BLOCK 03 — SERVICE PERFORMANCE AUDIT
====================================================================================================

STATUS
------
{status}

POPULATION
----------
Total orders:
    {n_total:,}

Observed service outcomes:
    {n_service:,}

Coverage:
    {100*n_service/n_total:.2f}%

Official late-delivery rate:
    {late_rate:.6f}

OFFICIAL VS RECOMPUTED DEFINITIONS
----------------------------------
Actual-duration max absolute error:
    {actual_max_error}

Promised-duration max absolute error:
    {promise_max_error}

Delivery-gap algebraic identity max error:
    {gap_identity_max_error}

Official target vs calendar-gap mismatches:
    {official_mismatch}

Official target vs continuous-timestamp-gap mismatches:
    {continuous_vs_official_mismatch}

Continuous timestamp says after estimate,
but delivery remains on same calendar day:
    {continuous_after_same_calendar}

SERVICE SUMMARY
---------------
Actual delivery days:
    mean   = {actual_valid.mean():.6f}
    median = {actual_valid.median():.6f}

Promised delivery days:
    mean   = {promise_valid.mean():.6f}
    median = {promise_valid.median():.6f}

Delivery gap:
    mean   = {gap_valid.mean():.6f}
    median = {gap_valid.median():.6f}

Positive tardiness:
    mean   = {tardiness_valid.mean():.6f}
    median = {tardiness_valid.median():.6f}

INTERPRETATION
--------------
Actual delivery time measures physical elapsed service time.

Promised delivery time measures the service expectation encoded
in the estimated-delivery commitment.

Continuous delivery gap measures how early/late the delivery
occurred relative to the promised timestamp.

Positive tardiness measures only the positive side of that gap.

The official project reliability target remains:

    LATE_DELIVERY_CALENDAR_DAY

A positive continuous timestamp gap is NOT automatically treated
as identical to the official calendar-day target.

This is an EX POST service audit.
Outcome fields remain forbidden as purchase-time predictors.

No causal service effect is claimed.

FAILED GATES
------------
{failed if failed else "NONE"}

NEXT BLOCK
----------
27A BLOCK 04 — COST-TO-SERVICE

That block will combine:

    Freight Burden
    Freight Cost Anomaly
    Delivery Gap
    Official Late Delivery

to answer:

    Does paying more than expected buy better service?

and to build the first:

    Cost x Service quadrants
    Critical Shipping Friction definition

PARAR AQUI
====================================================================================================
""".strip()


OUT_REPORT.write_text(
    report + "\n",
    encoding="utf-8",
)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 104)
print("SERVICE COVERAGE")
print("=" * 104)

print(
    coverage.to_string(
        index=False
    )
)

print()
print("=" * 104)
print("TARGET CONSISTENCY")
print("=" * 104)

print(
    consistency.to_string(
        index=False
    )
)

print()
print("=" * 104)
print("SERVICE QUANTILES")
print("=" * 104)

print(
    quantiles.to_string(
        index=False
    )
)

print()
print("=" * 104)
print("BLOCK 03 DECISION")
print("=" * 104)

print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False,
    )
)

print()

if status != "PASS":
    raise RuntimeError(
        f"BLOCK 03 FAILED: {failed}"
    )


print("[PASS 27A-B03] SERVICE PERFORMANCE AUDIT COMPLETE")
print("FINAL_PREDICTIVE_MODEL_REFIT = false")
print("RAW_MODIFIED = false")
print("CAUSAL_CLAIM = false")
print("PARAR AQUI")
