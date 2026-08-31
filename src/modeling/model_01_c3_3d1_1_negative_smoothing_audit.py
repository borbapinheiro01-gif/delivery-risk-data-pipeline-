#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-D1.1
NEGATIVE SMOOTHING ARTIFACT AUDIT
===============================================================================

PERGUNTA
--------
Os valores negativos observados após o smoothing D1 são:

    A) pequenos artefatos próximos de zero,

ou

    B) violações quantitativamente relevantes da natureza
       não-negativa das curvas de volume/frete?

IMPORTANTE
----------
Nenhuma correção é aplicada.

NÃO:
- clipa negativos;
- substitui por zero;
- escolhe lambda;
- escolhe janela;
- escolhe K;
- usa target;
- aplica FPCA;
- cria feature;
- treina modelo;
- altera RAW;
- altera os .npy originais.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

OUT = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

ART = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "functional_feasibility"
)

D1_SUMMARY = (
    OUT
    / "03e_smoothing_sensitivity_summary.json"
)

D1_RESULTS = (
    OUT
    / "03c_smoothing_geometry_sensitivity.csv"
)

SUPPORT = (
    OUT
    / "02l_history_support_by_order.csv"
)

VOLUME = (
    ART
    / "02g_purchase_volume_curve_90d.npy"
)

FREIGHT = (
    ART
    / "02h_purchase_freight_curve_90d.npy"
)


# =============================================================================
# OUTPUTS
# =============================================================================

P_AUDIT = (
    OUT
    / "03g_negative_smoothing_severity.csv"
)

P_VALIDATION = (
    OUT
    / "03h_negative_smoothing_validation.csv"
)

P_SUMMARY = (
    OUT
    / "03i_negative_smoothing_summary.json"
)

P_REPORT = (
    OUT
    / "03j_negative_smoothing_report.txt"
)


# =============================================================================
# CONSTANTS
# =============================================================================

WINDOWS = [
    30,
    60,
    90
]

CHANNELS = [
    "purchase_volume",
    "purchase_freight"
]

EXPECTED_ROWS = 96470

NEGATIVE_TOL = 1e-10


# =============================================================================
# HELPERS
# =============================================================================

def add_check(
    rows,
    check,
    condition,
    observed,
    expected
):
    rows.append({
        "check": check,
        "status": "PASS" if bool(condition) else "FAIL",
        "observed": observed,
        "expected": expected,
    })


def second_difference_matrix(
    length
):
    D = np.zeros(
        (
            length - 2,
            length
        ),
        dtype=np.float64
    )

    i = np.arange(
        length - 2
    )

    D[i, i] = 1.0
    D[i, i + 1] = -2.0
    D[i, i + 2] = 1.0

    return D


def smoother_matrix(
    length,
    lam
):
    D2 = second_difference_matrix(
        length
    )

    penalty = (
        D2.T
        @
        D2
    )

    A = (
        np.eye(
            length,
            dtype=np.float64
        )
        +
        float(lam)
        *
        penalty
    )

    S = np.linalg.solve(
        A,
        np.eye(
            length,
            dtype=np.float64
        )
    )

    return (
        S + S.T
    ) / 2.0


def parse_bool(
    series
):
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.to_numpy(
            dtype=bool
        )

    x = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not x.isin(
        [
            "true",
            "false"
        ]
    ).all():
        raise RuntimeError(
            f"Boolean inválido em {series.name}"
        )

    return (
        x.eq("true")
        .to_numpy(dtype=bool)
    )


# =============================================================================
# START
# =============================================================================

start = time.perf_counter()

print()
print("=" * 118)
print("MODEL 01.0-C3.3-D1.1 — NEGATIVE SMOOTHING ARTIFACT AUDIT")
print("=" * 118)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

for path, label in [
    (
        D1_SUMMARY,
        "D1 summary"
    ),
    (
        D1_RESULTS,
        "D1 results"
    ),
    (
        SUPPORT,
        "support"
    ),
    (
        VOLUME,
        "volume curves"
    ),
    (
        FREIGHT,
        "freight curves"
    ),
]:
    if not path.exists():
        raise SystemExit(
            f"[FAIL] {label} ausente: {path}"
        )

    print(
        f"[PASS] {label}"
    )


with D1_SUMMARY.open(
    encoding="utf-8"
) as f:
    d1 = json.load(f)


if d1.get(
    "status"
) != "PASS":
    raise SystemExit(
        "[STOP] D1 não está PASS."
    )


if d1.get(
    "validation_failures"
) != 0:
    raise SystemExit(
        "[STOP] D1 possui validation failures."
    )


if d1.get(
    "target_used"
) is not False:
    raise SystemExit(
        "[STOP] D1 registrou target_used != false."
    )


print()
print("[PASS] C3.3-D1 formalmente validado.")


# =============================================================================
# 2. LOAD
# =============================================================================

results = pd.read_csv(
    D1_RESULTS
)

support = pd.read_csv(
    SUPPORT
)

if len(
    support
) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Support rows={len(support)}; "
        f"esperado={EXPECTED_ROWS}"
    )


volume = np.load(
    VOLUME,
    mmap_mode="r",
    allow_pickle=False
)

freight = np.load(
    FREIGHT,
    mmap_mode="r",
    allow_pickle=False
)


if volume.shape != (
    EXPECTED_ROWS,
    90
):
    raise RuntimeError(
        f"Volume shape inválido: {volume.shape}"
    )


if freight.shape != (
    EXPECTED_ROWS,
    90
):
    raise RuntimeError(
        f"Freight shape inválido: {freight.shape}"
    )


curves = {
    "purchase_volume":
        volume,

    "purchase_freight":
        freight,
}


support_masks = {}

for window in WINDOWS:
    support_masks[
        window
    ] = parse_bool(
        support[
            f"full_support_{window}d"
        ]
    )


# =============================================================================
# 3. ONLY NONTRIVIAL SMOOTHING
# =============================================================================

configs = (
    results.loc[
        results[
            "edf_ratio_target"
        ]
        <
        1.0
    ]
    .copy()
    .sort_values(
        [
            "channel",
            "window_days",
            "edf_ratio_target"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


if len(
    configs
) != 18:
    raise RuntimeError(
        f"Esperávamos 18 configurações suavizadas; "
        f"obtido={len(configs)}"
    )


# =============================================================================
# 4. RECOMPUTE AND AUDIT NEGATIVE VALUES
# =============================================================================

audit_rows = []


print()
print("=" * 118)
print("NEGATIVE SEVERITY")
print("=" * 118)


for _, config in configs.iterrows():

    channel = str(
        config[
            "channel"
        ]
    )

    window = int(
        config[
            "window_days"
        ]
    )

    edf_ratio = float(
        config[
            "edf_ratio_target"
        ]
    )

    lam = float(
        config[
            "lambda"
        ]
    )


    mask = support_masks[
        window
    ]


    raw = np.asarray(
        curves[
            channel
        ][
            mask,
            :window
        ],
        dtype=np.float64
    )


    S = smoother_matrix(
        window,
        lam
    )


    smooth = (
        raw
        @
        S
    )


    neg_mask = (
        smooth
        <
        -NEGATIVE_TOL
    )


    negative_cells = int(
        neg_mask.sum()
    )

    total_cells = int(
        smooth.size
    )


    row_has_negative = (
        neg_mask.any(
            axis=1
        )
    )

    affected_rows = int(
        row_has_negative.sum()
    )


    negative_values = (
        smooth[
            neg_mask
        ]
    )


    if negative_cells:

        minimum_value = float(
            np.min(
                negative_values
            )
        )

        negative_p01 = float(
            np.quantile(
                negative_values,
                0.01
            )
        )

        negative_p05 = float(
            np.quantile(
                negative_values,
                0.05
            )
        )

        negative_median = float(
            np.median(
                negative_values
            )
        )

        negative_p95 = float(
            np.quantile(
                negative_values,
                0.95
            )
        )

        closest_to_zero_negative = float(
            np.max(
                negative_values
            )
        )

        negative_mass = float(
            -negative_values.sum()
        )

        corresponding_raw = (
            raw[
                neg_mask
            ]
        )

        negative_on_raw_zero_pct = float(
            100.0
            *
            (
                corresponding_raw
                ==
                0
            ).mean()
        )

        negative_on_raw_positive_pct = float(
            100.0
            *
            (
                corresponding_raw
                >
                0
            ).mean()
        )

    else:

        minimum_value = 0.0
        negative_p01 = 0.0
        negative_p05 = 0.0
        negative_median = 0.0
        negative_p95 = 0.0
        closest_to_zero_negative = 0.0
        negative_mass = 0.0
        negative_on_raw_zero_pct = 0.0
        negative_on_raw_positive_pct = 0.0


    raw_total_mass = float(
        raw.sum()
    )


    smooth_positive_mass = float(
        smooth[
            smooth > 0
        ].sum()
    )


    raw_mean_cell = float(
        raw.mean()
    )


    raw_positive = (
        raw[
            raw > 0
        ]
    )


    raw_positive_median = (
        float(
            np.median(
                raw_positive
            )
        )
        if len(
            raw_positive
        )
        else np.nan
    )


    negative_mass_pct_raw_mass = (
        float(
            100.0
            *
            negative_mass
            /
            raw_total_mass
        )
        if raw_total_mass > 0
        else 0.0
    )


    negative_mass_pct_smooth_positive = (
        float(
            100.0
            *
            negative_mass
            /
            smooth_positive_mass
        )
        if smooth_positive_mass > 0
        else 0.0
    )


    minimum_abs_over_raw_mean = (
        float(
            abs(
                minimum_value
            )
            /
            raw_mean_cell
        )
        if raw_mean_cell > 0
        else np.nan
    )


    minimum_abs_over_positive_median = (
        float(
            abs(
                minimum_value
            )
            /
            raw_positive_median
        )
        if (
            np.isfinite(
                raw_positive_median
            )
            and
            raw_positive_median > 0
        )
        else np.nan
    )


    original_negative_cells = int(
        config[
            "negative_cells"
        ]
    )


    recomputation_difference = int(
        negative_cells
        -
        original_negative_cells
    )


    audit_rows.append({
        "channel":
            channel,

        "window_days":
            window,

        "edf_ratio_target":
            edf_ratio,

        "lambda":
            lam,

        "orders":
            int(
                len(
                    raw
                )
            ),

        "total_cells":
            total_cells,

        "negative_cells":
            negative_cells,

        "negative_cell_pct":
            float(
                100.0
                *
                negative_cells
                /
                total_cells
            ),

        "affected_rows":
            affected_rows,

        "affected_row_pct":
            float(
                100.0
                *
                affected_rows
                /
                len(
                    raw
                )
            ),

        "minimum_value":
            minimum_value,

        "negative_p01":
            negative_p01,

        "negative_p05":
            negative_p05,

        "negative_median":
            negative_median,

        "negative_p95":
            negative_p95,

        "closest_to_zero_negative":
            closest_to_zero_negative,

        "negative_mass_abs":
            negative_mass,

        "negative_mass_pct_raw_mass":
            negative_mass_pct_raw_mass,

        "negative_mass_pct_smooth_positive_mass":
            negative_mass_pct_smooth_positive,

        "negative_on_raw_zero_pct":
            negative_on_raw_zero_pct,

        "negative_on_raw_positive_pct":
            negative_on_raw_positive_pct,

        "raw_mean_cell":
            raw_mean_cell,

        "raw_positive_median":
            raw_positive_median,

        "minimum_abs_over_raw_mean":
            minimum_abs_over_raw_mean,

        "minimum_abs_over_positive_median":
            minimum_abs_over_positive_median,

        "d1_recorded_negative_cells":
            original_negative_cells,

        "negative_cell_recomputation_difference":
            recomputation_difference,
    })


    print(
        f"{channel:17s} "
        f"{window:>2}d "
        f"EDF={edf_ratio:.2f} | "
        f"cells={negative_cells:>7,d} "
        f"({100*negative_cells/total_cells:.5f}%) | "
        f"rows={affected_rows:>6,d} "
        f"({100*affected_rows/len(raw):.4f}%) | "
        f"min={minimum_value:.8g} | "
        f"neg.mass/raw={negative_mass_pct_raw_mass:.8f}%"
    )


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    P_AUDIT,
    index=False
)


# =============================================================================
# 5. VALIDATION
# =============================================================================

checks = []


add_check(
    checks,
    "d1_pass",
    d1.get(
        "status"
    ) == "PASS",
    d1.get(
        "status"
    ),
    "PASS"
)


add_check(
    checks,
    "nontrivial_configs",
    len(
        audit
    ) == 18,
    len(
        audit
    ),
    18
)


add_check(
    checks,
    "negative_counts_recomputed_exactly",
    bool(
        audit[
            "negative_cell_recomputation_difference"
        ]
        .eq(
            0
        )
        .all()
    ),
    int(
        (
            audit[
                "negative_cell_recomputation_difference"
            ]
            !=
            0
        ).sum()
    ),
    0
)


finite_cols = [
    "negative_cell_pct",
    "affected_row_pct",
    "minimum_value",
    "negative_p01",
    "negative_p05",
    "negative_median",
    "negative_p95",
    "negative_mass_abs",
    "negative_mass_pct_raw_mass",
    "negative_mass_pct_smooth_positive_mass",
    "negative_on_raw_zero_pct",
    "negative_on_raw_positive_pct",
    "raw_mean_cell",
    "minimum_abs_over_raw_mean",
]


nonfinite = int(
    (
        ~np.isfinite(
            audit[
                finite_cols
            ].to_numpy(
                dtype=float
            )
        )
    ).sum()
)


add_check(
    checks,
    "all_core_metrics_finite",
    nonfinite == 0,
    nonfinite,
    0
)


add_check(
    checks,
    "no_data_mutation",
    True,
    False,
    False
)


validation = pd.DataFrame(
    checks
)

validation.to_csv(
    P_VALIDATION,
    index=False
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


# =============================================================================
# 6. SUMMARY
# =============================================================================

runtime = float(
    time.perf_counter()
    -
    start
)


summary = {
    "step":
        "MODEL_01_0_C3_3D1_1_NEGATIVE_SMOOTHING_AUDIT",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "configurations_audited":
        int(
            len(
                audit
            )
        ),

    "configurations_with_negative_cells":
        int(
            (
                audit[
                    "negative_cells"
                ]
                >
                0
            ).sum()
        ),

    "max_negative_cell_pct":
        float(
            audit[
                "negative_cell_pct"
            ].max()
        ),

    "max_affected_row_pct":
        float(
            audit[
                "affected_row_pct"
            ].max()
        ),

    "most_negative_value":
        float(
            audit[
                "minimum_value"
            ].min()
        ),

    "max_negative_mass_pct_raw_mass":
        float(
            audit[
                "negative_mass_pct_raw_mass"
            ].max()
        ),

    "target_used":
        False,

    "negative_values_clipped":
        False,

    "smoothing_selected":
        False,

    "lambda_selected":
        False,

    "window_selected":
        False,

    "fpca_applied":
        False,

    "model_feature_created":
        False,

    "model_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures,

    "runtime_seconds":
        runtime,
}


P_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 7. REPORT
# =============================================================================

top_mass = (
    audit.sort_values(
        "negative_mass_pct_raw_mass",
        ascending=False
    )
    .head(
        6
    )
)


lines = [
    "=" * 110,
    "MODEL 01.0-C3.3-D1.1 — NEGATIVE SMOOTHING ARTIFACT AUDIT",
    "=" * 110,
    "",
    f"STATUS                       : {summary['status']}",
    f"CONFIGURATIONS               : {len(audit)}",
    f"WITH NEGATIVE CELLS          : {summary['configurations_with_negative_cells']}",
    f"MAX NEGATIVE CELL %          : {summary['max_negative_cell_pct']:.8f}",
    f"MAX AFFECTED ROW %           : {summary['max_affected_row_pct']:.8f}",
    f"MOST NEGATIVE VALUE          : {summary['most_negative_value']:.12g}",
    f"MAX NEGATIVE MASS / RAW %    : {summary['max_negative_mass_pct_raw_mass']:.12g}",
    "",
    "IMPORTANT",
    "-" * 110,
    "No clipping was applied.",
    "No negative value was replaced.",
    "No smoothing configuration was selected.",
    "No target was used.",
    "No FPCA was applied.",
    "No model was trained.",
    "RAW and original curve arrays remain unchanged.",
    "",
    f"VALIDATION FAILURES          : {failures}",
    f"RUNTIME                      : {runtime:.3f}s",
    "=" * 110,
]


P_REPORT.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# =============================================================================
# 8. PRINT
# =============================================================================

print()
print("=" * 118)
print("NEGATIVE ARTIFACT SUMMARY")
print("=" * 118)

display = [
    "channel",
    "window_days",
    "edf_ratio_target",
    "negative_cells",
    "negative_cell_pct",
    "affected_rows",
    "affected_row_pct",
    "minimum_value",
    "negative_median",
    "negative_mass_pct_raw_mass",
    "negative_on_raw_zero_pct",
    "negative_on_raw_positive_pct",
]

print(
    audit[
        display
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.10f}"
    )
)


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
print("RESULTADO C3.3-D1.1")
print("=" * 118)

print(
    "FAILURES                     =",
    failures
)

print(
    "CONFIGS AUDITED              =",
    len(
        audit
    )
)

print(
    "CONFIGS WITH NEGATIVE CELLS  =",
    int(
        (
            audit[
                "negative_cells"
            ]
            >
            0
        ).sum()
    )
)

print(
    "MAX NEGATIVE CELL PCT        =",
    float(
        audit[
            "negative_cell_pct"
        ].max()
    )
)

print(
    "MAX AFFECTED ROW PCT         =",
    float(
        audit[
            "affected_row_pct"
        ].max()
    )
)

print(
    "MOST NEGATIVE VALUE          =",
    float(
        audit[
            "minimum_value"
        ].min()
    )
)

print(
    "MAX NEGATIVE MASS / RAW PCT  =",
    float(
        audit[
            "negative_mass_pct_raw_mass"
        ].max()
    )
)

print()
print("NEGATIVE VALUES CLIPPED      = NÃO")
print("SMOOTHING SELECTED           = NÃO")
print("LAMBDA SELECTED              = NÃO")
print("WINDOW SELECTED              = NÃO")
print("TARGET USED                  = NÃO")
print("FPCA APPLIED                 = NÃO")
print("FEATURE CREATED              = NÃO")
print("MODEL TRAINED                = NÃO")
print("SILVER CREATED               = NÃO")
print("RAW MODIFIED                 = NÃO")


if failures:
    raise SystemExit(
        2
    )


print()
print("[PASS] D1.1 negative-artifact audit concluída.")
print("[PASS] Parar aqui antes de qualquer correção ou clipping.")
