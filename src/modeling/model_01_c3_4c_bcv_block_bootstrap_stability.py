#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.4-C
TEMPORAL BLOCK-BOOTSTRAP STABILITY OF BCV RANK
===============================================================================

OBJETIVO
--------
Quantificar a estabilidade da escolha de rank K obtida pelo C3.4-B.

IMPORTANTE
----------
Este experimento NÃO:

- recalcula PCA;
- recalcula SVD;
- recalcula BCV;
- abre as matrizes funcionais .npy;
- usa target;
- treina classificador;
- escolhe K final;
- altera RAW.

A unidade de resampling é a evidência temporal BCV já calculada.

Como os meses possuem ordem temporal, usamos moving-block bootstrap
sobre os meses, preservando pequenos blocos consecutivos.

Blocos temporais testados:

    L = 2, 3, 4 meses

Réplicas:

    R = 2000 por configuração

Para cada réplica:

    K*_r = argmin_K E_r(K)

e então estimamos a frequência de seleção de cada K.
===============================================================================
"""

from pathlib import Path
from collections import Counter
import json
import math
import time

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
DIR = ROOT / "reports" / "modeling" / "model_01_order_logistic"

FOLD_ERRORS = DIR / "04m_temporal_bcv_fold_rank_errors.csv"
CANDIDATES = DIR / "04o_temporal_bcv_scheme_candidates.csv"
BCV_SUMMARY = DIR / "04r_temporal_bcv_summary.json"

MONTHLY_PROFILE = DIR / "04t_bcv_monthly_rank_profile.csv"
BOOT_FREQ = DIR / "04u_bcv_bootstrap_rank_frequency.csv"
BOOT_SUMMARY = DIR / "04v_bcv_bootstrap_stability_summary.csv"
VALIDATION = DIR / "04w_bcv_bootstrap_validation.csv"
SUMMARY_JSON = DIR / "04x_bcv_bootstrap_summary.json"
REPORT = DIR / "04y_bcv_bootstrap_report.txt"

R = 2000
BLOCK_LENGTHS = [2, 3, 4]
SEED = 20260830


# =============================================================================
# HELPERS
# =============================================================================

def detect_column(df, candidates, required=True):
    lower = {
        c.lower(): c
        for c in df.columns
    }

    # exact
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    # partial
    for candidate in candidates:
        token = candidate.lower()

        hits = [
            c
            for c in df.columns
            if token in c.lower()
        ]

        if len(hits) == 1:
            return hits[0]

    if required:
        raise RuntimeError(
            "Não consegui identificar coluna entre: "
            + str(candidates)
            + "\nColunas disponíveis:\n"
            + str(list(df.columns))
        )

    return None


def moving_block_indices(n, block_length, rng):
    """
    Moving block bootstrap não circular.

    Seleciona blocos consecutivos até completar n observações.
    """

    if block_length < 1:
        raise ValueError("block_length inválido")

    if block_length > n:
        raise ValueError("block_length > n")

    max_start = n - block_length

    pieces = []

    while sum(len(x) for x in pieces) < n:

        start = int(
            rng.integers(
                0,
                max_start + 1
            )
        )

        pieces.append(
            np.arange(
                start,
                start + block_length,
                dtype=int
            )
        )

    idx = np.concatenate(
        pieces
    )[:n]

    return idx


def normalized_entropy(values):
    """
    Entropia normalizada da distribuição empírica de K.
    0 -> seleção totalmente concentrada.
    1 -> seleção espalhada uniformemente pelos valores observados.
    """

    counts = np.array(
        list(
            Counter(values).values()
        ),
        dtype=float
    )

    if len(counts) <= 1:
        return 0.0

    p = counts / counts.sum()

    h = -np.sum(
        p * np.log(p)
    )

    return float(
        h / np.log(len(p))
    )


def smallest_argmin(k_values, score):
    """
    Em empate numérico escolhe o menor K.
    """

    minimum = np.nanmin(
        score
    )

    idx = np.where(
        np.isclose(
            score,
            minimum,
            rtol=1e-12,
            atol=1e-14
        )
    )[0]

    if len(idx) == 0:
        return int(
            k_values[
                np.nanargmin(score)
            ]
        )

    candidate_ks = (
        k_values[
            idx
        ]
    )

    return int(
        candidate_ks.min()
    )


# =============================================================================
# START
# =============================================================================

start = time.perf_counter()

print()
print("=" * 118)
print("MODEL 01.0-C3.4-C — TEMPORAL BLOCK-BOOTSTRAP BCV RANK STABILITY")
print("=" * 118)


# =============================================================================
# PREREQUISITES
# =============================================================================

for p in [
    FOLD_ERRORS,
    CANDIDATES,
    BCV_SUMMARY,
]:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Arquivo ausente: {p}"
        )

    print(
        f"[PASS] {p.name}"
    )


with BCV_SUMMARY.open(
    encoding="utf-8"
) as f:

    bcv_summary = json.load(f)


if bcv_summary.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "C3.4-B não está PASS."
    )


if bcv_summary.get(
    "final_k_selected",
    True
):

    raise RuntimeError(
        "C3.4-B já registra K final, "
        "o que contradiz este estágio."
    )


if bcv_summary.get(
    "target_used",
    True
):

    raise RuntimeError(
        "C3.4-B registra uso de target."
    )


print()
print("[PASS] C3.4-B formalmente validado.")
print("[PASS] K final ainda não congelado.")


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(
    FOLD_ERRORS
)

cand = pd.read_csv(
    CANDIDATES
)


# =============================================================================
# COLUMN DISCOVERY
# =============================================================================

channel_col = detect_column(
    df,
    [
        "channel",
    ]
)

scheme_col = detect_column(
    df,
    [
        "scheme",
        "holdout_scheme",
    ]
)

month_col = detect_column(
    df,
    [
        "current_month",
        "test_month",
        "purchase_month",
    ]
)

k_col = detect_column(
    df,
    [
        "k",
        "rank",
        "candidate_k",
    ]
)


sse_col = detect_column(
    df,
    [
        "sse",
        "prediction_sse",
        "holdout_sse",
        "bcv_sse",
    ],
    required=False
)

energy_col = detect_column(
    df,
    [
        "energy",
        "target_energy",
        "holdout_energy",
        "a_energy",
    ],
    required=False
)

rel_col = detect_column(
    df,
    [
        "relative_error",
        "fold_relative_error",
        "bcv_relative_error",
    ],
    required=False
)


print()
print("=" * 118)
print("DETECTED SCHEMA")
print("=" * 118)

print("channel :", channel_col)
print("scheme  :", scheme_col)
print("month   :", month_col)
print("K       :", k_col)
print("SSE     :", sse_col)
print("energy  :", energy_col)
print("rel.err :", rel_col)


if rel_col is None and (
    sse_col is None
    or
    energy_col is None
):

    raise RuntimeError(
        "Não existe informação suficiente para "
        "reconstruir o erro BCV."
    )


# =============================================================================
# STANDARDIZE
# =============================================================================

work = pd.DataFrame({
    "channel":
        df[channel_col].astype(str),

    "scheme":
        df[scheme_col].astype(str),

    "current_month":
        df[month_col].astype(str),

    "k":
        pd.to_numeric(
            df[k_col],
            errors="raise"
        ).astype(int),
})


if sse_col is not None:

    work["sse"] = pd.to_numeric(
        df[sse_col],
        errors="coerce"
    )


if energy_col is not None:

    work["energy"] = pd.to_numeric(
        df[energy_col],
        errors="coerce"
    )


if rel_col is not None:

    work["relative_error"] = pd.to_numeric(
        df[rel_col],
        errors="coerce"
    )


# =============================================================================
# CHOOSE OBJECTIVE REPRESENTATION
# =============================================================================

use_pooled = (
    "sse" in work.columns
    and
    "energy" in work.columns
    and
    work["sse"].notna().all()
    and
    work["energy"].notna().all()
    and
    (work["energy"] > 0).all()
)


if use_pooled:

    objective_method = (
        "SQRT_SUM_SSE_OVER_SUM_ENERGY"
    )

else:

    if "relative_error" not in work.columns:
        raise RuntimeError(
            "Erro relativo não disponível."
        )

    if work[
        "relative_error"
    ].isna().any():

        raise RuntimeError(
            "Há NaN em relative_error."
        )

    objective_method = (
        "MEAN_FOLD_RELATIVE_ERROR"
    )


print()
print(
    "Objective method:",
    objective_method
)


# =============================================================================
# MONTHLY K PROFILE
# =============================================================================

keys = [
    "channel",
    "scheme",
    "current_month",
    "k",
]


if use_pooled:

    monthly = (
        work
        .groupby(
            keys,
            as_index=False
        )
        .agg(
            sse=("sse", "sum"),
            energy=("energy", "sum"),
            fold_rows=("k", "size"),
        )
    )

    monthly[
        "relative_error"
    ] = np.sqrt(
        np.maximum(
            monthly["sse"],
            0.0
        )
        /
        monthly["energy"]
    )

else:

    monthly = (
        work
        .groupby(
            keys,
            as_index=False
        )
        .agg(
            relative_error=(
                "relative_error",
                "mean"
            ),
            fold_rows=(
                "k",
                "size"
            ),
        )
    )


monthly = monthly.sort_values(
    [
        "channel",
        "scheme",
        "current_month",
        "k",
    ]
).reset_index(
    drop=True
)


monthly.to_csv(
    MONTHLY_PROFILE,
    index=False
)


# =============================================================================
# REPRODUCE ORIGINAL BCV BEST K BEFORE BOOTSTRAP
# =============================================================================

baseline_rows = []


for (
    channel,
    scheme
), g in work.groupby(
    [
        "channel",
        "scheme"
    ]
):

    if use_pooled:

        p = (
            g
            .groupby(
                "k",
                as_index=False
            )
            .agg(
                sse=("sse", "sum"),
                energy=("energy", "sum"),
            )
        )

        p[
            "score"
        ] = np.sqrt(
            np.maximum(
                p["sse"],
                0.0
            )
            /
            p["energy"]
        )

    else:

        p = (
            g
            .groupby(
                "k",
                as_index=False
            )
            .agg(
                score=(
                    "relative_error",
                    "mean"
                )
            )
        )

    p = p.sort_values(
        "k"
    )

    best_k = smallest_argmin(
        p["k"].to_numpy(int),
        p["score"].to_numpy(float)
    )

    baseline_rows.append({
        "channel":
            channel,

        "scheme":
            scheme,

        "recomputed_best_k":
            best_k,
    })


baseline = pd.DataFrame(
    baseline_rows
)


cand_key = cand[
    [
        "channel",
        "scheme",
        "best_bcv_k",
    ]
].copy()


baseline = baseline.merge(
    cand_key,
    on=[
        "channel",
        "scheme",
    ],
    how="left",
    validate="one_to_one"
)


baseline[
    "match"
] = (
    baseline[
        "recomputed_best_k"
    ]
    ==
    baseline[
        "best_bcv_k"
    ]
)


print()
print("=" * 118)
print("BASELINE REPRODUCTION")
print("=" * 118)

print(
    baseline.to_string(
        index=False
    )
)


baseline_failures = int(
    (
        ~baseline["match"]
    ).sum()
)


if baseline_failures:

    raise RuntimeError(
        "A função objetivo reconstruída não reproduz "
        "os candidatos do C3.4-B. "
        "Monte Carlo interrompido."
    )


print()
print(
    "[PASS] 6/6 candidatos BCV reproduzidos "
    "antes do bootstrap."
)


# =============================================================================
# TEMPORAL BLOCK BOOTSTRAP
# =============================================================================

rng = np.random.default_rng(
    SEED
)

frequency_rows = []
summary_rows = []


print()
print("=" * 118)
print("BLOCK BOOTSTRAP")
print("=" * 118)


for (
    channel,
    scheme
), g in monthly.groupby(
    [
        "channel",
        "scheme"
    ]
):

    months = sorted(
        g[
            "current_month"
        ].unique()
    )

    n_months = len(
        months
    )

    ks = np.array(
        sorted(
            g[
                "k"
            ].unique()
        ),
        dtype=int
    )


    # -------------------------------------------------------------------------
    # matrices month x K
    # -------------------------------------------------------------------------

    err_matrix = np.full(
        (
            n_months,
            len(ks)
        ),
        np.nan,
        dtype=float
    )


    if use_pooled:

        sse_matrix = np.full_like(
            err_matrix,
            np.nan
        )

        energy_matrix = np.full_like(
            err_matrix,
            np.nan
        )


    month_lookup = {
        m: i
        for i, m in enumerate(
            months
        )
    }

    k_lookup = {
        int(k): j
        for j, k in enumerate(
            ks
        )
    }


    for row in g.itertuples(
        index=False
    ):

        i = month_lookup[
            str(
                row.current_month
            )
        ]

        j = k_lookup[
            int(
                row.k
            )
        ]

        err_matrix[
            i,
            j
        ] = float(
            row.relative_error
        )

        if use_pooled:

            sse_matrix[
                i,
                j
            ] = float(
                row.sse
            )

            energy_matrix[
                i,
                j
            ] = float(
                row.energy
            )


    if not np.isfinite(
        err_matrix
    ).all():

        raise RuntimeError(
            f"Perfil mensal incompleto: "
            f"{channel} / {scheme}"
        )


    if use_pooled:

        if not np.isfinite(
            sse_matrix
        ).all():

            raise RuntimeError(
                "SSE mensal incompleto."
            )

        if not np.isfinite(
            energy_matrix
        ).all():

            raise RuntimeError(
                "Energy mensal incompleto."
            )


    for L in BLOCK_LENGTHS:

        selected = np.empty(
            R,
            dtype=int
        )

        for r in range(R):

            idx = moving_block_indices(
                n_months,
                L,
                rng
            )


            if use_pooled:

                sse_sum = (
                    sse_matrix[
                        idx,
                        :
                    ]
                    .sum(
                        axis=0
                    )
                )

                energy_sum = (
                    energy_matrix[
                        idx,
                        :
                    ]
                    .sum(
                        axis=0
                    )
                )

                score = np.sqrt(
                    np.maximum(
                        sse_sum,
                        0.0
                    )
                    /
                    energy_sum
                )

            else:

                score = (
                    err_matrix[
                        idx,
                        :
                    ]
                    .mean(
                        axis=0
                    )
                )


            selected[
                r
            ] = smallest_argmin(
                ks,
                score
            )


        counts = Counter(
            selected.tolist()
        )

        mode_k, mode_count = sorted(
            counts.items(),
            key=lambda x: (
                -x[1],
                x[0]
            )
        )[0]


        for k in ks:

            count = int(
                counts.get(
                    int(k),
                    0
                )
            )

            frequency_rows.append({
                "channel":
                    channel,

                "scheme":
                    scheme,

                "block_length_months":
                    L,

                "k":
                    int(k),

                "selection_count":
                    count,

                "selection_pct":
                    100.0
                    *
                    count
                    /
                    R,
            })


        summary_rows.append({
            "channel":
                channel,

            "scheme":
                scheme,

            "block_length_months":
                L,

            "bootstrap_replicates":
                R,

            "months":
                n_months,

            "mode_k":
                int(
                    mode_k
                ),

            "mode_selection_pct":
                100.0
                *
                mode_count
                /
                R,

            "median_k":
                float(
                    np.median(
                        selected
                    )
                ),

            "q05_k":
                float(
                    np.quantile(
                        selected,
                        0.05
                    )
                ),

            "q95_k":
                float(
                    np.quantile(
                        selected,
                        0.95
                    )
                ),

            "min_selected_k":
                int(
                    selected.min()
                ),

            "max_selected_k":
                int(
                    selected.max()
                ),

            "unique_selected_k":
                int(
                    len(
                        np.unique(
                            selected
                        )
                    )
                ),

            "normalized_selection_entropy":
                normalized_entropy(
                    selected.tolist()
                ),

            "pct_k_le_6":
                float(
                    100.0
                    *
                    (
                        selected
                        <= 6
                    ).mean()
                ),

            "pct_k_le_10":
                float(
                    100.0
                    *
                    (
                        selected
                        <= 10
                    ).mean()
                ),
        })


        print(
            f"{channel:18s} "
            f"{scheme:26s} "
            f"L={L} | "
            f"mode K={mode_k:2d} "
            f"({100*mode_count/R:6.2f}%) | "
            f"median={np.median(selected):5.1f} | "
            f"P05-P95="
            f"{np.quantile(selected,0.05):4.1f}"
            f".."
            f"{np.quantile(selected,0.95):4.1f} | "
            f"H={normalized_entropy(selected.tolist()):.4f}"
        )


freq = pd.DataFrame(
    frequency_rows
)

summary = pd.DataFrame(
    summary_rows
)


freq.to_csv(
    BOOT_FREQ,
    index=False
)

summary.to_csv(
    BOOT_SUMMARY,
    index=False
)


# =============================================================================
# CROSS-BLOCK SENSITIVITY
# =============================================================================

cross = (
    summary
    .groupby(
        [
            "channel",
            "scheme",
        ],
        as_index=False
    )
    .agg(
        block_lengths_tested=(
            "block_length_months",
            "nunique"
        ),
        mode_k_min=(
            "mode_k",
            "min"
        ),
        mode_k_max=(
            "mode_k",
            "max"
        ),
        median_mode_selection_pct=(
            "mode_selection_pct",
            "median"
        ),
        max_entropy=(
            "normalized_selection_entropy",
            "max"
        ),
        median_entropy=(
            "normalized_selection_entropy",
            "median"
        ),
        min_pct_k_le_6=(
            "pct_k_le_6",
            "min"
        ),
        max_pct_k_le_6=(
            "pct_k_le_6",
            "max"
        ),
    )
)


# =============================================================================
# VALIDATION
# =============================================================================

rows = []


def add(
    check,
    ok,
    observed,
    expected
):
    rows.append({
        "check":
            check,

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
    "c34b_pass",
    bcv_summary.get(
        "status"
    ) == "PASS",
    bcv_summary.get(
        "status"
    ),
    "PASS"
)

add(
    "original_candidates_reproduced",
    baseline_failures == 0,
    baseline_failures,
    0
)

add(
    "six_channel_scheme_groups",
    len(
        baseline
    ) == 6,
    len(
        baseline
    ),
    6
)

add(
    "three_block_lengths",
    summary[
        "block_length_months"
    ].nunique() == 3,
    summary[
        "block_length_months"
    ].nunique(),
    3
)

add(
    "eighteen_bootstrap_groups",
    len(
        summary
    ) == 18,
    len(
        summary
    ),
    18
)

add(
    "all_bootstrap_replicates_complete",
    int(
        freq[
            "selection_count"
        ].sum()
    )
    ==
    (
        6
        *
        3
        *
        R
    ),
    int(
        freq[
            "selection_count"
        ].sum()
    ),
    6 * 3 * R
)

add(
    "selection_pct_sums_100",
    float(
        (
            freq
            .groupby(
                [
                    "channel",
                    "scheme",
                    "block_length_months",
                ]
            )[
                "selection_pct"
            ]
            .sum()
            -
            100.0
        )
        .abs()
        .max()
    )
    < 1e-10,
    float(
        (
            freq
            .groupby(
                [
                    "channel",
                    "scheme",
                    "block_length_months",
                ]
            )[
                "selection_pct"
            ]
            .sum()
            -
            100.0
        )
        .abs()
        .max()
    ),
    "<1e-10"
)

add(
    "all_metrics_finite",
    int(
        np.isfinite(
            summary.select_dtypes(
                include=[
                    np.number
                ]
            )
            .to_numpy()
        )
        .size
        -
        np.isfinite(
            summary.select_dtypes(
                include=[
                    np.number
                ]
            )
            .to_numpy()
        )
        .sum()
    )
    == 0,
    int(
        (
            ~np.isfinite(
                summary.select_dtypes(
                    include=[
                        np.number
                    ]
                )
                .to_numpy()
            )
        ).sum()
    ),
    0
)

add(
    "target_not_used",
    True,
    False,
    False
)

add(
    "final_k_not_selected",
    True,
    False,
    False
)


validation = pd.DataFrame(
    rows
)

validation.to_csv(
    VALIDATION,
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
# REPORT
# =============================================================================

elapsed = (
    time.perf_counter()
    -
    start
)


summary_payload = {
    "step":
        "MODEL_01_0_C3_4C_TEMPORAL_BLOCK_BOOTSTRAP_BCV_STABILITY",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "representation":
        "RAW_30D",

    "source_method":
        "TEMPORAL_BLOCK_BICROSSVALIDATION",

    "bootstrap_method":
        "MOVING_BLOCK_BOOTSTRAP_OF_OUT_OF_TIME_BCV_EVIDENCE",

    "important_scope":
        "STABILITY_OF_BCV_RANK_SELECTION_NOT_FULL_DATA_GENERATIVE_MONTE_CARLO",

    "bootstrap_replicates_per_group":
        R,

    "block_lengths_months":
        BLOCK_LENGTHS,

    "seed":
        SEED,

    "objective_method":
        objective_method,

    "channel_scheme_groups":
        6,

    "bootstrap_groups":
        len(
            summary
        ),

    "original_candidates_reproduced":
        bool(
            baseline_failures
            ==
            0
        ),

    "target_used":
        False,

    "final_k_selected":
        False,

    "adaptive_evidence_run":
        False,

    "classifier_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures,

    "runtime_seconds":
        elapsed,
}


SUMMARY_JSON.write_text(
    json.dumps(
        summary_payload,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


report_lines = [
    "=" * 118,
    "MODEL 01.0-C3.4-C — TEMPORAL BLOCK-BOOTSTRAP BCV RANK STABILITY",
    "=" * 118,
    "",
    f"STATUS                     : {summary_payload['status']}",
    f"REPLICATES/GROUP           : {R}",
    f"BLOCK LENGTHS              : {BLOCK_LENGTHS}",
    f"OBJECTIVE                  : {objective_method}",
    "",
    "IMPORTANT",
    "-" * 118,
    "This is a bootstrap of already computed out-of-time BCV evidence.",
    "It quantifies rank-selection stability under temporal resampling.",
    "It is NOT a generative Monte Carlo model of raw Olist observations.",
    "",
    "CROSS-BLOCK SUMMARY",
    "-" * 118,
    cross.to_string(index=False),
    "",
    "GOVERNANCE",
    "-" * 118,
    "Target used              : NO",
    "Final K selected         : NO",
    "Adaptive evidence run    : NO",
    "Classifier trained       : NO",
    "Silver created           : NO",
    "RAW modified             : NO",
    "",
    f"Validation failures      : {failures}",
    f"Runtime seconds          : {elapsed:.4f}",
    "=" * 118,
]


REPORT.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8"
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 118)
print("BOOTSTRAP STABILITY SUMMARY")
print("=" * 118)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print()
print("=" * 118)
print("CROSS-BLOCK SENSITIVITY")
print("=" * 118)

print(
    cross.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
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
print("RESULTADO C3.4-C")
print("=" * 118)

print(
    "STATUS                  =",
    summary_payload[
        "status"
    ]
)

print(
    "BOOTSTRAP GROUPS        =",
    len(
        summary
    )
)

print(
    "REPLICATES / GROUP      =",
    R
)

print(
    "TOTAL SELECTIONS        =",
    int(
        freq[
            "selection_count"
        ].sum()
    )
)

print()
print(
    "TARGET USED             = NÃO"
)
print(
    "K FINAL                 = NÃO"
)
print(
    "ADAPTIVE                = AINDA NÃO"
)
print(
    "CLASSIFIER TRAINED      = NÃO"
)
print(
    "SILVER CREATED          = NÃO"
)
print(
    "RAW MODIFIED            = NÃO"
)


if failures:

    raise SystemExit(
        2
    )


print()
print(
    "[PASS] C3.4-C concluído."
)
print(
    "[PASS] Parar antes do Adaptive Evidence."
)

