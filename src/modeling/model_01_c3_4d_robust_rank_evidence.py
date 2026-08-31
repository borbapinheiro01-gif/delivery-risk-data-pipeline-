#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.4-D
ROBUST RANK EVIDENCE SYNTHESIS
===============================================================================

Consolida:

C3.4-A.1  -> reconstruction criterion is non-identifying
C3.4-B    -> temporal block BCV
C3.4-C    -> temporal block-bootstrap stability

NÃO:
- executa PCA/SVD;
- recalcula BCV;
- recalcula bootstrap;
- usa target;
- inventa pesos;
- usa softmax;
- escolhe K final;
- treina modelo;
- altera RAW.

Para cada grupo g:

    p_g(k) = frequência bootstrap de seleção de k

e:

    C_g(K) = sum_{k <= K} p_g(k)

A evidência robusta é:

    C_min(K) = min_g C_g(K)

Também são reportados:

    C_median(K)
    C_mean(K)

Nenhum threshold é promovido automaticamente.
===============================================================================
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
DIR = ROOT / "reports/modeling/model_01_order_logistic"

FREQ = DIR / "04u_bcv_bootstrap_rank_frequency.csv"
BOOT = DIR / "04v_bcv_bootstrap_stability_summary.csv"
BCV = DIR / "04o_temporal_bcv_scheme_candidates.csv"
BOOT_JSON = DIR / "04x_bcv_bootstrap_summary.json"

OUT_CURVE = DIR / "05a_robust_rank_cumulative_evidence.csv"
OUT_THRESH = DIR / "05b_robust_rank_threshold_sensitivity.csv"
OUT_MODES = DIR / "05c_scheme_conditioned_modes.csv"
OUT_VALID = DIR / "05d_robust_rank_evidence_validation.csv"
OUT_JSON = DIR / "05e_robust_rank_evidence_summary.json"
OUT_REPORT = DIR / "05f_robust_rank_evidence_report.txt"


print()
print("=" * 118)
print("MODEL 01.0-C3.4-D — ROBUST RANK EVIDENCE SYNTHESIS")
print("=" * 118)


# =============================================================================
# PREREQUISITES
# =============================================================================

for p in [FREQ, BOOT, BCV, BOOT_JSON]:

    if not p.exists():
        raise SystemExit(
            f"[FAIL] Ausente: {p}"
        )

    print(
        f"[PASS] {p.name}"
    )


with BOOT_JSON.open(encoding="utf-8") as f:
    js = json.load(f)


if js.get("status") != "PASS":
    raise RuntimeError(
        "C3.4-C não está PASS."
    )

if js.get("final_k_selected", True):
    raise RuntimeError(
        "K final já consta selecionado."
    )

if js.get("target_used", True):
    raise RuntimeError(
        "Bootstrap registra target usado."
    )


freq = pd.read_csv(FREQ)
boot = pd.read_csv(BOOT)
bcv = pd.read_csv(BCV)


# =============================================================================
# BASIC VALIDATION
# =============================================================================

required_freq = {
    "channel",
    "scheme",
    "block_length_months",
    "k",
    "selection_count",
    "selection_pct",
}

missing = required_freq - set(freq.columns)

if missing:
    raise RuntimeError(
        f"Colunas ausentes em 04u: {sorted(missing)}"
    )


group_cols = [
    "channel",
    "scheme",
    "block_length_months",
]


# =============================================================================
# COMPLETE K GRID
# =============================================================================

min_k = int(
    min(
        0,
        freq["k"].min()
    )
)

max_k = int(
    freq["k"].max()
)

all_k = list(
    range(
        min_k,
        max_k + 1
    )
)


groups = (
    freq[group_cols]
    .drop_duplicates()
    .sort_values(group_cols)
    .reset_index(drop=True)
)


records = []


for g in groups.itertuples(index=False):

    channel = g.channel
    scheme = g.scheme
    L = g.block_length_months

    x = freq.loc[
        (freq["channel"] == channel)
        &
        (freq["scheme"] == scheme)
        &
        (freq["block_length_months"] == L),
        ["k", "selection_pct"]
    ].copy()

    p = {
        int(row.k):
            float(row.selection_pct) / 100.0
        for row in x.itertuples(index=False)
    }

    cumulative = 0.0

    for k in all_k:

        cumulative += p.get(
            k,
            0.0
        )

        records.append({
            "channel": channel,
            "scheme": scheme,
            "block_length_months": L,
            "k_cap": k,
            "selection_probability_at_k":
                p.get(k, 0.0),
            "cumulative_probability_k_le_cap":
                cumulative,
        })


cdf = pd.DataFrame(records)


# =============================================================================
# ROBUST ENVELOPE
# =============================================================================

curve = (
    cdf.groupby(
        "k_cap",
        as_index=False
    )
    .agg(
        worst_case_cumulative_probability=(
            "cumulative_probability_k_le_cap",
            "min"
        ),
        median_cumulative_probability=(
            "cumulative_probability_k_le_cap",
            "median"
        ),
        mean_cumulative_probability=(
            "cumulative_probability_k_le_cap",
            "mean"
        ),
        best_case_cumulative_probability=(
            "cumulative_probability_k_le_cap",
            "max"
        ),
    )
)


curve[
    "worst_case_pct"
] = (
    100.0
    *
    curve[
        "worst_case_cumulative_probability"
    ]
)

curve[
    "median_pct"
] = (
    100.0
    *
    curve[
        "median_cumulative_probability"
    ]
)

curve[
    "mean_pct"
] = (
    100.0
    *
    curve[
        "mean_cumulative_probability"
    ]
)

curve.to_csv(
    OUT_CURVE,
    index=False
)


# =============================================================================
# THRESHOLD SENSITIVITY
# =============================================================================

thresholds = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
]


threshold_rows = []


for threshold in thresholds:

    eligible = curve.loc[
        curve[
            "worst_case_cumulative_probability"
        ]
        >= threshold
    ]

    if len(eligible):

        robust_cap = int(
            eligible.iloc[0][
                "k_cap"
            ]
        )

    else:

        robust_cap = None


    threshold_rows.append({
        "required_worst_case_probability":
            threshold,

        "required_worst_case_pct":
            100.0 * threshold,

        "smallest_k_cap_meeting_requirement":
            robust_cap,

        "candidate_only_not_final":
            True,
    })


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    OUT_THRESH,
    index=False
)


# =============================================================================
# MODE CONSISTENCY BY SCHEME
# =============================================================================

mode_summary = (
    boot.groupby(
        [
            "channel",
            "scheme",
        ],
        as_index=False
    )
    .agg(
        block_lengths=(
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
        mode_selection_pct_min=(
            "mode_selection_pct",
            "min"
        ),
        mode_selection_pct_median=(
            "mode_selection_pct",
            "median"
        ),
        mode_selection_pct_max=(
            "mode_selection_pct",
            "max"
        ),
        entropy_min=(
            "normalized_selection_entropy",
            "min"
        ),
        entropy_median=(
            "normalized_selection_entropy",
            "median"
        ),
        entropy_max=(
            "normalized_selection_entropy",
            "max"
        ),
    )
)


mode_summary[
    "mode_stable_across_temporal_block_lengths"
] = (
    mode_summary[
        "mode_k_min"
    ]
    ==
    mode_summary[
        "mode_k_max"
    ]
)


mode_summary.to_csv(
    OUT_MODES,
    index=False
)


# =============================================================================
# CENTRAL FACTS
# =============================================================================

exact_modes = sorted(
    set(
        mode_summary[
            "mode_k_min"
        ].astype(int)
    )
)


global_exact_k_consensus = (
    len(exact_modes)
    ==
    1
)


k6 = curve.loc[
    curve["k_cap"] == 6
].iloc[0]


k10 = curve.loc[
    curve["k_cap"] == 10
].iloc[0]


best_bcv_range = [
    int(
        bcv[
            "best_bcv_k"
        ].min()
    ),
    int(
        bcv[
            "best_bcv_k"
        ].max()
    ),
]


one_se_range = [
    int(
        bcv[
            "one_se_candidate_k"
        ].min()
    ),
    int(
        bcv[
            "one_se_candidate_k"
        ].max()
    ),
]


# =============================================================================
# VALIDATION
# =============================================================================

checks = []


def add(name, ok, observed, expected):

    checks.append({
        "check": name,
        "status":
            "PASS"
            if ok
            else "FAIL",
        "observed": observed,
        "expected": expected,
    })


add(
    "c34c_pass",
    js.get("status") == "PASS",
    js.get("status"),
    "PASS"
)

add(
    "bootstrap_groups",
    len(groups) == 18,
    len(groups),
    18
)

add(
    "scheme_channel_groups",
    len(mode_summary) == 6,
    len(mode_summary),
    6
)

add(
    "mode_stable_within_each_scheme",
    mode_summary[
        "mode_stable_across_temporal_block_lengths"
    ].all(),
    int(
        (
            ~mode_summary[
                "mode_stable_across_temporal_block_lengths"
            ]
        ).sum()
    ),
    0
)

add(
    "cdf_monotonic",
    (
        curve[
            "worst_case_cumulative_probability"
        ]
        .diff()
        .fillna(0)
        >= -1e-12
    ).all(),
    True,
    True
)

add(
    "cdf_reaches_one",
    abs(
        float(
            curve.iloc[-1][
                "worst_case_cumulative_probability"
            ]
        )
        -
        1.0
    )
    < 1e-10,
    float(
        curve.iloc[-1][
            "worst_case_cumulative_probability"
        ]
    ),
    1.0
)

add(
    "all_probabilities_valid",
    (
        (
            cdf[
                "cumulative_probability_k_le_cap"
            ]
            >= -1e-12
        )
        &
        (
            cdf[
                "cumulative_probability_k_le_cap"
            ]
            <= 1.0 + 1e-12
        )
    ).all(),
    True,
    True
)

add(
    "target_not_used",
    js.get("target_used") is False,
    js.get("target_used"),
    False
)

add(
    "final_k_not_selected",
    js.get("final_k_selected") is False,
    js.get("final_k_selected"),
    False
)


validation = pd.DataFrame(
    checks
)

validation.to_csv(
    OUT_VALID,
    index=False
)


failures = int(
    validation[
        "status"
    ].eq("FAIL").sum()
)


# =============================================================================
# INTERPRETATION
# =============================================================================

if global_exact_k_consensus:

    exact_state = (
        "EXACT_K_CONSENSUS_OBSERVED"
    )

else:

    exact_state = (
        "EXACT_K_NOT_IDENTIFIED_ACROSS_HOLDOUT_SCHEMES"
    )


summary = {
    "step":
        "MODEL_01_0_C3_4D_ROBUST_RANK_EVIDENCE",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "method":
        "WORST_CASE_CUMULATIVE_BOOTSTRAP_SELECTION_EVIDENCE",

    "bootstrap_groups":
        len(groups),

    "scheme_channel_groups":
        len(mode_summary),

    "scheme_conditioned_modal_ranks":
        exact_modes,

    "exact_rank_assessment":
        exact_state,

    "exact_k_global_consensus":
        bool(
            global_exact_k_consensus
        ),

    "best_bcv_k_range":
        best_bcv_range,

    "one_se_bcv_candidate_range":
        one_se_range,

    "worst_case_probability_k_le_6":
        float(
            k6[
                "worst_case_cumulative_probability"
            ]
        ),

    "worst_case_probability_k_le_10":
        float(
            k10[
                "worst_case_cumulative_probability"
            ]
        ),

    "threshold_sensitivity":
        threshold_rows,

    "important_interpretation":
        (
            "Evidence supports low-rank structure, "
            "but does not identify one exact global K."
        ),

    "adaptive_softmax_used":
        False,

    "arbitrary_evidence_weights_used":
        False,

    "target_used":
        False,

    "final_k_selected":
        False,

    "supervised_stage_unlocked":
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


OUT_JSON.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


report = f"""
======================================================================================================================
MODEL 01.0-C3.4-D — ROBUST RANK EVIDENCE SYNTHESIS
======================================================================================================================

STATUS
----------------------------------------------------------------------------------------------------------------------
Status                              : {summary["status"]}

SCHEME-CONDITIONED MODES
----------------------------------------------------------------------------------------------------------------------
{mode_summary.to_string(index=False)}

ROBUST CUMULATIVE EVIDENCE
----------------------------------------------------------------------------------------------------------------------
Worst-case P(K <= 6)                : {100*summary["worst_case_probability_k_le_6"]:.4f}%
Worst-case P(K <= 10)               : {100*summary["worst_case_probability_k_le_10"]:.4f}%

BCV RANGE
----------------------------------------------------------------------------------------------------------------------
Best BCV K range                    : {best_bcv_range}
1-SE candidate range                : {one_se_range}

EXACT-RANK ASSESSMENT
----------------------------------------------------------------------------------------------------------------------
{exact_state}

IMPORTANT
----------------------------------------------------------------------------------------------------------------------
No arbitrary evidence weighting is used.
No softmax is interpreted as posterior probability.
No exact K is selected.
Target is not used.
Classifier is not trained.
RAW is not modified.

NEXT
----------------------------------------------------------------------------------------------------------------------
Use this structural evidence to define the supervised temporal component-selection experiment.

Validation failures                 : {failures}
======================================================================================================================
""".strip()


OUT_REPORT.write_text(
    report,
    encoding="utf-8"
)


# =============================================================================
# PRINT
# =============================================================================

print()
print("=" * 118)
print("SCHEME-CONDITIONED MODE STABILITY")
print("=" * 118)

print(
    mode_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print()
print("=" * 118)
print("ROBUST CUMULATIVE EVIDENCE — SELECTED CAPS")
print("=" * 118)

print(
    curve.loc[
        curve["k_cap"].isin(
            [1, 2, 3, 4, 5, 6, 10, 15, 20, max_k]
        )
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print()
print("=" * 118)
print("THRESHOLD SENSITIVITY — NÃO É DECISÃO FINAL")
print("=" * 118)

print(
    threshold_df.to_string(
        index=False
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
print("RESULTADO C3.4-D")
print("=" * 118)

print(
    "SCHEME MODAL K          =",
    exact_modes
)

print(
    "EXACT K CONSENSUS       =",
    global_exact_k_consensus
)

print(
    "WORST P(K<=6)           =",
    f"{100*summary['worst_case_probability_k_le_6']:.4f}%"
)

print(
    "BEST BCV RANGE          =",
    best_bcv_range
)

print(
    "1-SE BCV RANGE          =",
    one_se_range
)

print()
print("K FINAL                 = NÃO")
print("TARGET                  = NÃO")
print("ADAPTIVE SOFTMAX        = NÃO")
print("MODELO                  = NÃO TREINADO")
print("SILVER                  = NÃO CRIADA")
print("RAW                     = INTACTO")


if failures:
    raise SystemExit(2)


print()
print("[PASS] C3.4-D concluído.")
print("[PASS] Evidência estrutural consolidada sem fabricar um K exato.")
print("[PASS] Parar antes da etapa supervisionada.")

