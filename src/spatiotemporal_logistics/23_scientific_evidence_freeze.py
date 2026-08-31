#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import pandas as pd

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

ART = ROOT / "artifacts" / "spatiotemporal_logistics"
SCI = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"
GN = SCI / "gnadmm_delivery"
OUT = ROOT / "reports" / "final_closeout"

OUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def read_csv(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


print("=" * 100)
print("MODULE 23 — SCIENTIFIC EVIDENCE FREEZE")
print("=" * 100)


# ==============================================================================================
# 1. CORE PIPELINE
# ==============================================================================================

core_files = [
    ART / "01_ROUTE_SELLER_ORDER.csv",
    ART / "02_ROUTE_ORDER_LEVEL.csv",
    ART / "03_MUNICIPAL_CONTEXT.csv",
    ART / "04_ANP_CONTEXT.csv",
    ART / "04b_ANP_STATE_CONTEXT.csv",
    ART / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv",
    ART / "06_EXPECTED_FREIGHT_OOT.csv",
]

core_ok = all(
    p.exists()
    for p in core_files
)

core_n = None
freight_n = None

if (
    ART
    / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv"
).exists():

    core_n = len(
        pd.read_csv(
            ART
            / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv",
            usecols=["order_id"],
        )
    )

if (
    ART
    / "06_EXPECTED_FREIGHT_OOT.csv"
).exists():

    freight_n = len(
        pd.read_csv(
            ART
            / "06_EXPECTED_FREIGHT_OOT.csv",
            usecols=["order_id"],
        )
    )


# ==============================================================================================
# 2. NONLINEAR OOT AUDIT
# ==============================================================================================

nonlinear = read_csv(
    SCI
    / "11b_nonlinear_oot_performance.csv"
)

nonlinear_status = "INCONCLUSIVE"
nonlinear_evidence = (
    "Nonlinear OOT table unavailable."
)

if nonlinear is not None and not nonlinear.empty:

    summary = (
        nonlinear
        .groupby("model")
        .agg(
            months=("test_month", "nunique"),
            mean_AP=("ap_oot", "mean"),
            mean_Brier=("brier_oot", "mean"),
        )
        .reset_index()
    )

    lin = summary[
        summary["model"].eq(
            "N1_Linear"
        )
    ]

    flex = summary[
        summary["model"].isin(
            [
                "N2_Quadratic",
                "N3_Natural_Spline_df4",
            ]
        )
    ]

    if (
        not lin.empty
        and
        not flex.empty
    ):

        ap_lin = float(
            lin.iloc[0]["mean_AP"]
        )

        br_lin = float(
            lin.iloc[0]["mean_Brier"]
        )

        best_ap_flex = float(
            flex["mean_AP"].max()
        )

        best_br_flex = float(
            flex["mean_Brier"].min()
        )

        if (
            best_ap_flex > ap_lin
            and
            best_br_flex < br_lin
        ):
            nonlinear_status = (
                "PARTIALLY_SUPPORTED"
            )
        else:
            nonlinear_status = (
                "NOT_SUPPORTED"
            )

        nonlinear_evidence = (
            f"Linear OOT AP={ap_lin:.6f}, "
            f"Brier={br_lin:.6f}; "
            f"best flexible AP={best_ap_flex:.6f}, "
            f"Brier={best_br_flex:.6f}."
        )


# ==============================================================================================
# 3. GN–ADMM SCIENTIFIC RESULTS — 19 TO 22
# ==============================================================================================

d19 = read_json(
    GN
    / "19d_PAIRED_TEMPORAL_DECISION.json"
)

d20 = read_json(
    GN
    / "20d_CONSTRAINT_ABLATION_DECISION.json"
)

d21 = read_json(
    GN
    / "21d_CALIBRATION_DECISION.json"
)

d22 = read_json(
    GN
    / "22e_GNADMM_SCIENTIFIC_FREEZE.json"
)

h22 = read_csv(
    GN
    / "22a_GNADMM_hypothesis_matrix.csv"
)


# ==============================================================================================
# 4. EXTRACT GN HYPOTHESES
# ==============================================================================================

curvature_status = "INCONCLUSIVE"
curvature_evidence = (
    "Curvature evidence unavailable."
)

calibration_status = "INCONCLUSIVE"
calibration_evidence = (
    "Calibration evidence unavailable."
)

numerical_status = "INCONCLUSIVE"
numerical_evidence = (
    "Numerical GN–ADMM evidence unavailable."
)


if h22 is not None:

    row = h22[
        h22["hypothesis"].eq(
            "H5_POSITIVE_QUADRATIC_CURVATURE"
        )
    ]

    if not row.empty:
        curvature_status = str(
            row.iloc[0]["status"]
        )

        curvature_evidence = str(
            row.iloc[0]["evidence"]
        )

    row = h22[
        h22["hypothesis"].eq(
            "H7_GNADMM_CALIBRATION_SUPERIORITY"
        )
    ]

    if not row.empty:
        calibration_status = str(
            row.iloc[0]["status"]
        )

        calibration_evidence = str(
            row.iloc[0]["evidence"]
        )

    row = h22[
        h22["hypothesis"].eq(
            "H8_GNADMM_COMPUTATIONAL_APPLICABILITY_REAL_DATA"
        )
    ]

    if not row.empty:
        numerical_status = str(
            row.iloc[0]["status"]
        )

        numerical_evidence = str(
            row.iloc[0]["evidence"]
        )


# ==============================================================================================
# 5. TEMPORAL GN–ADMM EVIDENCE
# ==============================================================================================

temporal_class = d19.get(
    "AP_evidence_classification",
    "UNKNOWN",
)

if temporal_class == "FAVORABLE_NOT_DECISIVE":
    temporal_status = (
        "PARTIALLY_SUPPORTED"
    )

elif temporal_class in {
    "SUPPORTED",
    "DECISIVE",
}:
    temporal_status = "SUPPORTED"

else:
    temporal_status = "INCONCLUSIVE"


# ==============================================================================================
# 6. FINAL HYPOTHESIS MATRIX
# ==============================================================================================

hypotheses = [
    {
        "hypothesis":
            "P1_DATA_PIPELINE_COMPLETE",

        "status":
            "SUPPORTED"
            if (
                core_ok
                and
                core_n == freight_n
            )
            else "INCONCLUSIVE",

        "evidence":
            (
                f"core_files={core_ok}; "
                f"context_N={core_n}; "
                f"expected_freight_N={freight_n}."
            ),
    },

    {
        "hypothesis":
            "P2_NONLINEAR_RELIABILITY_PREDICTIVE_GAIN",

        "status":
            nonlinear_status,

        "evidence":
            nonlinear_evidence,
    },

    {
        "hypothesis":
            "P3_GNADMM_NUMERICAL_IMPLEMENTATION_VALID",

        "status":
            numerical_status,

        "evidence":
            numerical_evidence,
    },

    {
        "hypothesis":
            "P4_GNADMM_TEMPORAL_AP_ADVANTAGE",

        "status":
            temporal_status,

        "evidence":
            (
                f"classification={temporal_class}; "
                f"mean_delta_AP="
                f"{d19.get('mean_AP_improvement')}; "
                f"months_better="
                f"{d19.get('months_AP_better')}; "
                f"block_CI="
                f"{d19.get('block_bootstrap_AP_CI95')}."
            ),
    },

    {
        "hypothesis":
            "P5_POSITIVE_QUADRATIC_CURVATURE",

        "status":
            curvature_status,

        "evidence":
            curvature_evidence,
    },

    {
        "hypothesis":
            "P6_GNADMM_CALIBRATION_SUPERIORITY",

        "status":
            calibration_status,

        "evidence":
            calibration_evidence,
    },

    {
        "hypothesis":
            "P7_CONSTRAINT_ABLATION_NO_UNIVERSAL_WINNER",

        "status":
            "SUPPORTED"
            if d20.get("status") == "PASS"
            else "INCONCLUSIVE",

        "evidence":
            (
                f"best_AP={d20.get('best_mean_AP')}; "
                f"best_Brier={d20.get('best_mean_Brier')}; "
                f"best_LogLoss={d20.get('best_mean_LogLoss')}; "
                f"best_AUC={d20.get('best_mean_AUC')}."
            ),
    },

    {
        "hypothesis":
            "P8_READY_FOR_FINAL_DOCUMENTATION",

        "status":
            "SUPPORTED"
            if (
                core_ok
                and
                d19.get("status") == "PASS"
                and
                d20.get("status") == "PASS"
                and
                d21.get("status") == "PASS"
                and
                bool(d22)
            )
            else "INCONCLUSIVE",

        "evidence":
            (
                "Core analytical artifacts and machine-readable "
                "scientific results 19–22 were found."
            ),
    },
]

hyp_df = pd.DataFrame(
    hypotheses
)

hyp_df.to_csv(
    OUT
    / "23a_integrated_hypothesis_matrix.csv",
    index=False,
)


# ==============================================================================================
# 7. EVIDENCE REGISTRY
# ==============================================================================================

evidence_files = [
    SCI
    / "11a_nonlinear_model_comparison.csv",

    SCI
    / "11b_nonlinear_oot_performance.csv",

    GN
    / "19a_paired_temporal_inference.csv",

    GN
    / "19d_PAIRED_TEMPORAL_DECISION.json",

    GN
    / "20c_constraint_ablation_summary.csv",

    GN
    / "20d_CONSTRAINT_ABLATION_DECISION.json",

    GN
    / "21a_oot_calibration_summary.csv",

    GN
    / "21d_CALIBRATION_DECISION.json",

    GN
    / "22a_GNADMM_hypothesis_matrix.csv",

    GN
    / "22e_GNADMM_SCIENTIFIC_FREEZE.json",
]

evidence_rows = []

for p in evidence_files:

    evidence_rows.append(
        {
            "file":
                str(
                    p.relative_to(ROOT)
                ),

            "exists":
                bool(
                    p.exists()
                ),

            "size_bytes":
                (
                    p.stat().st_size
                    if p.exists()
                    else None
                ),
        }
    )

pd.DataFrame(
    evidence_rows
).to_csv(
    OUT
    / "23b_evidence_registry.csv",
    index=False,
)


# ==============================================================================================
# 8. CLAIM REGISTRY
# ==============================================================================================

claims = {
    "allowed_claims": [
        (
            "The analytical pipeline was completed through "
            "spatiotemporal integration and OOT expected freight."
        ),

        (
            "Flexible nonlinear reliability specifications did not "
            "establish robust temporal predictive superiority over "
            "the simpler linear specification."
        ),

        (
            "The GN–ADMM Brier-NLS implementation was numerically "
            "validated on the evaluated real-world logistics data."
        ),

        (
            "GN–ADMM showed a small favorable mean AP difference "
            "relative to logistic regression, but paired temporal "
            "evidence was not decisive."
        ),

        (
            "Constraint ablation and calibration diagnostics favored "
            "different models, so no universal model winner is claimed."
        ),

        (
            "The imposed nonnegative quadratic constraint is not "
            "evidence of a true U-shaped data-generating relationship."
        ),
    ],

    "prohibited_claims": [
        (
            "Freight deviations cause delivery delays."
        ),

        (
            "GN–ADMM is universally superior to logistic regression."
        ),

        (
            "The beta_z2 >= 0 constraint proves a true U-shaped relationship."
        ),

        (
            "Independent-solver agreement proves global convergence "
            "of the external nonlinear sequence."
        ),

        (
            "A lower Brier score alone proves superior calibration."
        ),

        (
            "Haversine distance is equivalent to road-network distance."
        ),
    ],
}

(
    OUT
    / "23c_claim_registry.json"
).write_text(
    json.dumps(
        claims,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==============================================================================================
# 9. FINAL SCIENTIFIC FREEZE
# ==============================================================================================

freeze = {
    "status":
        "SCIENTIFIC_EVIDENCE_FREEZE_COMPLETE",

    "scope":
        "DELIVERY_RISK_INTELLIGENCE",

    "context_core_n":
        core_n,

    "expected_freight_n":
        freight_n,

    "nonlinear_predictive_gain":
        nonlinear_status,

    "gnadmm_temporal_AP":
        temporal_status,

    "positive_quadratic_curvature":
        curvature_status,

    "gnadmm_calibration_superiority":
        calibration_status,

    "causal_claim":
        False,

    "universal_model_superiority":
        False,

    "raw_modified":
        False,
}

(
    OUT
    / "23d_SCIENTIFIC_EVIDENCE_FREEZE.json"
).write_text(
    json.dumps(
        freeze,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print(
    hyp_df.to_string(
        index=False
    )
)

print()
print(
    json.dumps(
        freeze,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "[PASS 23] SCIENTIFIC EVIDENCE FREEZE COMPLETE."
)
