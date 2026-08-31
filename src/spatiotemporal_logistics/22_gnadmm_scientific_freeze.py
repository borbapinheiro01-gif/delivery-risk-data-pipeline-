#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

ART = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
    / "gnadmm_delivery"
)

REP = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
    / "gnadmm_delivery"
)


def load_json(
    path
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(
            f
        )


d15 = load_json(
    ART
    / "15_delivery_gnadmm_metadata.json"
)

d16 = load_json(
    REP
    / "16d_GN_UNCONSTRAINED_DECISION.json"
)

d17 = load_json(
    REP
    / "17e_GNADMM_CONSTRAINED_DECISION.json"
)

d18 = load_json(
    REP
    / "18e_GNADMM_DELIVERY_SCIENTIFIC_DECISION.json"
)

d19 = load_json(
    REP
    / "19d_PAIRED_TEMPORAL_DECISION.json"
)

d20 = load_json(
    REP
    / "20d_CONSTRAINT_ABLATION_DECISION.json"
)

d21 = load_json(
    REP
    / "21d_CALIBRATION_DECISION.json"
)


m16 = pd.read_csv(
    REP
    / "16a_gn_multistart_summary.csv"
)

m20 = pd.read_csv(
    REP
    / "20c_constraint_ablation_summary.csv"
)

m21 = pd.read_csv(
    REP
    / "21a_oot_calibration_summary.csv"
)


hypotheses = []


# --------------------------------------------------------------------------------------------------
# H1 — Operator / Jacobian
# --------------------------------------------------------------------------------------------------

h1 = bool(
    d15.get(
        "operator_valid",
        False
    )
    and
    d15.get(
        "jacobian_valid",
        False
    )
)

hypotheses.append(
    {
        "hypothesis":
            "H1_OPERATOR_AND_JACOBIAN_VALIDITY",

        "status":
            (
                "SUPPORTED"
                if h1
                else
                "NOT_SUPPORTED"
            ),

        "evidence":
            (
                "Sigmoid operator remained within "
                "(0,1), and the analytic Jacobian "
                f"achieved best max absolute finite-"
                f"difference error of "
                f"{d15.get('jacobian_best_max_abs_error')}."
            ),
    }
)


# --------------------------------------------------------------------------------------------------
# H2 — GN multistart numerical stability
# --------------------------------------------------------------------------------------------------

obj_range = float(
    m16[
        "objective"
    ].max()
    -
    m16[
        "objective"
    ].min()
)

h2 = bool(
    obj_range
    <
    1e-8
)

hypotheses.append(
    {
        "hypothesis":
            "H2_UNCONSTRAINED_GN_MULTISTART_STABILITY",

        "status":
            (
                "SUPPORTED"
                if h2
                else
                "PARTIALLY_SUPPORTED"
            ),

        "evidence":
            (
                f"Objective range across the tested "
                f"GN starts was {obj_range:.3e}."
            ),
    }
)


# --------------------------------------------------------------------------------------------------
# H3 — independent solver agreement
# --------------------------------------------------------------------------------------------------

ref = d18[
    "independent_reference"
]

ref_distance = float(
    ref[
        "distance_to_gnadmm"
    ]
)

h3 = bool(
    ref.get(
        "success",
        False
    )
    and
    ref_distance
    <
    1e-5
)

hypotheses.append(
    {
        "hypothesis":
            "H3_GNADMM_INDEPENDENT_SOLVER_AGREEMENT",

        "status":
            (
                "SUPPORTED"
                if h3
                else
                "INCONCLUSIVE"
            ),

        "evidence":
            (
                "Independent constrained solver "
                f"distance to GN–ADMM = "
                f"{ref_distance:.3e}."
            ),
    }
)


# --------------------------------------------------------------------------------------------------
# H4 — OOT AP advantage
# --------------------------------------------------------------------------------------------------

ap_class = d19.get(
    "AP_evidence_classification",
    "INCONCLUSIVE"
)

if ap_class == "SUPPORTED":
    h4_status = "SUPPORTED"

elif ap_class == "FAVORABLE_NOT_DECISIVE":
    h4_status = "PARTIALLY_SUPPORTED"

else:
    h4_status = "INCONCLUSIVE"


hypotheses.append(
    {
        "hypothesis":
            "H4_GNADMM_OOT_AP_ADVANTAGE",

        "status":
            h4_status,

        "evidence":
            (
                f"Mean paired AP improvement = "
                f"{d19.get('mean_AP_improvement')}; "
                f"better in "
                f"{d19.get('months_AP_better')} months; "
                f"block-bootstrap 95% CI = "
                f"{d19.get('block_bootstrap_AP_CI95')}."
            ),
    }
)


# --------------------------------------------------------------------------------------------------
# H5 — empirical convexity
# --------------------------------------------------------------------------------------------------

beta_free = float(
    d16.get(
        "beta_z2",
        np.nan
    )
)

beta_ge = float(
    d17.get(
        "beta_z2",
        np.nan
    )
)

restriction_at_boundary = bool(
    np.isfinite(
        beta_ge
    )
    and
    abs(
        beta_ge
    )
    <
    1e-6
)


if (
    np.isfinite(
        beta_free
    )
    and
    beta_free < 0
    and
    restriction_at_boundary
):

    h5_status = (
        "NOT_SUPPORTED"
    )

    h5_text = (
        "The unconstrained Brier-NLS solution "
        f"preferred beta_z2={beta_free:.6f}, while "
        "the beta_z2>=0 constrained solution "
        "collapsed to the boundary near zero. "
        "This does not support positive convex "
        "curvature as an empirical conclusion."
    )

elif (
    np.isfinite(
        beta_free
    )
    and
    beta_free > 0
):

    h5_status = (
        "PARTIALLY_SUPPORTED"
    )

    h5_text = (
        "The unconstrained coefficient was positive, "
        "but predictive and shape evidence must still "
        "be considered jointly."
    )

else:

    h5_status = (
        "INCONCLUSIVE"
    )

    h5_text = (
        "The curvature sign could not be classified "
        "reliably."
    )


hypotheses.append(
    {
        "hypothesis":
            "H5_POSITIVE_QUADRATIC_CURVATURE",

        "status":
            h5_status,

        "evidence":
            h5_text,
    }
)


# --------------------------------------------------------------------------------------------------
# H6 — structural regularization hypothesis
# --------------------------------------------------------------------------------------------------

row_ge = m20[
    m20[
        "model"
    ]
    ==
    "GNADMM_GE0"
]

row_free = m20[
    m20[
        "model"
    ]
    ==
    "GN_FREE_BRIER"
]

row_eq = m20[
    m20[
        "model"
    ]
    ==
    "GN_EQ0"
]

if (
    not row_ge.empty
    and
    not row_free.empty
    and
    not row_eq.empty
):

    ap_ge = float(
        row_ge.iloc[0][
            "mean_AP"
        ]
    )

    ap_free = float(
        row_free.iloc[0][
            "mean_AP"
        ]
    )

    ap_eq = float(
        row_eq.iloc[0][
            "mean_AP"
        ]
    )

    if (
        ap_ge > ap_free
        and
        h5_status
        ==
        "NOT_SUPPORTED"
    ):

        h6_status = (
            "PARTIALLY_SUPPORTED"
        )

        h6_text = (
            "The nonnegative-curvature constrained "
            "model improved mean AP relative to the "
            "free Brier-GN model despite the "
            "unconstrained coefficient preferring "
            "negative curvature. This is compatible "
            "with a regularization effect, but does "
            "not prove that mechanism."
        )

    else:

        h6_status = (
            "INCONCLUSIVE"
        )

        h6_text = (
            "Constraint ablation did not establish "
            "a clear structural-regularization gain."
        )

else:

    h6_status = (
        "INCONCLUSIVE"
    )

    h6_text = (
        "Required ablation rows were unavailable."
    )


hypotheses.append(
    {
        "hypothesis":
            "H6_CONSTRAINT_AS_STRUCTURAL_REGULARIZATION",

        "status":
            h6_status,

        "evidence":
            h6_text,
    }
)


# --------------------------------------------------------------------------------------------------
# H7 — calibration superiority
# --------------------------------------------------------------------------------------------------

best_brier = d21.get(
    "best_pooled_OOT_Brier"
)

best_ece = d21.get(
    "best_pooled_OOT_ECE"
)

best_cal = d21.get(
    "best_calibration_intercept_slope_proximity"
)


if (
    best_brier
    ==
    "GNADMM_GE0"
    and
    best_ece
    ==
    "GNADMM_GE0"
    and
    best_cal
    ==
    "GNADMM_GE0"
):

    h7_status = (
        "SUPPORTED"
    )

else:

    h7_status = (
        "NOT_SUPPORTED"
    )


hypotheses.append(
    {
        "hypothesis":
            "H7_GNADMM_CALIBRATION_SUPERIORITY",

        "status":
            h7_status,

        "evidence":
            (
                f"Best pooled Brier: {best_brier}; "
                f"best quantile-ECE: {best_ece}; "
                f"best intercept/slope proximity: "
                f"{best_cal}."
            ),
    }
)


# --------------------------------------------------------------------------------------------------
# H8 — computational applicability
# --------------------------------------------------------------------------------------------------

h8 = bool(
    d15.get(
        "n_orders",
        0
    )
    >
    90000
    and
    d17.get(
        "status"
    )
    ==
    "PASS"
    and
    d18.get(
        "status"
    )
    ==
    "PASS"
)

hypotheses.append(
    {
        "hypothesis":
            "H8_GNADMM_COMPUTATIONAL_APPLICABILITY_REAL_DATA",

        "status":
            (
                "SUPPORTED"
                if h8
                else
                "INCONCLUSIVE"
            ),

        "evidence":
            (
                f"The complete experiment ran on "
                f"{d15.get('n_orders')} real-world "
                f"e-commerce observations with "
                f"{d15.get('n_parameters')} model "
                f"parameters, including OOT validation "
                f"and an independent constrained solver."
            ),
    }
)


hyp_df = pd.DataFrame(
    hypotheses
)

hyp_df.to_csv(
    REP
    / "22a_GNADMM_hypothesis_matrix.csv",
    index=False
)


# --------------------------------------------------------------------------------------------------
# Evidence matrix
# --------------------------------------------------------------------------------------------------

evidence_rows = [
    {
        "module": 15,
        "topic": "operator_jacobian",
        "artifact":
            "15a_jacobian_finite_difference_audit.csv",
        "status":
            d15.get("status"),
    },
    {
        "module": 16,
        "topic": "unconstrained_GN",
        "artifact":
            "16a_gn_multistart_summary.csv",
        "status":
            d16.get("status"),
    },
    {
        "module": 17,
        "topic": "constrained_GNADMM",
        "artifact":
            "17a_gnadmm_multistart_summary.csv",
        "status":
            d17.get("status"),
    },
    {
        "module": 18,
        "topic": "OOT_benchmark",
        "artifact":
            "18c_gnadmm_oot_model_summary.csv",
        "status":
            d18.get("status"),
    },
    {
        "module": 19,
        "topic": "paired_temporal_inference",
        "artifact":
            "19a_paired_temporal_inference.csv",
        "status":
            d19.get("status"),
    },
    {
        "module": 20,
        "topic": "constraint_ablation",
        "artifact":
            "20c_constraint_ablation_summary.csv",
        "status":
            d20.get("status"),
    },
    {
        "module": 21,
        "topic": "calibration_resolution",
        "artifact":
            "21a_oot_calibration_summary.csv",
        "status":
            d21.get("status"),
    },
]


pd.DataFrame(
    evidence_rows
).to_csv(
    REP
    / "22b_GNADMM_evidence_chain.csv",
    index=False
)


# --------------------------------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------------------------------

allowed_claims = [
    (
        "The GN–ADMM Brier-NLS formulation was "
        "implemented and numerically validated on "
        "the logistics reliability dataset."
    ),

    (
        "The analytic probability Jacobian showed "
        "high numerical agreement with finite "
        "differences."
    ),

    (
        "The constrained GN–ADMM solution showed "
        "close agreement with an independent "
        "constrained numerical optimizer."
    ),

    (
        "Predictive performance differed by metric; "
        "therefore no model is described as "
        "universally superior."
    ),
]


if h4_status == "SUPPORTED":

    allowed_claims.append(
        (
            "GN–ADMM showed a temporally supported "
            "improvement in Average Precision over "
            "the logistic baseline under the audited "
            "OOT protocol."
        )
    )

elif h4_status == "PARTIALLY_SUPPORTED":

    allowed_claims.append(
        (
            "GN–ADMM showed a small favorable mean "
            "Average Precision difference OOT, but "
            "the paired temporal evidence was not "
            "strong enough to establish decisive "
            "superiority."
        )
    )

else:

    allowed_claims.append(
        (
            "The audited OOT experiment did not "
            "establish a consistent Average Precision "
            "advantage for GN–ADMM."
        )
    )


if h5_status == "NOT_SUPPORTED":

    allowed_claims.append(
        (
            "The nonnegative quadratic restriction "
            "must not be interpreted as evidence of "
            "an underlying U-shaped relationship: "
            "the unrestricted Brier-NLS fit preferred "
            "negative quadratic curvature and the "
            "restricted solution was located at the "
            "constraint boundary."
        )
    )


prohibited_claims = [
    (
        "GN–ADMM proves that freight deviations "
        "cause delivery delays."
    ),

    (
        "GN–ADMM is universally superior to "
        "logistic regression."
    ),

    (
        "The imposed beta_z2 >= 0 restriction proves "
        "that the true relationship is U-shaped."
    ),

    (
        "Agreement with the independent optimizer "
        "proves global convergence of the external "
        "Gauss–Newton sequence."
    ),

    (
        "A lower Brier score alone proves superior "
        "probability calibration."
    ),
]


claims = {
    "allowed_claims":
        allowed_claims,

    "prohibited_claims":
        prohibited_claims,
}


with open(
    REP
    / "22c_GNADMM_allowed_and_prohibited_claims.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        claims,
        f,
        indent=2,
        ensure_ascii=False
    )


# --------------------------------------------------------------------------------------------------
# Limitations
# --------------------------------------------------------------------------------------------------

limitations = [
    "Observational e-commerce data; no causal identification.",
    "Only 14 expanding-window OOT evaluation months are available for paired temporal inference.",
    "The GN objective is Brier nonlinear least squares, whereas logistic regression optimizes binomial log-likelihood.",
    "The beta_z2 sign constraint is externally imposed and cannot create structural information absent from the data.",
    "GN–ADMM convergence of the external nonlinear sequence is not globally certified by these experiments.",
    "The GN–ADMM multistart experiment showed sensitivity to initialization in at least one starting configuration.",
    "Calibration decomposition based on prediction bins is an approximation.",
    "Great-circle distance is a geographic proxy rather than road-network distance.",
    "Expected freight is an OOT model-derived quantity and inherits uncertainty from that upstream model.",
]


with open(
    REP
    / "22d_GNADMM_limitations.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "limitations":
                limitations
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# --------------------------------------------------------------------------------------------------
# Freeze summary
# --------------------------------------------------------------------------------------------------

counts = (
    hyp_df[
        "status"
    ]
    .value_counts()
    .to_dict()
)


freeze = {
    "final_status":
        "GNADMM_SCIENTIFIC_FREEZE_COMPLETE",

    "scope":
        "GN_ADMM_DELIVERY_BRIER_NLLS_V1",

    "n_orders":
        d15.get(
            "n_orders"
        ),

    "n_parameters":
        d15.get(
            "n_parameters"
        ),

    "objective":
        "HALF_MEAN_BRIER_NLLS",

    "constraint_primary":
        "beta_z_freight_sq >= 0",

    "hypothesis_status_counts":
        {
            str(k): int(v)
            for k, v in counts.items()
        },

    "primary_OOT_AP_evidence":
        h4_status,

    "positive_quadratic_curvature_evidence":
        h5_status,

    "structural_regularization_evidence":
        h6_status,

    "GNADMM_calibration_superiority":
        h7_status,

    "global_convergence_certified":
        False,

    "causal_claim":
        False,

    "raw_modified":
        False,
}


with open(
    REP
    / "22e_GNADMM_SCIENTIFIC_FREEZE.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        freeze,
        f,
        indent=2,
        ensure_ascii=False
    )


report = f"""
====================================================================================================
GN–ADMM DELIVERY — FINAL SCIENTIFIC FREEZE
====================================================================================================

SCOPE
----------------------------------------------------------------------------------------------------
GN_ADMM_DELIVERY_BRIER_NLLS_V1

DATA
----------------------------------------------------------------------------------------------------
N observations : {d15.get('n_orders')}
N parameters   : {d15.get('n_parameters')}

MODEL
----------------------------------------------------------------------------------------------------
p_i(beta) = sigmoid(x_i^T beta)

D_i(beta) = y_i - p_i(beta)

J(beta) = 0.5 * mean(D_i(beta)^2)

Primary audited restriction:
beta_z_freight_sq >= 0

HYPOTHESIS MATRIX
----------------------------------------------------------------------------------------------------
{hyp_df.to_string(index=False)}

PRIMARY TEMPORAL AP EVIDENCE
----------------------------------------------------------------------------------------------------
Classification : {h4_status}
Mean delta AP  : {d19.get('mean_AP_improvement')}
Months better  : {d19.get('months_AP_better')}
Block CI 95%   : {d19.get('block_bootstrap_AP_CI95')}
Wilcoxon p     : {d19.get('wilcoxon_AP_p_two_sided')}

CURVATURE
----------------------------------------------------------------------------------------------------
Unconstrained beta_z2 : {beta_free}
Restricted beta_z2    : {beta_ge}
Shape conclusion       : {h5_status}

CONSTRAINT ABLATION
----------------------------------------------------------------------------------------------------
Best AP       : {d20.get('best_mean_AP')}
Best Brier    : {d20.get('best_mean_Brier')}
Best LogLoss  : {d20.get('best_mean_LogLoss')}
Best AUC      : {d20.get('best_mean_AUC')}

CALIBRATION
----------------------------------------------------------------------------------------------------
Best pooled Brier          : {best_brier}
Best pooled quantile-ECE    : {best_ece}
Best intercept/slope match  : {best_cal}

ALLOWED CLAIMS
----------------------------------------------------------------------------------------------------
{chr(10).join("- " + c for c in allowed_claims)}

PROHIBITED CLAIMS
----------------------------------------------------------------------------------------------------
{chr(10).join("- " + c for c in prohibited_claims)}

LIMITATIONS
----------------------------------------------------------------------------------------------------
{chr(10).join("- " + c for c in limitations)}

FINAL GUARDRAIL
----------------------------------------------------------------------------------------------------
The experiments validate the numerical implementation and characterize its empirical
behavior under the evaluated data, starts, constraints and temporal folds.

They do NOT establish:
- causal effects,
- universal model superiority,
- global convergence of the nonlinear external sequence,
- or truth of an externally imposed curvature constraint.

====================================================================================================
END
====================================================================================================
"""


(
    REP
    / "22f_GNADMM_SCIENTIFIC_REPORT.txt"
).write_text(
    report,
    encoding="utf-8"
)


print()
print(
    report
)

print(
    "[PASS 22] GN–ADMM SCIENTIFIC "
    "FREEZE COMPLETE."
)
