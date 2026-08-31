#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.special import expit

from scipy.optimize import minimize

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

import statsmodels.api as sm


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

data = np.load(
    ART
    / "15_delivery_gnadmm_design.npz"
)

X = data["X"].astype(
    np.float64
)

y = data["y"].astype(
    np.float64
)

months = data["months"].astype(
    str
)

m = X.shape[1]


# ==================================================================================================
# GENERIC OBJECTIVE
# ==================================================================================================

def brier_objective(
    beta,
    X_local,
    y_local,
):

    p = expit(
        X_local @ beta
    )

    r = y_local - p

    return (
        0.5
        *
        float(
            np.mean(
                r * r
            )
        )
    )


def brier_gradient(
    beta,
    X_local,
    y_local,
):

    p = expit(
        X_local @ beta
    )

    r = y_local - p

    J = (
        p
        *
        (1.0 - p)
    )[:, None] * X_local

    return -(
        J.T @ r
    ) / len(y_local)


# ==================================================================================================
# LOCAL GN
# ==================================================================================================

def fit_gn(
    Xtr,
    ytr,
    beta_init=None,
    lambda_reg=1e-4,
    max_iter=80,
    tol=1e-7,
):

    if beta_init is None:
        beta = np.zeros(
            Xtr.shape[1]
        )
    else:
        beta = beta_init.copy()

    for _ in range(
        max_iter
    ):

        p = expit(
            Xtr @ beta
        )

        r = ytr - p

        J = (
            p
            *
            (1.0 - p)
        )[:, None] * Xtr

        n = len(ytr)

        H = (
            J.T @ J
        ) / n

        H += (
            lambda_reg
            *
            np.eye(
                Xtr.shape[1]
            )
        )

        rhs = (
            J.T @ r
        ) / n

        delta = np.linalg.solve(
            H,
            rhs
        )

        old = brier_objective(
            beta,
            Xtr,
            ytr
        )

        alpha = 1.0

        for _ in range(25):

            cand = (
                beta
                +
                alpha * delta
            )

            new = brier_objective(
                cand,
                Xtr,
                ytr
            )

            if new < old:
                break

            alpha *= 0.5

        beta = cand

        if (
            np.linalg.norm(
                alpha * delta
            )
            <
            tol
        ):
            break

    return beta


# ==================================================================================================
# LOCAL GN–ADMM
# ==================================================================================================

def admm_delta(
    Xtr,
    ytr,
    beta,
    lambda_reg=1e-4,
    rho=1.0,
    max_admm=2000,
    tol=1e-6,
):

    p = expit(
        Xtr @ beta
    )

    r = ytr - p

    J = (
        p
        *
        (1.0 - p)
    )[:, None] * Xtr

    n = len(ytr)

    H = (
        J.T @ J
    ) / n

    H += (
        lambda_reg
        *
        np.eye(
            Xtr.shape[1]
        )
    )

    q = (
        J.T @ r
    ) / n

    # beta_z2 >= 0
    #
    # -beta_z2 <= 0

    A = np.zeros(
        (1, Xtr.shape[1])
    )

    A[0, 2] = -1.0

    h = np.array(
        [beta[2]]
    )

    K = (
        H
        +
        rho
        *
        (A.T @ A)
    )

    delta = np.zeros(
        Xtr.shape[1]
    )

    z = np.minimum(
        A @ delta,
        h
    )

    u = np.zeros_like(
        z
    )

    for _ in range(
        max_admm
    ):

        rhs = (
            q
            +
            rho
            *
            A.T
            @
            (z - u)
        )

        delta = np.linalg.solve(
            K,
            rhs
        )

        Ad = A @ delta

        z_old = z.copy()

        z = np.minimum(
            Ad + u,
            h
        )

        rpri = (
            Ad - z
        )

        u += rpri

        rdual = (
            rho
            *
            A.T
            @
            (z - z_old)
        )

        if (
            np.linalg.norm(
                rpri
            )
            < tol
            and
            np.linalg.norm(
                rdual
            )
            < tol
        ):
            break

    return delta


def fit_gnadmm(
    Xtr,
    ytr,
    beta_init=None,
    max_gn=80,
    tol=1e-7,
):

    if beta_init is None:
        beta = np.zeros(
            Xtr.shape[1]
        )
    else:
        beta = beta_init.copy()

    beta[2] = max(
        beta[2],
        0.0
    )

    for _ in range(
        max_gn
    ):

        delta = admm_delta(
            Xtr,
            ytr,
            beta
        )

        old = brier_objective(
            beta,
            Xtr,
            ytr
        )

        alpha = 1.0

        for _ in range(25):

            cand = (
                beta
                +
                alpha * delta
            )

            if cand[2] < -1e-10:
                alpha *= 0.5
                continue

            new = brier_objective(
                cand,
                Xtr,
                ytr
            )

            if new < old:
                break

            alpha *= 0.5

        cand[2] = max(
            cand[2],
            0.0
        )

        beta = cand

        if (
            np.linalg.norm(
                alpha * delta
            )
            <
            tol
        ):
            break

    return beta


# ==================================================================================================
# INDEPENDENT FULL-SAMPLE REFERENCE — SCIPY SLSQP
# ==================================================================================================

print("=" * 110)
print("MODULE 18 — GN / GN–ADMM / LOGIT OOT BENCHMARK")
print("=" * 110)

beta_gnadmm = np.load(
    ART
    / "17_gnadmm_constrained_beta.npy"
)

constraint = {
    "type":
        "ineq",

    "fun":
        lambda beta:
            beta[2],
}

ref = minimize(
    fun=lambda beta:
        brier_objective(
            beta,
            X,
            y
        ),

    x0=beta_gnadmm,

    jac=lambda beta:
        brier_gradient(
            beta,
            X,
            y
        ),

    constraints=[
        constraint
    ],

    method="SLSQP",

    options={
        "maxiter": 1000,
        "ftol": 1e-12,
        "disp": False,
    },
)

reference = {
    "success":
        bool(
            ref.success
        ),

    "message":
        str(
            ref.message
        ),

    "objective":
        float(
            ref.fun
        ),

    "brier":
        float(
            2.0 * ref.fun
        ),

    "beta_z2":
        float(
            ref.x[2]
        ),

    "distance_to_gnadmm":
        float(
            np.linalg.norm(
                ref.x
                -
                beta_gnadmm
            )
        ),
}

with open(
    REP
    / "18a_independent_reference.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        reference,
        f,
        indent=2
    )


# ==================================================================================================
# EXPANDING-WINDOW OOT
# ==================================================================================================

months_sorted = sorted(
    np.unique(
        months
    )
)

MIN_TRAIN_MONTHS = 3

rows = []

warm_gn = None
warm_gnadmm = None

for i in range(
    MIN_TRAIN_MONTHS,
    len(months_sorted)
):

    train_months = (
        months_sorted[:i]
    )

    test_month = (
        months_sorted[i]
    )

    tr = np.isin(
        months,
        train_months
    )

    te = (
        months == test_month
    )

    Xtr = X[tr]
    ytr = y[tr]

    Xte = X[te]
    yte = y[te]

    if (
        len(yte) == 0
        or
        np.unique(ytr).size < 2
        or
        np.unique(yte).size < 2
    ):
        continue


    # ----------------------------------------------------------------------------------------------
    # MODEL 1 — STANDARD LOGISTIC MLE
    # ----------------------------------------------------------------------------------------------

    try:

        logit = sm.Logit(
            ytr,
            Xtr
        )

        logit_fit = logit.fit(
            disp=False,
            maxiter=200
        )

        pred_logit = np.clip(
            logit_fit.predict(
                Xte
            ),
            1e-8,
            1.0 - 1e-8
        )

    except Exception:

        # Regularized logistic fallback.
        logit_fit = sm.Logit(
            ytr,
            Xtr
        ).fit_regularized(
            alpha=1e-6,
            disp=False
        )

        pred_logit = np.clip(
            logit_fit.predict(
                Xte
            ),
            1e-8,
            1.0 - 1e-8
        )


    # ----------------------------------------------------------------------------------------------
    # MODEL 2 — GN / BRIER
    # ----------------------------------------------------------------------------------------------

    beta_gn = fit_gn(
        Xtr,
        ytr,
        beta_init=warm_gn
    )

    warm_gn = beta_gn.copy()

    pred_gn = np.clip(
        expit(
            Xte @ beta_gn
        ),
        1e-8,
        1.0 - 1e-8
    )


    # ----------------------------------------------------------------------------------------------
    # MODEL 3 — GN–ADMM / BRIER / beta_z² >= 0
    # ----------------------------------------------------------------------------------------------

    beta_ga = fit_gnadmm(
        Xtr,
        ytr,
        beta_init=warm_gnadmm
    )

    warm_gnadmm = beta_ga.copy()

    pred_ga = np.clip(
        expit(
            Xte @ beta_ga
        ),
        1e-8,
        1.0 - 1e-8
    )


    preds = {
        "LOGIT_MLE":
            pred_logit,

        "GN_BRIER":
            pred_gn,

        "GNADMM_BRIER_CONSTRAINED":
            pred_ga,
    }


    for model_name, pred in preds.items():

        rows.append(
            {
                "test_month":
                    test_month,

                "model":
                    model_name,

                "n_train":
                    int(
                        tr.sum()
                    ),

                "n_test":
                    int(
                        te.sum()
                    ),

                "prevalence":
                    float(
                        np.mean(
                            yte
                        )
                    ),

                "average_precision":
                    float(
                        average_precision_score(
                            yte,
                            pred
                        )
                    ),

                "brier":
                    float(
                        brier_score_loss(
                            yte,
                            pred
                        )
                    ),

                "log_loss":
                    float(
                        log_loss(
                            yte,
                            pred
                        )
                    ),

                "roc_auc":
                    float(
                        roc_auc_score(
                            yte,
                            pred
                        )
                    ),
            }
        )

    print(
        f"[OOT] {test_month} "
        f"| train={tr.sum():,} "
        f"| test={te.sum():,}"
    )


oot = pd.DataFrame(
    rows
)

oot.to_csv(
    REP
    / "18b_gnadmm_oot_fold_metrics.csv",
    index=False
)


summary = (
    oot
    .groupby("model")
    .agg(
        months=(
            "test_month",
            "nunique"
        ),

        mean_ap=(
            "average_precision",
            "mean"
        ),

        median_ap=(
            "average_precision",
            "median"
        ),

        mean_brier=(
            "brier",
            "mean"
        ),

        mean_log_loss=(
            "log_loss",
            "mean"
        ),

        mean_auc=(
            "roc_auc",
            "mean"
        ),
    )
    .reset_index()
)


summary.to_csv(
    REP
    / "18c_gnadmm_oot_model_summary.csv",
    index=False
)


# ==================================================================================================
# PAIRED MONTHLY COMPARISON
# ==================================================================================================

pivot_ap = oot.pivot(
    index="test_month",
    columns="model",
    values="average_precision"
)

pivot_brier = oot.pivot(
    index="test_month",
    columns="model",
    values="brier"
)

paired = []

for candidate in [
    "GN_BRIER",
    "GNADMM_BRIER_CONSTRAINED",
]:

    common = (
        pivot_ap[
            [
                "LOGIT_MLE",
                candidate
            ]
        ]
        .dropna()
    )

    common_b = (
        pivot_brier[
            [
                "LOGIT_MLE",
                candidate
            ]
        ]
        .dropna()
    )

    paired.append(
        {
            "candidate":
                candidate,

            "months":
                int(
                    len(common)
                ),

            "mean_delta_AP_vs_logit":
                float(
                    (
                        common[
                            candidate
                        ]
                        -
                        common[
                            "LOGIT_MLE"
                        ]
                    ).mean()
                ),

            "median_delta_AP_vs_logit":
                float(
                    (
                        common[
                            candidate
                        ]
                        -
                        common[
                            "LOGIT_MLE"
                        ]
                    ).median()
                ),

            "months_AP_better":
                int(
                    (
                        common[
                            candidate
                        ]
                        >
                        common[
                            "LOGIT_MLE"
                        ]
                    ).sum()
                ),

            "mean_delta_Brier_vs_logit":
                float(
                    (
                        common_b[
                            candidate
                        ]
                        -
                        common_b[
                            "LOGIT_MLE"
                        ]
                    ).mean()
                ),

            "months_Brier_better":
                int(
                    (
                        common_b[
                            candidate
                        ]
                        <
                        common_b[
                            "LOGIT_MLE"
                        ]
                    ).sum()
                ),
        }
    )


paired_df = pd.DataFrame(
    paired
)

paired_df.to_csv(
    REP
    / "18d_gnadmm_paired_oot_comparison.csv",
    index=False
)


# ==================================================================================================
# FINAL DECISION
# ==================================================================================================

best_ap_model = (
    summary
    .sort_values(
        "mean_ap",
        ascending=False
    )
    .iloc[0][
        "model"
    ]
)

best_brier_model = (
    summary
    .sort_values(
        "mean_brier",
        ascending=True
    )
    .iloc[0][
        "model"
    ]
)

decision = {
    "status":
        "PASS",

    "scope":
        "GN_ADMM_DELIVERY_EXPERIMENT_V1",

    "objective_gn":
        "HALF_MEAN_BRIER_NLLS",

    "constraint":
        "beta_z_freight_sq >= 0",

    "independent_reference":
        reference,

    "best_mean_AP_model":
        str(
            best_ap_model
        ),

    "best_mean_Brier_model":
        str(
            best_brier_model
        ),

    "important_guardrail":
        (
            "GN–ADMM is evaluated as a "
            "constrained Brier-NLLS benchmark. "
            "It does not replace logistic "
            "likelihood merely because it is "
            "nonlinear or constrained."
        ),

    "causal_claim":
        False,

    "global_optimum_certified":
        False,

    "raw_modified":
        False,
}

with open(
    REP
    / "18e_GNADMM_DELIVERY_SCIENTIFIC_DECISION.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        decision,
        f,
        indent=2,
        ensure_ascii=False
    )


report = f"""
====================================================================================================
GN–ADMM DELIVERY EXPERIMENT — FINAL REPORT
====================================================================================================

OBJECTIVE
----------------------------------------------------------------------------------------------------
p_i(beta) = sigmoid(x_i^T beta)

r_i(beta) = y_i - p_i(beta)

J(beta) = 0.5 * mean(r_i(beta)^2)

This is a nonlinear least-squares formulation whose unscaled
mean squared residual is the Brier score.

CONSTRAINT
----------------------------------------------------------------------------------------------------
beta_z_freight_sq >= 0

MODELS
----------------------------------------------------------------------------------------------------
1. LOGIT_MLE
2. GN_BRIER
3. GNADMM_BRIER_CONSTRAINED

INDEPENDENT REFERENCE
----------------------------------------------------------------------------------------------------
success                  : {reference['success']}
objective                : {reference['objective']:.12f}
Brier                    : {reference['brier']:.12f}
beta_z2                  : {reference['beta_z2']:.12f}
distance to GN–ADMM      : {reference['distance_to_gnadmm']:.12e}

OOT SUMMARY
----------------------------------------------------------------------------------------------------
{summary.to_string(index=False)}

PAIRED OOT COMPARISON
----------------------------------------------------------------------------------------------------
{paired_df.to_string(index=False)}

BEST MEAN AP
----------------------------------------------------------------------------------------------------
{best_ap_model}

BEST MEAN BRIER
----------------------------------------------------------------------------------------------------
{best_brier_model}

INTERPRETATION GUARDRAILS
----------------------------------------------------------------------------------------------------
- This experiment evaluates GN–ADMM as a constrained Brier-NLLS estimator.
- It does not prove global convergence of the external GN sequence.
- It does not establish causal effects of freight on delivery delay.
- The inequality beta_z² >= 0 is an imposed scientific hypothesis, not information created by data.
- Predictive conclusions must be based on OOT results, not only in-sample objective values.
- Great-circle distance remains a geographic proxy, not road-network distance.

====================================================================================================
END
====================================================================================================
"""


(
    REP
    / "18f_GNADMM_DELIVERY_REPORT.txt"
).write_text(
    report,
    encoding="utf-8"
)

print()
print("OOT MODEL SUMMARY")
print(
    summary.to_string(
        index=False
    )
)

print()
print("PAIRED OOT")
print(
    paired_df.to_string(
        index=False
    )
)

print()
print(
    "[PASS 18] GN–ADMM "
    "DELIVERY BENCHMARK COMPLETE."
)
