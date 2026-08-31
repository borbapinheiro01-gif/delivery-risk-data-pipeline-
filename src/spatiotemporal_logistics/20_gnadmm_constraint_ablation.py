#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.special import expit

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
    / "15_delivery_gnadmm_design.npz",
    allow_pickle=False
)

X = data[
    "X"
].astype(
    np.float64
)

y = data[
    "y"
].astype(
    np.float64
)

months = data[
    "months"
].astype(
    str
)

m = X.shape[1]


# --------------------------------------------------------------------------------------------------
# OBJECTIVE
# --------------------------------------------------------------------------------------------------

def objective(
    beta,
    X_local,
    y_local,
):

    p = expit(
        X_local
        @
        beta
    )

    r = (
        y_local
        -
        p
    )

    return (
        0.5
        *
        float(
            np.mean(
                r * r
            )
        )
    )


# --------------------------------------------------------------------------------------------------
# GAUSS–NEWTON SYSTEM
# --------------------------------------------------------------------------------------------------

def gn_system(
    beta,
    X_local,
    y_local,
    lambda_reg=1e-4,
):

    p = expit(
        X_local
        @
        beta
    )

    r = (
        y_local
        -
        p
    )

    J = (
        p
        *
        (1.0 - p)
    )[:, None] * X_local

    n = len(
        y_local
    )

    H = (
        J.T
        @
        J
    ) / n

    H += (
        lambda_reg
        *
        np.eye(
            X_local.shape[1]
        )
    )

    q = (
        J.T
        @
        r
    ) / n

    return H, q


# --------------------------------------------------------------------------------------------------
# FREE GN
# --------------------------------------------------------------------------------------------------

def fit_free(
    Xtr,
    ytr,
    beta_init,
    max_gn=80,
    tol=1e-7,
):

    beta = (
        beta_init
        .copy()
    )

    for _ in range(
        max_gn
    ):

        H, q = gn_system(
            beta,
            Xtr,
            ytr
        )

        try:
            delta = np.linalg.solve(
                H,
                q
            )
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(
                H,
                q,
                rcond=None
            )[0]

        old = objective(
            beta,
            Xtr,
            ytr
        )

        accepted = False
        alpha = 1.0

        for _ in range(30):

            cand = (
                beta
                +
                alpha
                *
                delta
            )

            new = objective(
                cand,
                Xtr,
                ytr
            )

            if new < old:
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            break

        beta = cand

        if (
            np.linalg.norm(
                alpha
                *
                delta
            )
            <
            tol
        ):
            break

    return beta


# --------------------------------------------------------------------------------------------------
# ADMM INEQUALITY:
#
# A beta <= b
# --------------------------------------------------------------------------------------------------

def admm_delta_ineq(
    beta,
    Xtr,
    ytr,
    A,
    b,
    rho=1.0,
    max_admm=3000,
    eps_abs=1e-7,
    eps_rel=1e-5,
):

    H, q = gn_system(
        beta,
        Xtr,
        ytr
    )

    h = (
        b
        -
        A
        @
        beta
    )

    K = (
        H
        +
        rho
        *
        (
            A.T
            @
            A
        )
    )

    delta = np.zeros(
        len(beta)
    )

    z = np.minimum(
        A
        @
        delta,
        h
    )

    u = np.zeros_like(
        z
    )

    for k in range(
        1,
        max_admm + 1
    ):

        rhs = (
            q
            +
            rho
            *
            A.T
            @
            (
                z
                -
                u
            )
        )

        try:
            delta = np.linalg.solve(
                K,
                rhs
            )
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(
                K,
                rhs,
                rcond=None
            )[0]

        Ad = (
            A
            @
            delta
        )

        z_old = (
            z.copy()
        )

        z = np.minimum(
            Ad
            +
            u,
            h
        )

        r_pri = (
            Ad
            -
            z
        )

        u = (
            u
            +
            r_pri
        )

        r_dual = (
            rho
            *
            A.T
            @
            (
                z
                -
                z_old
            )
        )

        pri_norm = float(
            np.linalg.norm(
                r_pri
            )
        )

        dual_norm = float(
            np.linalg.norm(
                r_dual
            )
        )

        eps_pri = (
            np.sqrt(
                len(z)
            )
            *
            eps_abs
            +
            eps_rel
            *
            max(
                float(
                    np.linalg.norm(
                        Ad
                    )
                ),
                float(
                    np.linalg.norm(
                        z
                    )
                ),
            )
        )

        eps_dual = (
            np.sqrt(
                len(beta)
            )
            *
            eps_abs
            +
            eps_rel
            *
            float(
                np.linalg.norm(
                    rho
                    *
                    A.T
                    @
                    u
                )
            )
        )

        if (
            pri_norm
            <=
            eps_pri
            and
            dual_norm
            <=
            eps_dual
        ):
            break

    return (
        delta,
        k,
        pri_norm,
        dual_norm,
    )


def fit_inequality(
    Xtr,
    ytr,
    beta_init,
    mode,
    max_gn=80,
    tol=1e-7,
):

    beta = (
        beta_init
        .copy()
    )

    A = np.zeros(
        (
            1,
            len(beta)
        )
    )

    b = np.array(
        [0.0]
    )

    if mode == "GE0":
        # beta_z2 >= 0
        #
        # -beta_z2 <= 0
        A[
            0,
            2
        ] = -1.0

        beta[2] = max(
            0.0,
            beta[2]
        )

    elif mode == "LE0":
        # beta_z2 <= 0
        A[
            0,
            2
        ] = 1.0

        beta[2] = min(
            0.0,
            beta[2]
        )

    else:
        raise ValueError(
            mode
        )

    total_admm = 0

    for _ in range(
        max_gn
    ):

        (
            delta,
            k,
            pri,
            dual,
        ) = admm_delta_ineq(
            beta,
            Xtr,
            ytr,
            A,
            b,
        )

        total_admm += k

        old = objective(
            beta,
            Xtr,
            ytr
        )

        alpha = 1.0
        accepted = False

        for _ in range(
            30
        ):

            cand = (
                beta
                +
                alpha
                *
                delta
            )

            feasible = bool(
                np.all(
                    A
                    @
                    cand
                    <=
                    b
                    +
                    1e-8
                )
            )

            if not feasible:
                alpha *= 0.5
                continue

            new = objective(
                cand,
                Xtr,
                ytr
            )

            if new < old:
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            break

        beta = cand

        if mode == "GE0":
            beta[2] = max(
                0.0,
                beta[2]
            )
        else:
            beta[2] = min(
                0.0,
                beta[2]
            )

        if (
            np.linalg.norm(
                alpha
                *
                delta
            )
            <
            tol
        ):
            break

    return (
        beta,
        total_admm
    )


# --------------------------------------------------------------------------------------------------
# EQUALITY beta_z2 = 0
#
# Exact local equality-constrained GN via KKT.
# --------------------------------------------------------------------------------------------------

def fit_equal_zero(
    Xtr,
    ytr,
    beta_init,
    max_gn=80,
    tol=1e-7,
):

    beta = (
        beta_init
        .copy()
    )

    beta[2] = 0.0

    A = np.zeros(
        (
            1,
            len(beta)
        )
    )

    A[
        0,
        2
    ] = 1.0

    for _ in range(
        max_gn
    ):

        H, q = gn_system(
            beta,
            Xtr,
            ytr
        )

        h = (
            -
            A
            @
            beta
        )

        KKT = np.block(
            [
                [
                    H,
                    A.T
                ],
                [
                    A,
                    np.zeros(
                        (1, 1)
                    )
                ],
            ]
        )

        rhs = np.concatenate(
            [
                q,
                h
            ]
        )

        try:
            sol = np.linalg.solve(
                KKT,
                rhs
            )
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(
                KKT,
                rhs,
                rcond=None
            )[0]

        delta = sol[
            :len(beta)
        ]

        old = objective(
            beta,
            Xtr,
            ytr
        )

        alpha = 1.0
        accepted = False

        for _ in range(
            30
        ):

            cand = (
                beta
                +
                alpha
                *
                delta
            )

            cand[2] = 0.0

            new = objective(
                cand,
                Xtr,
                ytr
            )

            if new < old:
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            break

        beta = cand

        if (
            np.linalg.norm(
                alpha
                *
                delta
            )
            <
            tol
        ):
            break

    return beta


# --------------------------------------------------------------------------------------------------
# LOGISTIC MLE
# --------------------------------------------------------------------------------------------------

def fit_logit(
    Xtr,
    ytr,
):

    model = sm.Logit(
        ytr,
        Xtr
    )

    try:
        fit = model.fit(
            disp=False,
            maxiter=300
        )

    except Exception:
        fit = model.fit_regularized(
            alpha=1e-6,
            disp=False,
            maxiter=500,
        )

    return np.asarray(
        fit.params,
        dtype=float
    )


# --------------------------------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------------------------------

def metric_row(
    month,
    model,
    yte,
    pred,
    beta,
    n_train,
    n_test,
    admm_iterations=0,
):

    pred = np.clip(
        pred,
        1e-8,
        1.0 - 1e-8
    )

    return {
        "test_month":
            month,

        "model":
            model,

        "n_train":
            int(n_train),

        "n_test":
            int(n_test),

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

        "beta_z2":
            float(
                beta[2]
            ),

        "admm_iterations":
            int(
                admm_iterations
            ),
    }


print("=" * 110)
print("MODULE 20 — CONSTRAINT ABLATION")
print("=" * 110)

months_sorted = sorted(
    np.unique(
        months
    )
)

MIN_TRAIN_MONTHS = 3

metrics = []

prediction_rows = []

previous = {
    "GN_FREE_BRIER": None,
    "GNADMM_GE0": None,
    "GN_EQ0": None,
    "GNADMM_LE0": None,
}


for i in range(
    MIN_TRAIN_MONTHS,
    len(months_sorted)
):

    train_months = months_sorted[
        :i
    ]

    test_month = months_sorted[
        i
    ]

    tr = np.isin(
        months,
        train_months
    )

    te = (
        months
        ==
        test_month
    )

    Xtr = X[
        tr
    ]

    ytr = y[
        tr
    ]

    Xte = X[
        te
    ]

    yte = y[
        te
    ]

    row_ids = np.where(
        te
    )[0]

    if (
        len(yte) == 0
        or
        np.unique(ytr).size < 2
        or
        np.unique(yte).size < 2
    ):
        continue


    # ==============================================================================================
    # LOGIT
    # ==============================================================================================

    beta_logit = fit_logit(
        Xtr,
        ytr
    )

    pred_logit = expit(
        Xte
        @
        beta_logit
    )


    # ==============================================================================================
    # Stable initialization:
    # use the current-fold logistic solution,
    # optionally blended with previous GN solution.
    # ==============================================================================================

    init_free = (
        previous[
            "GN_FREE_BRIER"
        ]
        if previous[
            "GN_FREE_BRIER"
        ]
        is not None
        else beta_logit
    )

    beta_free = fit_free(
        Xtr,
        ytr,
        init_free
    )

    previous[
        "GN_FREE_BRIER"
    ] = beta_free.copy()


    init_ge = (
        previous[
            "GNADMM_GE0"
        ]
        if previous[
            "GNADMM_GE0"
        ]
        is not None
        else beta_logit
    )

    beta_ge, admm_ge = (
        fit_inequality(
            Xtr,
            ytr,
            init_ge,
            mode="GE0"
        )
    )

    previous[
        "GNADMM_GE0"
    ] = beta_ge.copy()


    init_eq = (
        previous[
            "GN_EQ0"
        ]
        if previous[
            "GN_EQ0"
        ]
        is not None
        else beta_logit
    )

    beta_eq = fit_equal_zero(
        Xtr,
        ytr,
        init_eq
    )

    previous[
        "GN_EQ0"
    ] = beta_eq.copy()


    init_le = (
        previous[
            "GNADMM_LE0"
        ]
        if previous[
            "GNADMM_LE0"
        ]
        is not None
        else beta_logit
    )

    beta_le, admm_le = (
        fit_inequality(
            Xtr,
            ytr,
            init_le,
            mode="LE0"
        )
    )

    previous[
        "GNADMM_LE0"
    ] = beta_le.copy()


    preds = {
        "LOGIT_MLE":
            (
                beta_logit,
                pred_logit,
                0
            ),

        "GN_FREE_BRIER":
            (
                beta_free,
                expit(
                    Xte
                    @
                    beta_free
                ),
                0
            ),

        "GNADMM_GE0":
            (
                beta_ge,
                expit(
                    Xte
                    @
                    beta_ge
                ),
                admm_ge
            ),

        "GN_EQ0":
            (
                beta_eq,
                expit(
                    Xte
                    @
                    beta_eq
                ),
                0
            ),

        "GNADMM_LE0":
            (
                beta_le,
                expit(
                    Xte
                    @
                    beta_le
                ),
                admm_le
            ),
    }


    pred_store = {
        "row_id":
            row_ids,

        "test_month":
            np.repeat(
                test_month,
                len(yte)
            ),

        "y_true":
            yte,
    }


    for (
        model_name,
        (
            beta_model,
            pred_model,
            admm_count
        )
    ) in preds.items():

        pred_model = np.clip(
            pred_model,
            1e-8,
            1.0 - 1e-8
        )

        metrics.append(
            metric_row(
                test_month,
                model_name,
                yte,
                pred_model,
                beta_model,
                int(
                    tr.sum()
                ),
                int(
                    te.sum()
                ),
                admm_count,
            )
        )

        pred_store[
            "pred_"
            +
            model_name
        ] = pred_model


    prediction_rows.append(
        pd.DataFrame(
            pred_store
        )
    )


    print(
        f"[ABLATION OOT] {test_month} "
        f"| train={tr.sum():,} "
        f"| test={te.sum():,} "
        f"| z2 free={beta_free[2]:+.5f} "
        f"| GE0={beta_ge[2]:+.3e} "
        f"| EQ0={beta_eq[2]:+.3e} "
        f"| LE0={beta_le[2]:+.5f}"
    )


metrics_df = pd.DataFrame(
    metrics
)

metrics_df.to_csv(
    REP
    / "20a_constraint_ablation_oot_metrics.csv",
    index=False
)


pred_df = pd.concat(
    prediction_rows,
    ignore_index=True
)

pred_df.to_csv(
    REP
    / "20b_constraint_ablation_oot_predictions.csv",
    index=False
)


summary = (
    metrics_df
    .groupby("model")
    .agg(
        months=(
            "test_month",
            "nunique"
        ),

        mean_AP=(
            "average_precision",
            "mean"
        ),

        median_AP=(
            "average_precision",
            "median"
        ),

        mean_Brier=(
            "brier",
            "mean"
        ),

        mean_LogLoss=(
            "log_loss",
            "mean"
        ),

        mean_AUC=(
            "roc_auc",
            "mean"
        ),

        mean_beta_z2=(
            "beta_z2",
            "mean"
        ),

        median_beta_z2=(
            "beta_z2",
            "median"
        ),

        total_ADMM_iterations=(
            "admm_iterations",
            "sum"
        ),
    )
    .reset_index()
)


summary.to_csv(
    REP
    / "20c_constraint_ablation_summary.csv",
    index=False
)


best_ap = str(
    summary.loc[
        summary[
            "mean_AP"
        ].idxmax(),
        "model"
    ]
)

best_brier = str(
    summary.loc[
        summary[
            "mean_Brier"
        ].idxmin(),
        "model"
    ]
)

best_logloss = str(
    summary.loc[
        summary[
            "mean_LogLoss"
        ].idxmin(),
        "model"
    ]
)

best_auc = str(
    summary.loc[
        summary[
            "mean_AUC"
        ].idxmax(),
        "model"
    ]
)


decision = {
    "status":
        "PASS",

    "models":
        summary[
            "model"
        ].tolist(),

    "best_mean_AP":
        best_ap,

    "best_mean_Brier":
        best_brier,

    "best_mean_LogLoss":
        best_logloss,

    "best_mean_AUC":
        best_auc,

    "scientific_question":
        (
            "Does the previous GN–ADMM gain arise "
            "from positive curvature, removal of the "
            "quadratic term, negative curvature, or "
            "the constraint acting as regularization?"
        ),

    "important_guardrail":
        (
            "A predictive benefit from a sign "
            "constraint is not evidence that the "
            "constrained sign is the true structural "
            "shape of the data-generating process."
        ),
}


with open(
    REP
    / "20d_CONSTRAINT_ABLATION_DECISION.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        decision,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("CONSTRAINT ABLATION SUMMARY")
print(
    summary.to_string(
        index=False
    )
)

print()
print(
    json.dumps(
        decision,
        indent=2,
        ensure_ascii=False
    )
)

print()
print(
    "[PASS 20] CONSTRAINT "
    "ABLATION COMPLETE."
)
