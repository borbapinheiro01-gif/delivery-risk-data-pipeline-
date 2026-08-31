#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit

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

with open(
    ART
    / "15_delivery_gnadmm_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    meta = json.load(f)

FEATURE_NAMES = meta[
    "feature_names"
]

m = X.shape[1]

# Constraint:
#
# beta_z2 >= 0
#
# A beta <= b
#
# - beta_z2 <= 0

A = np.zeros(
    (1, m),
    dtype=np.float64
)

A[0, 2] = -1.0

b = np.array(
    [0.0],
    dtype=np.float64
)


def objective(beta):

    p = expit(
        X @ beta
    )

    r = y - p

    return (
        0.5
        *
        float(
            np.mean(
                r * r
            )
        )
    )


def admm_qp_step(
    J,
    r,
    beta_current,
    lambda_reg=1e-4,
    rho=1.0,
    max_admm=5000,
    eps_abs=1e-7,
    eps_rel=1e-5,
):

    n = len(r)

    H = (
        J.T @ J
    ) / n

    H += (
        lambda_reg
        *
        np.eye(m)
    )

    q = (
        J.T @ r
    ) / n

    # Incremental constraint:
    #
    # A(beta + delta) <= b
    #
    # A delta <= b - A beta

    h = (
        b
        -
        A @ beta_current
    )

    K = (
        H
        +
        rho
        *
        (A.T @ A)
    )

    delta = np.zeros(
        m
    )

    z = np.minimum(
        A @ delta,
        h
    )

    u = np.zeros_like(
        z
    )

    history = []

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
            (z - u)
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

        Ad = A @ delta

        z_old = z.copy()

        # Projection on:
        #
        # z <= h

        z = np.minimum(
            Ad + u,
            h
        )

        r_pri_vec = (
            Ad - z
        )

        u = (
            u
            +
            r_pri_vec
        )

        r_dual_vec = (
            rho
            *
            A.T
            @
            (z - z_old)
        )

        r_pri = float(
            np.linalg.norm(
                r_pri_vec
            )
        )

        r_dual = float(
            np.linalg.norm(
                r_dual_vec
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
            np.sqrt(m)
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
            k == 1
            or
            k % 25 == 0
        ):
            history.append(
                {
                    "admm_iteration":
                        k,

                    "r_pri":
                        r_pri,

                    "r_dual":
                        r_dual,

                    "eps_pri":
                        eps_pri,

                    "eps_dual":
                        eps_dual,
                }
            )

        if (
            r_pri <= eps_pri
            and
            r_dual <= eps_dual
        ):
            break

    return (
        delta,
        k,
        r_pri,
        r_dual,
        pd.DataFrame(
            history
        ),
    )


def gnadmm_fit(
    beta_init,
    max_gn=100,
    tol_gn=1e-8,
    lambda_reg=1e-4,
    rho=1.0,
    armijo_c1=1e-4,
    armijo_beta=0.5,
):

    beta = beta_init.copy()

    # Project initial parameter if needed.
    beta[2] = max(
        0.0,
        beta[2]
    )

    outer_hist = []

    admm_hist_all = []

    total_admm = 0

    for j in range(
        1,
        max_gn + 1
    ):

        p = expit(
            X @ beta
        )

        r = y - p

        J = (
            p
            *
            (1.0 - p)
        )[:, None] * X

        delta, k_admm, r_pri, r_dual, h_admm = (
            admm_qp_step(
                J,
                r,
                beta,
                lambda_reg=lambda_reg,
                rho=rho,
            )
        )

        total_admm += k_admm

        if not h_admm.empty:
            h_admm[
                "gn_iteration"
            ] = j

            admm_hist_all.append(
                h_admm
            )

        obj_old = objective(
            beta
        )

        alpha = 1.0

        accepted = False

        for _ in range(30):

            cand = (
                beta
                +
                alpha * delta
            )

            # Numerical protection only.
            # Constraint should already be respected
            # by the incremental ADMM problem.
            if cand[2] < -1e-10:
                alpha *= armijo_beta
                continue

            obj_new = objective(
                cand
            )

            if obj_new < obj_old:
                accepted = True
                break

            alpha *= armijo_beta

        if not accepted:

            cand = (
                beta
                +
                alpha * delta
            )

            cand[2] = max(
                0.0,
                cand[2]
            )

            obj_new = objective(
                cand
            )

        step_norm = float(
            np.linalg.norm(
                alpha * delta
            )
        )

        beta = cand

        violation = max(
            0.0,
            float(
                -beta[2]
            )
        )

        outer_hist.append(
            {
                "gn_iteration":
                    j,

                "objective":
                    obj_new,

                "alpha":
                    alpha,

                "step_norm":
                    step_norm,

                "admm_iterations":
                    k_admm,

                "r_pri_final":
                    r_pri,

                "r_dual_final":
                    r_dual,

                "beta_z2":
                    float(
                        beta[2]
                    ),

                "constraint_violation":
                    violation,
            }
        )

        if (
            step_norm
            <
            tol_gn
        ):
            break

    admm_hist = (
        pd.concat(
            admm_hist_all,
            ignore_index=True
        )
        if admm_hist_all
        else pd.DataFrame()
    )

    return (
        beta,
        pd.DataFrame(
            outer_hist
        ),
        admm_hist,
        total_admm,
    )


print("=" * 110)
print("MODULE 17 — GN–ADMM / BRIER NLLS / beta_z² >= 0")
print("=" * 110)

starts = [
    np.zeros(m),

    np.array(
        [
            -2.5,
            0.0,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),

    np.array(
        [
            -3.0,
            0.1,
            0.10,
            0.1,
            0.05,
            0.05,
            -0.1,
            0.1,
        ]
    ),
]

best_beta = None
best_obj = np.inf
best_outer = None
best_admm = None
best_start = None
best_total_admm = None

summaries = []

for sid, start in enumerate(
    starts
):

    beta, outer, admm_hist, total_admm = (
        gnadmm_fit(
            start
        )
    )

    obj = objective(
        beta
    )

    summaries.append(
        {
            "start_id":
                sid,

            "gn_iterations":
                int(
                    len(outer)
                ),

            "total_admm_iterations":
                int(
                    total_admm
                ),

            "objective":
                float(obj),

            "brier":
                float(
                    2.0 * obj
                ),

            "beta_z2":
                float(
                    beta[2]
                ),

            "constraint_violation":
                float(
                    max(
                        0.0,
                        -beta[2]
                    )
                ),
        }
    )

    if obj < best_obj:
        best_obj = obj
        best_beta = beta.copy()
        best_outer = outer.copy()
        best_admm = admm_hist.copy()
        best_start = sid
        best_total_admm = total_admm


pd.DataFrame(
    summaries
).to_csv(
    REP
    / "17a_gnadmm_multistart_summary.csv",
    index=False
)

best_outer.to_csv(
    REP
    / "17b_gnadmm_outer_history.csv",
    index=False
)

if not best_admm.empty:
    best_admm.to_csv(
        REP
        / "17c_gnadmm_admm_residual_history.csv",
        index=False
    )

coef_df = pd.DataFrame(
    {
        "feature":
            FEATURE_NAMES,

        "beta_gnadmm":
            best_beta,
    }
)

coef_df.to_csv(
    REP
    / "17d_gnadmm_coefficients.csv",
    index=False
)

np.save(
    ART
    / "17_gnadmm_constrained_beta.npy",
    best_beta
)

summary = {
    "status":
        "PASS",

    "method":
        "GN_ADMM_BRIER_NLLS",

    "constraint":
        "beta_z_freight_sq >= 0",

    "selected_start":
        int(best_start),

    "gn_iterations":
        int(
            len(best_outer)
        ),

    "total_admm_iterations":
        int(
            best_total_admm
        ),

    "objective_final":
        float(
            best_obj
        ),

    "brier_in_sample":
        float(
            2.0 * best_obj
        ),

    "beta_z2":
        float(
            best_beta[2]
        ),

    "constraint_violation":
        float(
            max(
                0.0,
                -best_beta[2]
            )
        ),

    "raw_modified":
        False,
}

with open(
    REP
    / "17e_GNADMM_CONSTRAINED_DECISION.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

print()
print(
    pd.DataFrame(
        summaries
    ).to_string(
        index=False
    )
)

print()
print(
    coef_df.to_string(
        index=False
    )
)

print()
print(
    "[PASS 17] GN–ADMM "
    "CONSTRAINED COMPLETE."
)
