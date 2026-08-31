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


def sigmoid_eta(eta):
    return expit(eta)


def objective(beta):
    p = sigmoid_eta(
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


def jacobian(beta):
    p = sigmoid_eta(
        X @ beta
    )

    return (
        p
        *
        (1.0 - p)
    )[:, None] * X


def gn_fit(
    beta_init,
    lambda_reg=1e-4,
    max_iter=100,
    tol=1e-8,
    armijo_c1=1e-4,
    armijo_beta=0.5,
):

    beta = beta_init.copy()

    hist = []

    for it in range(
        1,
        max_iter + 1
    ):

        p = sigmoid_eta(
            X @ beta
        )

        r = y - p

        J = (
            p
            *
            (1.0 - p)
        )[:, None] * X

        n = len(y)

        H = (
            J.T @ J
        ) / n

        H += (
            lambda_reg
            *
            np.eye(
                X.shape[1]
            )
        )

        g_rhs = (
            J.T @ r
        ) / n

        try:
            delta = np.linalg.solve(
                H,
                g_rhs
            )

        except np.linalg.LinAlgError:

            delta = np.linalg.lstsq(
                H,
                g_rhs,
                rcond=None
            )[0]

        step_norm = float(
            np.linalg.norm(
                delta
            )
        )

        obj_old = objective(
            beta
        )

        # Directional derivative for objective:
        #
        # objective = .5 mean(r^2)
        # grad = -J^T r / n
        #
        grad = -g_rhs

        directional = float(
            grad @ delta
        )

        alpha = 1.0

        accepted = False

        for _ in range(30):

            candidate = (
                beta
                +
                alpha * delta
            )

            obj_new = objective(
                candidate
            )

            rhs = (
                obj_old
                +
                armijo_c1
                *
                alpha
                *
                directional
            )

            if (
                obj_new <= rhs
                or
                obj_new < obj_old
            ):
                accepted = True
                break

            alpha *= armijo_beta

        if not accepted:
            candidate = (
                beta
                +
                alpha * delta
            )

            obj_new = objective(
                candidate
            )

        beta = candidate

        hist.append(
            {
                "iteration":
                    it,

                "objective":
                    obj_new,

                "step_norm":
                    step_norm,

                "alpha":
                    alpha,

                "condition_H":
                    float(
                        np.linalg.cond(
                            H
                        )
                    ),
            }
        )

        if (
            alpha
            *
            step_norm
            <
            tol
        ):
            break

    return (
        beta,
        pd.DataFrame(hist)
    )


print("=" * 110)
print("MODULE 16 — REGULARIZED GN / BRIER NLLS")
print("=" * 110)

starts = [
    np.zeros(
        X.shape[1]
    ),
    np.array(
        [
            -2.5,
            0, 0, 0, 0, 0, 0, 0
        ],
        dtype=float
    ),
    np.array(
        [
            -3.0,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            -0.1,
            0.1
        ],
        dtype=float
    ),
]

all_results = []

best_beta = None
best_obj = np.inf
best_hist = None
best_start = None

for s_id, start in enumerate(
    starts
):

    beta, hist = gn_fit(
        start
    )

    obj = objective(
        beta
    )

    all_results.append(
        {
            "start_id":
                s_id,

            "iterations":
                int(len(hist)),

            "objective":
                float(obj),

            "brier":
                float(
                    2.0 * obj
                ),
        }
    )

    if obj < best_obj:
        best_obj = obj
        best_beta = beta.copy()
        best_hist = hist.copy()
        best_start = s_id


pd.DataFrame(
    all_results
).to_csv(
    REP
    / "16a_gn_multistart_summary.csv",
    index=False
)

best_hist.to_csv(
    REP
    / "16b_gn_iteration_history.csv",
    index=False
)

coef_df = pd.DataFrame(
    {
        "feature":
            FEATURE_NAMES,

        "beta_gn":
            best_beta,
    }
)

coef_df.to_csv(
    REP
    / "16c_gn_coefficients.csv",
    index=False
)

np.save(
    ART
    / "16_gn_unconstrained_beta.npy",
    best_beta
)

summary = {
    "status":
        "PASS",

    "method":
        "REGULARIZED_GAUSS_NEWTON",

    "objective":
        "HALF_MEAN_BRIER_NLLS",

    "selected_start":
        int(best_start),

    "iterations":
        int(len(best_hist)),

    "objective_final":
        float(best_obj),

    "brier_in_sample":
        float(
            2.0 * best_obj
        ),

    "beta_z2":
        float(
            best_beta[2]
        ),

    "constraint_enforced":
        False,

    "raw_modified":
        False,
}

with open(
    REP
    / "16d_GN_UNCONSTRAINED_DECISION.json",
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
        all_results
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
    "[PASS 16] REGULARIZED "
    "GAUSS–NEWTON COMPLETE."
)
