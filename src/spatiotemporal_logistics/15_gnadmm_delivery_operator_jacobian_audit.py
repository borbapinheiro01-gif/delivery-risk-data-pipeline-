#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

INPUT = (
    ROOT
    / "artifacts"
    / "spatiotemporal_logistics"
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

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

ART.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

USECOLS = [
    "late_delivery_calendar_day",
    "total_freight",
    "expected_freight_oot",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "promised_delivery_days",
    "any_interstate_route",
    "year_month",
]

FEATURE_NAMES = [
    "intercept",
    "z_freight",
    "z_freight_sq",
    "log_distance",
    "log_weight",
    "log_volume",
    "promised_days",
    "interstate",
]


def safe_log1p(s):
    return np.log1p(
        pd.to_numeric(s, errors="coerce").where(
            pd.to_numeric(s, errors="coerce") >= 0
        )
    )


def operator(beta, X):
    return expit(X @ beta)


def residual(beta, X, y):
    return y - operator(beta, X)


def analytic_jacobian(beta, X):
    p = operator(beta, X)
    return (p * (1.0 - p))[:, None] * X


print("=" * 110)
print("MODULE 15 — GN–ADMM DELIVERY OPERATOR / JACOBIAN AUDIT")
print("=" * 110)

df = pd.read_csv(
    INPUT,
    usecols=USECOLS,
)

valid = (
    df["total_freight"].ge(0)
    & df["expected_freight_oot"].gt(0)
)

df = df.loc[valid].copy()

EPS = 1e-4

df["z_freight"] = np.log(
    (df["total_freight"] + EPS)
    /
    (df["expected_freight_oot"] + EPS)
)

df["z_freight_sq"] = (
    df["z_freight"] ** 2
)

df["log_distance"] = safe_log1p(
    df["distance_freight_weighted_km"]
)

df["log_weight"] = safe_log1p(
    df["total_weight_g"]
)

df["log_volume"] = safe_log1p(
    df["product_volume_sum_proxy_cm3"]
)

required = [
    "late_delivery_calendar_day",
    "z_freight",
    "z_freight_sq",
    "log_distance",
    "log_weight",
    "log_volume",
    "promised_delivery_days",
    "any_interstate_route",
    "year_month",
]

df = (
    df
    .dropna(subset=required)
    .reset_index(drop=True)
)

continuous = [
    "z_freight",
    "z_freight_sq",
    "log_distance",
    "log_weight",
    "log_volume",
    "promised_delivery_days",
]

scaler = {}

for col in continuous:
    mu = float(df[col].mean())
    sd = float(df[col].std())

    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(
            f"Variável sem variabilidade: {col}"
        )

    scaler[col] = {
        "mean": mu,
        "std": sd,
    }

    df[col + "_std"] = (
        df[col] - mu
    ) / sd


X = np.column_stack(
    [
        np.ones(len(df)),
        df["z_freight_std"].to_numpy(),
        df["z_freight_sq_std"].to_numpy(),
        df["log_distance_std"].to_numpy(),
        df["log_weight_std"].to_numpy(),
        df["log_volume_std"].to_numpy(),
        df["promised_delivery_days_std"].to_numpy(),
        df["any_interstate_route"].astype(float).to_numpy(),
    ]
).astype(np.float64)

y = (
    df["late_delivery_calendar_day"]
    .astype(float)
    .to_numpy()
)

months = (
    df["year_month"]
    .astype(str)
    .to_numpy(dtype="U7")
)


# --------------------------------------------------------------------------------------------------
# OPERATOR AUDIT
# --------------------------------------------------------------------------------------------------

beta0 = np.zeros(
    X.shape[1],
    dtype=np.float64
)

p0 = operator(
    beta0,
    X
)

operator_ok = bool(
    np.isfinite(p0).all()
    and
    (p0 > 0).all()
    and
    (p0 < 1).all()
)


# --------------------------------------------------------------------------------------------------
# JACOBIAN FINITE DIFFERENCE AUDIT
# --------------------------------------------------------------------------------------------------

rng = np.random.default_rng(42)

n_audit = min(
    250,
    len(df)
)

idx = rng.choice(
    len(df),
    size=n_audit,
    replace=False
)

X_a = X[idx]

beta_test = np.array(
    [
        -2.0,
        0.15,
        0.10,
        0.25,
        0.08,
        0.03,
        -0.12,
        0.10,
    ],
    dtype=np.float64
)

J_analytic = analytic_jacobian(
    beta_test,
    X_a
)

hs = [
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
]

rows = []

for h in hs:

    J_fd = np.zeros_like(
        J_analytic
    )

    for j in range(
        X_a.shape[1]
    ):
        e = np.zeros(
            X_a.shape[1]
        )

        e[j] = h

        fp = operator(
            beta_test + e,
            X_a
        )

        fm = operator(
            beta_test - e,
            X_a
        )

        J_fd[:, j] = (
            fp - fm
        ) / (2.0 * h)

    abs_err = np.abs(
        J_analytic - J_fd
    )

    denom = np.maximum(
        np.abs(J_fd),
        1e-12
    )

    rel_err = (
        abs_err / denom
    )

    rows.append(
        {
            "h":
                h,

            "max_abs_error":
                float(
                    np.max(abs_err)
                ),

            "mean_abs_error":
                float(
                    np.mean(abs_err)
                ),

            "max_rel_error":
                float(
                    np.max(rel_err)
                ),

            "mean_rel_error":
                float(
                    np.mean(rel_err)
                ),
        }
    )


jac_df = pd.DataFrame(
    rows
)

jac_df.to_csv(
    REP
    / "15a_jacobian_finite_difference_audit.csv",
    index=False
)

best_row = (
    jac_df
    .sort_values(
        "max_abs_error"
    )
    .iloc[0]
)

jacobian_ok = bool(
    best_row[
        "max_abs_error"
    ] < 1e-5
)


# --------------------------------------------------------------------------------------------------
# SAVE DATA CHECKPOINT
# --------------------------------------------------------------------------------------------------

np.savez_compressed(
    ART
    / "15_delivery_gnadmm_design.npz",
    X=X,
    y=y,
    months=months,
)

metadata = {
    "status":
        "PASS"
        if operator_ok and jacobian_ok
        else "FAIL",

    "n_orders":
        int(len(df)),

    "n_parameters":
        int(X.shape[1]),

    "feature_names":
        FEATURE_NAMES,

    "objective":
        "HALF_MEAN_BRIER_NLLS",

    "operator":
        "sigmoid(X beta)",

    "residual":
        "y - sigmoid(X beta)",

    "jacobian":
        "p*(1-p)*X",

    "operator_valid":
        operator_ok,

    "jacobian_valid":
        jacobian_ok,

    "jacobian_best_h":
        float(
            best_row["h"]
        ),

    "jacobian_best_max_abs_error":
        float(
            best_row[
                "max_abs_error"
            ]
        ),

    "scaler":
        scaler,

    "primary_constraint_candidate":
        "beta_z_freight_sq >= 0",

    "raw_modified":
        False,
}

with open(
    ART
    / "15_delivery_gnadmm_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"[PASS] N analytical = "
    f"{len(df):,}"
)

print(
    f"[PASS] parameters = "
    f"{X.shape[1]}"
)

print(
    "[PASS] operator probabilities "
    "strictly inside (0,1)"
)

print()
print("JACOBIAN AUDIT")
print(
    jac_df.to_string(
        index=False
    )
)

print()

if not operator_ok:
    raise RuntimeError(
        "Operator audit failed."
    )

if not jacobian_ok:
    raise RuntimeError(
        "Analytic Jacobian audit failed."
    )

print(
    "[PASS 15] OPERATOR + "
    "ANALYTIC JACOBIAN VALIDATED."
)
