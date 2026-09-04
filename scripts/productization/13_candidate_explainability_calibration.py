#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import math
import platform
import warnings

import duckdb
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]

OUT = (
    REPO
    / "reports/productization/"
    / "candidate_explainability_calibration"
)

DB = REPO / "data/bronze/olist_medallion.duckdb"

M1_FEATURES = (
    REPO
    / "reports/productization/m1_feature_materialization/"
    / "03_m1_candidate_features.csv"
)

LABEL_SOURCE = (
    REPO
    / "artifacts/spatiotemporal_logistics/"
    / "06_EXPECTED_FREIGHT_OOT.csv"
)

CANDIDATE_CONTRACT = REPO / "configs/production_candidate_v1.json"
BLOCK11_CONTRACT = REPO / "configs/algorithm_benchmark_v1.json"
BLOCK10_CONTRACT = (
    REPO / "configs/m0_vs_m1_paired_logistic_benchmark_v1.json"
)

BLOCK11_OOF = (
    REPO
    / "reports/productization/algorithm_benchmark/"
    / "06_oof_predictions.csv"
)

BLOCK11_FOLDS = (
    REPO
    / "reports/productization/algorithm_benchmark/"
    / "02_fold_contract_parity.csv"
)

CONTRACT = (
    REPO / "configs/candidate_explainability_calibration_v1.json"
)

DOC = (
    REPO
    / "docs/architecture/"
    / "CANDIDATE_EXPLAINABILITY_CALIBRATION_V1.md"
)

M0 = [
    "item_count",
    "unique_product_count",
    "unique_seller_count",
    "total_price",
    "mean_price",
    "max_price",
    "min_price",
    "price_range",
    "total_freight",
    "mean_freight",
    "max_freight",
    "merchandise_plus_freight",
    "freight_price_ratio",
]

TARGET = "late_delivery_calendar_day"
PRED_TIME = "prediction_time"
LABEL_AVAILABLE_TIME = "order_delivered_customer_date"

PERM_REPEATS = 5
PERM_SEED = 20260904
PRED_ATOL = 1e-12
PRED_RTOL = 1e-10
LOGIT_IDENTITY_ATOL = 1e-10


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def make_model(logit_kwargs):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(**logit_kwargs)),
    ])


def calibration_intercept_slope(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    x = np.log(p / (1.0 - p))

    def nll(theta):
        a, b = theta
        z = np.clip(a + b * x, -40.0, 40.0)
        q = 1.0 / (1.0 + np.exp(-z))
        q = np.clip(q, eps, 1.0 - eps)
        return -np.sum(
            y * np.log(q)
            + (1.0 - y) * np.log(1.0 - q)
        )

    result = minimize(
        nll,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )

    if not result.success:
        # BFGS may report precision warnings despite a stable optimum.
        # Accept only if the returned values are finite.
        if not np.isfinite(result.x).all():
            raise RuntimeError(
                f"Calibration optimization failed: {result.message}"
            )

    return float(result.x[0]), float(result.x[1])


def reliability_bins(y, p, n_bins=10):
    df = pd.DataFrame({
        "y": np.asarray(y, dtype=int),
        "p": np.asarray(p, dtype=float),
    })

    # Quantile bins make the diagnostic stable under class imbalance.
    df["bin"] = pd.qcut(
        df["p"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for bin_id, g in df.groupby(
        "bin",
        dropna=False,
        sort=True,
    ):
        rows.append({
            "bin": int(bin_id) if pd.notna(bin_id) else -1,
            "rows": len(g),
            "mean_predicted_probability": float(g["p"].mean()),
            "observed_late_rate": float(g["y"].mean()),
            "absolute_gap": float(
                abs(g["p"].mean() - g["y"].mean())
            ),
            "min_score": float(g["p"].min()),
            "max_score": float(g["p"].max()),
        })

    bins = pd.DataFrame(rows)
    total = bins["rows"].sum()

    bins["weight"] = bins["rows"] / total

    ece = float(
        (
            bins["weight"]
            * bins["absolute_gap"]
        ).sum()
    )

    mce = float(bins["absolute_gap"].max())

    return bins, ece, mce


def main():
    candidate_cfg = json.loads(
        CANDIDATE_CONTRACT.read_text(encoding="utf-8")
    )
    block11 = json.loads(
        BLOCK11_CONTRACT.read_text(encoding="utf-8")
    )
    block10 = json.loads(
        BLOCK10_CONTRACT.read_text(encoding="utf-8")
    )

    candidate = candidate_cfg["governed_candidate"]

    if candidate["algorithm"] != "LOGISTIC" or candidate["arm"] != "M0":
        raise RuntimeError(
            f"Candidate drift: {candidate['algorithm']}/{candidate['arm']}"
        )

    if candidate_cfg["paired_population_rows"] != 91254:
        raise RuntimeError("Candidate population drift.")

    if candidate_cfg["folds"] != 14:
        raise RuntimeError("Candidate fold drift.")

    logit_kwargs = dict(
        block10["estimator"]["logistic_kwargs"]
    )

    con = duckdb.connect(str(DB), read_only=True)

    gold = con.execute(
        """
        SELECT *
        FROM gold.ml_delivery_supervised_core_v1
        ORDER BY prediction_time, order_id
        """
    ).df()

    con.close()

    m1f = pd.read_csv(M1_FEATURES)

    labels = pd.read_csv(
        LABEL_SOURCE,
        usecols=[
            "order_id",
            LABEL_AVAILABLE_TIME,
        ],
    )

    oof_ref = pd.read_csv(BLOCK11_OOF)
    folds_ref = pd.read_csv(BLOCK11_FOLDS)

    for name, df in [
        ("gold", gold),
        ("m1_features", m1f),
        ("label_source", labels),
        ("oof_ref", oof_ref),
    ]:
        if df["order_id"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate order_id")

    if "LOGISTIC__M0" not in oof_ref.columns:
        raise RuntimeError("Block-11 LOGISTIC__M0 OOF column missing.")

    data = gold.merge(
        m1f[
            [
                "order_id",
                "m1_feature_vector_available",
            ]
        ],
        on="order_id",
        how="left",
        validate="1:1",
    ).merge(
        labels,
        on="order_id",
        how="left",
        validate="1:1",
    )

    data[PRED_TIME] = pd.to_datetime(
        data[PRED_TIME],
        errors="coerce",
    )

    data[LABEL_AVAILABLE_TIME] = pd.to_datetime(
        data[LABEL_AVAILABLE_TIME],
        errors="coerce",
    )

    eligible = data[
        data["m1_feature_vector_available"]
        .fillna(False)
        .astype(bool)
    ].copy()

    if len(eligible) != 91254:
        raise RuntimeError(
            f"Expected 91,254 paired rows; found {len(eligible)}"
        )

    if eligible[M0].isna().any().any():
        missing = eligible[M0].isna().sum()
        raise RuntimeError(
            f"Candidate feature missingness: "
            f"{missing[missing > 0].to_dict()}"
        )

    eligible["test_month"] = (
        eligible[PRED_TIME]
        .dt.to_period("M")
        .astype(str)
    )

    if len(folds_ref) != 14:
        raise RuntimeError(
            f"Expected 14 Block-11 fold rows; found {len(folds_ref)}"
        )

    coef_rows = []
    permutation_rows = []
    identity_rows = []
    parity_rows = []
    top_driver_counts = {feature: 0 for feature in M0}
    abs_contrib_sum = {feature: 0.0 for feature in M0}
    contrib_rows_total = 0

    oof_ref_index = (
        oof_ref.set_index("order_id")[
            [
                "y_true",
                "LOGISTIC__M0",
                "fold_id",
                "test_month",
            ]
        ]
    )

    rng = np.random.default_rng(PERM_SEED)

    for fold in folds_ref.itertuples(index=False):
        fold_id = int(fold.fold_id)
        test_month = str(fold.test_month)
        cutoff = pd.Timestamp(f"{test_month}-01")

        train = eligible[
            (eligible[PRED_TIME] < cutoff)
            & (eligible[LABEL_AVAILABLE_TIME] < cutoff)
        ].copy()

        test = eligible[
            eligible["test_month"].eq(test_month)
        ].copy()

        if len(train) != int(fold.recomputed_n_train):
            raise RuntimeError(
                f"Fold {fold_id} train-count drift."
            )

        if len(test) != int(fold.recomputed_n_test):
            raise RuntimeError(
                f"Fold {fold_id} test-count drift."
            )

        y_train = train[TARGET].astype(int).to_numpy()
        y_test = test[TARGET].astype(int).to_numpy()

        X_train = train[M0].astype(float)
        X_test = test[M0].astype(float)

        model = make_model(logit_kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model.fit(X_train, y_train)

        p = model.predict_proba(X_test)[:, 1]

        ref = (
            oof_ref_index.loc[
                test["order_id"].astype(str),
                "LOGISTIC__M0",
            ]
            .to_numpy(dtype=float)
        )

        close = np.isclose(
            p,
            ref,
            atol=PRED_ATOL,
            rtol=PRED_RTOL,
        )

        max_abs_pred_diff = float(
            np.max(np.abs(p - ref))
        )

        parity_rows.append({
            "fold_id": fold_id,
            "test_month": test_month,
            "rows": len(test),
            "prediction_mismatch": int((~close).sum()),
            "max_abs_prediction_diff": max_abs_pred_diff,
            "status": "PASS" if close.all() else "FAIL",
        })

        if not close.all():
            raise RuntimeError(
                f"Candidate prediction parity failed in fold {fold_id}: "
                f"mismatch={(~close).sum()}, maxdiff={max_abs_pred_diff}"
            )

        scaler = model.named_steps["scale"]
        logistic = model.named_steps["model"]

        beta = logistic.coef_[0].astype(float)
        intercept = float(logistic.intercept_[0])

        transformed = scaler.transform(X_test)

        manual_logit = intercept + transformed @ beta
        model_logit = model.decision_function(X_test)

        identity_diff = np.abs(
            manual_logit - model_logit
        )

        max_identity_diff = float(identity_diff.max())

        identity_rows.append({
            "fold_id": fold_id,
            "test_month": test_month,
            "rows": len(test),
            "max_abs_logit_identity_diff": max_identity_diff,
            "status": (
                "PASS"
                if max_identity_diff <= LOGIT_IDENTITY_ATOL
                else "FAIL"
            ),
        })

        if max_identity_diff > LOGIT_IDENTITY_ATOL:
            raise RuntimeError(
                f"Linear contribution identity failed in fold {fold_id}."
            )

        contrib = transformed * beta

        abs_contrib = np.abs(contrib)
        top_idx = np.argmax(abs_contrib, axis=1)

        for j, feature in enumerate(M0):
            feature_abs = abs_contrib[:, j]
            abs_contrib_sum[feature] += float(
                feature_abs.sum()
            )

            coef_rows.append({
                "fold_id": fold_id,
                "test_month": test_month,
                "feature": feature,
                "standardized_coefficient": float(beta[j]),
                "odds_ratio_per_1sd": float(np.exp(beta[j])),
                "abs_standardized_coefficient": float(abs(beta[j])),
                "raw_unit_logodds_coefficient": float(
                    beta[j] / scaler.scale_[j]
                ) if scaler.scale_[j] != 0 else np.nan,
                "train_mean": float(scaler.mean_[j]),
                "train_scale": float(scaler.scale_[j]),
            })

        for j, feature in enumerate(M0):
            top_driver_counts[feature] += int(
                (top_idx == j).sum()
            )

        contrib_rows_total += len(test)

        baseline_ap = average_precision_score(
            y_test,
            p,
        )

        for feature in M0:
            j = M0.index(feature)

            for repeat in range(PERM_REPEATS):
                X_perm = X_test.copy()

                values = X_perm.iloc[:, j].to_numpy(copy=True)
                rng.shuffle(values)
                X_perm.iloc[:, j] = values

                p_perm = model.predict_proba(
                    X_perm
                )[:, 1]

                perm_ap = average_precision_score(
                    y_test,
                    p_perm,
                )

                permutation_rows.append({
                    "fold_id": fold_id,
                    "test_month": test_month,
                    "feature": feature,
                    "repeat": repeat + 1,
                    "baseline_ap": float(baseline_ap),
                    "permuted_ap": float(perm_ap),
                    "ap_drop": float(
                        baseline_ap - perm_ap
                    ),
                })

    parity = pd.DataFrame(parity_rows)
    coefs = pd.DataFrame(coef_rows)
    permutation = pd.DataFrame(permutation_rows)
    identity = pd.DataFrame(identity_rows)

    parity.to_csv(
        OUT / "01_candidate_prediction_parity.csv",
        index=False,
    )

    coefs.to_csv(
        OUT / "02_fold_standardized_coefficients.csv",
        index=False,
    )

    permutation.to_csv(
        OUT / "03_fold_permutation_ap_sensitivity.csv",
        index=False,
    )

    identity.to_csv(
        OUT / "04_linear_logit_identity.csv",
        index=False,
    )

    if not parity["status"].eq("PASS").all():
        raise RuntimeError("Prediction parity not fully PASS.")

    if not identity["status"].eq("PASS").all():
        raise RuntimeError("Linear identity not fully PASS.")

    coef_summary = (
        coefs.groupby("feature")
        .agg(
            mean_standardized_coefficient=(
                "standardized_coefficient",
                "mean",
            ),
            median_standardized_coefficient=(
                "standardized_coefficient",
                "median",
            ),
            mean_abs_standardized_coefficient=(
                "abs_standardized_coefficient",
                "mean",
            ),
            mean_odds_ratio_per_1sd=(
                "odds_ratio_per_1sd",
                "mean",
            ),
            positive_folds=(
                "standardized_coefficient",
                lambda x: int((x > 0).sum()),
            ),
            negative_folds=(
                "standardized_coefficient",
                lambda x: int((x < 0).sum()),
            ),
            zero_folds=(
                "standardized_coefficient",
                lambda x: int((x == 0).sum()),
            ),
        )
        .reset_index()
    )

    perm_summary = (
        permutation.groupby("feature")
        .agg(
            mean_ap_drop=("ap_drop", "mean"),
            median_ap_drop=("ap_drop", "median"),
            min_ap_drop=("ap_drop", "min"),
            max_ap_drop=("ap_drop", "max"),
        )
        .reset_index()
    )

    explain = coef_summary.merge(
        perm_summary,
        on="feature",
        validate="1:1",
    )

    explain["mean_abs_logit_contribution"] = explain[
        "feature"
    ].map(
        {
            feature: (
                abs_contrib_sum[feature]
                / contrib_rows_total
            )
            for feature in M0
        }
    )

    explain["top_driver_count"] = explain[
        "feature"
    ].map(top_driver_counts)

    explain["top_driver_share"] = (
        explain["top_driver_count"]
        / contrib_rows_total
    )

    explain = explain.sort_values(
        [
            "mean_ap_drop",
            "mean_abs_logit_contribution",
            "mean_abs_standardized_coefficient",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    explain.insert(
        0,
        "importance_rank",
        np.arange(1, len(explain) + 1),
    )

    explain.to_csv(
        OUT / "05_global_explainability_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # Calibration from frozen OOF predictions.
    # ----------------------------------------------------------------------

    y_oof = oof_ref["y_true"].astype(int).to_numpy()
    p_oof = oof_ref["LOGISTIC__M0"].astype(float).to_numpy()

    pooled_ap = float(
        average_precision_score(
            y_oof,
            p_oof,
        )
    )

    pooled_auc = float(
        safe_auc(
            y_oof,
            p_oof,
        )
    )

    pooled_brier = float(
        brier_score_loss(
            y_oof,
            p_oof,
        )
    )

    pooled_log_loss = float(
        log_loss(
            y_oof,
            p_oof,
            labels=[0, 1],
        )
    )

    cal_intercept, cal_slope = calibration_intercept_slope(
        y_oof,
        p_oof,
    )

    bins, ece, mce = reliability_bins(
        y_oof,
        p_oof,
        n_bins=10,
    )

    bins.to_csv(
        OUT / "06_oof_calibration_bins.csv",
        index=False,
    )

    fold_cal_rows = []

    for fold_id, g in oof_ref.groupby(
        "fold_id",
        sort=True,
    ):
        y = g["y_true"].astype(int).to_numpy()
        p = g["LOGISTIC__M0"].astype(float).to_numpy()

        _, fold_ece, fold_mce = reliability_bins(
            y,
            p,
            n_bins=10,
        )

        fold_cal_rows.append({
            "fold_id": int(fold_id),
            "test_month": str(g["test_month"].iloc[0]),
            "rows": len(g),
            "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "average_precision": float(
                average_precision_score(y, p)
            ),
            "roc_auc": float(safe_auc(y, p)),
            "brier": float(
                brier_score_loss(y, p)
            ),
            "log_loss": float(
                log_loss(
                    y,
                    p,
                    labels=[0, 1],
                )
            ),
            "ece_quantile_10": float(fold_ece),
            "mce_quantile_10": float(fold_mce),
            "mean_score": float(p.mean()),
        })

    fold_cal = pd.DataFrame(fold_cal_rows)

    fold_cal.to_csv(
        OUT / "07_fold_calibration_metrics.csv",
        index=False,
    )

    calibration = {
        "oof_rows": int(len(oof_ref)),
        "oof_positives": int(y_oof.sum()),
        "oof_prevalence": float(y_oof.mean()),
        "pooled_average_precision": pooled_ap,
        "pooled_roc_auc": pooled_auc,
        "pooled_brier": pooled_brier,
        "pooled_log_loss": pooled_log_loss,
        "mean_predicted_probability": float(p_oof.mean()),
        "calibration_intercept_ideal_0": cal_intercept,
        "calibration_slope_ideal_1": cal_slope,
        "ece_quantile_10": ece,
        "mce_quantile_10": mce,
        "score_quantiles": {
            str(q): float(np.quantile(p_oof, q))
            for q in [
                0.0,
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.0,
            ]
        },
    }

    (OUT / "08_calibration_summary.json").write_text(
        json.dumps(
            calibration,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # Diagnostic classification only; no calibrator is applied here.
    abs_intercept = abs(cal_intercept)
    slope_dev = abs(cal_slope - 1.0)

    if (
        abs_intercept <= 0.10
        and slope_dev <= 0.10
        and ece <= 0.02
    ):
        calibration_class = "CALIBRATION_DIAGNOSTIC_ACCEPTABLE"
    else:
        calibration_class = "CALIBRATION_REVIEW_REQUIRED_BEFORE_PRODUCTION"

    top_features = explain.head(5)[
        [
            "importance_rank",
            "feature",
            "mean_ap_drop",
            "mean_abs_logit_contribution",
            "mean_standardized_coefficient",
        ]
    ].to_dict(orient="records")

    contract = {
        "contract": "CANDIDATE_EXPLAINABILITY_CALIBRATION_V1",
        "status": (
            "CANDIDATE_EXPLAINABILITY_CALIBRATION_AUDIT_COMPLETE"
        ),
        "candidate": {
            "algorithm": "LOGISTIC",
            "arm": "M0",
            "paired_population_rows": 91254,
            "folds": 14,
        },
        "prediction_parity": {
            "block11_oof_parity": "PASS",
            "total_mismatches": int(
                parity["prediction_mismatch"].sum()
            ),
            "max_abs_prediction_diff": float(
                parity["max_abs_prediction_diff"].max()
            ),
        },
        "explainability": {
            "standardized_coefficients": True,
            "odds_ratio_per_1sd": True,
            "heldout_permutation_ap_sensitivity": True,
            "permutation_repeats": PERM_REPEATS,
            "exact_logit_contribution_identity": "PASS",
            "top_features": top_features,
            "causal_interpretation": False,
            "collinearity_warning": (
                "ORDER_CORE_V1 contains known algebraic dependencies; "
                "coefficient and one-feature permutation rankings are model "
                "sensitivity diagnostics, not causal effects."
            ),
            "shap_required_for_linear_candidate": False,
        },
        "calibration": {
            **calibration,
            "classification": calibration_class,
            "calibrator_applied": False,
        },
        "governance": {
            "threshold_selected": False,
            "topk_budget_selected": False,
            "calibration_transform_selected": False,
            "production_model_released": False,
            "full_frozen_cohort_backtest_completed": False,
        },
        "next_gate": (
            "FULL_FROZEN_COHORT_M0_BACKTEST_AND_CALIBRATION_DECISION"
        ),
    }

    CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    with (
        OUT / "09_explainability_calibration_decision.txt"
    ).open("w", encoding="utf-8") as fh:
        fh.write("=" * 100 + "\n")
        fh.write(
            "DELIVERY RISK — BLOCK 13 EXPLAINABILITY + CALIBRATION DECISION\n"
        )
        fh.write("=" * 100 + "\n\n")
        fh.write("CANDIDATE                    : LOGISTIC/M0\n")
        fh.write("PAIRED_POPULATION_ROWS       : 91254\n")
        fh.write("VALID_TEMPORAL_FOLDS         : 14\n")
        fh.write(
            f"PREDICTION_PARITY_MISMATCHES : "
            f"{int(parity['prediction_mismatch'].sum())}\n"
        )
        fh.write(
            f"MAX_ABS_PREDICTION_DIFF      : "
            f"{parity['max_abs_prediction_diff'].max():.3e}\n"
        )
        fh.write("LOGIT_CONTRIBUTION_IDENTITY  : PASS\n")
        fh.write(
            f"POOLED_OOF_AP                : {pooled_ap:.12f}\n"
        )
        fh.write(
            f"POOLED_OOF_AUC               : {pooled_auc:.12f}\n"
        )
        fh.write(
            f"POOLED_OOF_BRIER             : {pooled_brier:.12f}\n"
        )
        fh.write(
            f"POOLED_OOF_LOG_LOSS          : {pooled_log_loss:.12f}\n"
        )
        fh.write(
            f"CALIBRATION_INTERCEPT        : {cal_intercept:.12f}\n"
        )
        fh.write(
            f"CALIBRATION_SLOPE            : {cal_slope:.12f}\n"
        )
        fh.write(
            f"ECE_QUANTILE_10              : {ece:.12f}\n"
        )
        fh.write(
            f"MCE_QUANTILE_10              : {mce:.12f}\n"
        )
        fh.write(
            f"CALIBRATION_CLASS            : {calibration_class}\n"
        )
        fh.write("CALIBRATOR_APPLIED           : NO\n")
        fh.write("THRESHOLD_SELECTED           : NO\n")
        fh.write("TOPK_BUDGET_SELECTED         : NO\n")
        fh.write("PRODUCTION_MODEL_RELEASED    : NO\n")
        fh.write(
            "NEXT_GATE                    : "
            "FULL_FROZEN_COHORT_M0_BACKTEST_AND_CALIBRATION_DECISION\n\n"
        )
        fh.write(
            "STATUS = "
            "CANDIDATE_EXPLAINABILITY_CALIBRATION_AUDIT_COMPLETE\n"
        )

    DOC.write_text(
        f"""# Candidate Explainability + Calibration V1

## Candidate

`LOGISTIC / M0`

This audit stays on the same 91,254 paired orders and the same 14 temporal folds
used for model selection.

## Reproduction parity

The candidate is re-fit independently on each governed fold and its predictions
are checked against the frozen Block-11 OOF scores.

- Prediction mismatches: {int(parity["prediction_mismatch"].sum())}
- Maximum absolute prediction difference:
  {parity["max_abs_prediction_diff"].max():.3e}

## Explainability

The selected model is linear in standardized feature space. Therefore each logit
has an exact additive decomposition:

`logit(p) = intercept + Σ beta_j * z_j`

The implementation verifies this identity numerically for every fold.

The audit reports:

- standardized coefficients;
- odds ratio per one-standard-deviation change;
- coefficient sign stability;
- held-out permutation AP sensitivity;
- mean absolute logit contribution;
- most frequent top local driver.

These quantities describe model behavior, not causal effects.

ORDER_CORE_V1 has known algebraic dependencies, so coefficient and one-feature
permutation rankings should not be read as unique structural importance.

A SHAP dependency is not necessary to decompose this linear candidate exactly.

## Calibration

Frozen OOF diagnostic:

- rows: {len(oof_ref)}
- prevalence: {y_oof.mean():.12f}
- pooled AP: {pooled_ap:.12f}
- pooled ROC-AUC: {pooled_auc:.12f}
- Brier: {pooled_brier:.12f}
- Log Loss: {pooled_log_loss:.12f}
- calibration intercept (ideal 0): {cal_intercept:.12f}
- calibration slope (ideal 1): {cal_slope:.12f}
- quantile-bin ECE: {ece:.12f}
- quantile-bin MCE: {mce:.12f}

Diagnostic classification:

`{calibration_class}`

No calibration transform is fitted or selected here.

## Release state

No threshold, Top-K budget, calibration transform, or production model is
released.

The next gate must backtest the selected M0 candidate on the entire frozen
supervised cohort, because candidate selection occurred on the 91,254-row
paired M1-eligible population while M0 itself is available on all 96,470 frozen
Gold orders.
""",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("GLOBAL EXPLAINABILITY SUMMARY")
    print("=" * 100)
    print(explain.to_string(index=False))

    print()
    print("=" * 100)
    print("CALIBRATION SUMMARY")
    print("=" * 100)
    print(json.dumps(
        calibration,
        indent=2,
        ensure_ascii=False,
    ))

    print()
    print(
        (OUT / "09_explainability_calibration_decision.txt")
        .read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
