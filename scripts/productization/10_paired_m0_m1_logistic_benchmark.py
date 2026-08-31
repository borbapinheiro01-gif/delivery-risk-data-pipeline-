#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import platform
import warnings

import duckdb
import numpy as np
import pandas as pd
import scipy
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

OUT = REPO / "reports/productization/m0_vs_m1_paired_logistic"
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
M1_CONTRACT = REPO / "configs/m1_feature_contract_v1.json"
FROZEN_TEMPORAL = REPO / "configs/model_01_supervised_temporal_contract_v1.json"
BENCHMARK_CONTRACT = REPO / "configs/m0_vs_m1_paired_logistic_benchmark_v1.json"
DOC = REPO / "docs/architecture/M0_VS_M1_PAIRED_LOGISTIC_BENCHMARK_V1.md"

OUT.mkdir(parents=True, exist_ok=True)

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

M1_EXTRA = [
    "expected_freight_oot",
    "freight_burden",
    "freight_expected_ratio",
    "freight_log_ratio",
]

M1 = M0 + M1_EXTRA

TARGET = "late_delivery_calendar_day"
PRED_TIME = "prediction_time"
LABEL_AVAILABLE_TIME = "order_delivered_customer_date"

MIN_PRIOR_ELIGIBLE_MONTHS = 3
TOP_FRACTIONS = [0.05, 0.10, 0.20]
N_BOOT = 20000
BOOT_SEED = 20260831

# Explicit L2 logistic regression with version-aware sklearn API.
# sklearn >= 1.8 deprecates the penalty parameter; l1_ratio=0.0 is L2.
_version_parts = []
for _token in sklearn.__version__.split(".")[:2]:
    _digits = "".join(ch for ch in _token if ch.isdigit())
    _version_parts.append(int(_digits or 0))
_SKLEARN_MAJOR_MINOR = tuple(_version_parts)

if _SKLEARN_MAJOR_MINOR >= (1, 8):
    LOGIT_KWARGS = {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
    }
else:
    LOGIT_KWARGS = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(**LOGIT_KWARGS)),
    ])


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def recall_at_fraction(y_true, score, fraction):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    positives = int(y_true.sum())
    if positives == 0:
        return np.nan, 0

    n = len(y_true)
    k = max(1, int(math.ceil(fraction * n)))

    order = np.argsort(-score, kind="mergesort")
    selected = order[:k]

    recall = float(y_true[selected].sum() / positives)
    return recall, k


def main():
    for path in [
        DB,
        M1_FEATURES,
        LABEL_SOURCE,
        M1_CONTRACT,
        FROZEN_TEMPORAL,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    m1_contract = json.loads(
        M1_CONTRACT.read_text(encoding="utf-8")
    )

    frozen_temporal = json.loads(
        FROZEN_TEMPORAL.read_text(encoding="utf-8")
    )

    if m1_contract["status"] != "READY_FOR_PAIRED_M0_VS_M1_BENCHMARK":
        raise RuntimeError("M1 feature contract drift.")

    con = duckdb.connect(str(DB), read_only=True)

    gold = con.execute(
        """
        SELECT *
        FROM gold.ml_delivery_supervised_core_v1
        ORDER BY prediction_time, order_id
        """
    ).df()

    con.close()

    if len(gold) != 96470:
        raise RuntimeError(f"Gold row drift: {len(gold)}")

    if gold["order_id"].duplicated().any():
        raise RuntimeError("Gold order_id duplicates.")

    required_gold = ["order_id", PRED_TIME, TARGET] + M0
    missing_gold = [c for c in required_gold if c not in gold.columns]
    if missing_gold:
        raise RuntimeError(f"Gold missing columns: {missing_gold}")

    m1f = pd.read_csv(M1_FEATURES)

    if len(m1f) != 96470:
        raise RuntimeError(f"M1 feature row drift: {len(m1f)}")

    if m1f["order_id"].duplicated().any():
        raise RuntimeError("M1 feature order_id duplicates.")

    label = pd.read_csv(
        LABEL_SOURCE,
        usecols=[
            "order_id",
            LABEL_AVAILABLE_TIME,
        ],
    )

    if label["order_id"].duplicated().any():
        raise RuntimeError("Label-source order_id duplicates.")

    data = gold.merge(
        m1f[
            [
                "order_id",
                *M1_EXTRA,
                "m1_feature_vector_available",
                "availability_reason",
                "benchmark_population",
                "serving_model",
            ]
        ],
        on="order_id",
        how="left",
        validate="1:1",
    ).merge(
        label,
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

    if data[PRED_TIME].isna().any():
        raise RuntimeError("Missing prediction_time.")

    eligible = data[
        data["m1_feature_vector_available"]
        .fillna(False)
        .astype(bool)
    ].copy()

    if len(eligible) != 91254:
        raise RuntimeError(
            f"Expected 91,254 M1 eligible rows; found {len(eligible)}"
        )

    if eligible[LABEL_AVAILABLE_TIME].isna().any():
        n = int(eligible[LABEL_AVAILABLE_TIME].isna().sum())
        raise RuntimeError(
            f"M1 eligible rows missing label availability: {n}"
        )

    if eligible[M0].isna().any().any():
        cols = eligible[M0].isna().sum()
        raise RuntimeError(
            f"M0 missingness on paired population: "
            f"{cols[cols > 0].to_dict()}"
        )

    if eligible[M1_EXTRA].isna().any().any():
        cols = eligible[M1_EXTRA].isna().sum()
        raise RuntimeError(
            f"M1 missingness on paired population: "
            f"{cols[cols > 0].to_dict()}"
        )

    eligible["test_month"] = (
        eligible[PRED_TIME]
        .dt.to_period("M")
        .astype(str)
    )

    months = sorted(
        eligible["test_month"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(months) <= MIN_PRIOR_ELIGIBLE_MONTHS:
        raise RuntimeError("Insufficient eligible months.")

    audit = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "gold_rows": int(len(data)),
        "m1_eligible_rows": int(len(eligible)),
        "m0_fallback_rows": int(len(data) - len(eligible)),
        "eligible_months": months,
        "minimum_prior_eligible_months": MIN_PRIOR_ELIGIBLE_MONTHS,
        "m0_features": M0,
        "m1_extra_features": M1_EXTRA,
        "logistic_kwargs": LOGIT_KWARGS,
        "top_fractions": TOP_FRACTIONS,
        "label_availability_rule": (
            "training label available strictly before test-month cutoff"
        ),
        "frozen_temporal_contract_sha256": sha256(FROZEN_TEMPORAL),
        "frozen_temporal_contract_top_level_keys": sorted(
            frozen_temporal.keys()
        ),
    }

    (OUT / "02_population_and_environment.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fold_manifest = []
    fold_metrics = []
    recall_rows = []
    oof_rows = []

    test_months = months[MIN_PRIOR_ELIGIBLE_MONTHS:]

    for fold_id, test_month in enumerate(test_months, start=1):
        cutoff = pd.Timestamp(f"{test_month}-01")

        test_mask = eligible["test_month"].eq(test_month)

        train_mask = (
            (eligible[PRED_TIME] < cutoff)
            & (eligible[LABEL_AVAILABLE_TIME] < cutoff)
        )

        train = eligible.loc[train_mask].copy()
        test = eligible.loc[test_mask].copy()

        # Strict fold safety.
        purchase_leak = int(
            (train[PRED_TIME] >= cutoff).sum()
        )

        label_leak = int(
            (train[LABEL_AVAILABLE_TIME] >= cutoff).sum()
        )

        train_pos = int(train[TARGET].sum())
        test_pos = int(test[TARGET].sum())

        status = "READY"

        if len(train) == 0 or len(test) == 0:
            status = "EMPTY"

        if train[TARGET].nunique() < 2:
            status = "TRAIN_SINGLE_CLASS"

        if test[TARGET].nunique() < 2:
            status = "TEST_SINGLE_CLASS"

        if purchase_leak or label_leak:
            status = "PIT_VIOLATION"

        fold_manifest.append({
            "fold_id": fold_id,
            "test_month": test_month,
            "cutoff": cutoff,
            "n_train": len(train),
            "n_test": len(test),
            "train_positive": train_pos,
            "test_positive": test_pos,
            "train_prevalence": (
                train_pos / len(train)
                if len(train)
                else np.nan
            ),
            "test_prevalence": (
                test_pos / len(test)
                if len(test)
                else np.nan
            ),
            "max_train_purchase": (
                train[PRED_TIME].max()
                if len(train)
                else pd.NaT
            ),
            "max_train_label_available": (
                train[LABEL_AVAILABLE_TIME].max()
                if len(train)
                else pd.NaT
            ),
            "purchase_time_violations": purchase_leak,
            "label_availability_violations": label_leak,
            "status": status,
        })

        if status != "READY":
            continue

        X0_train = train[M0].astype(float)
        X1_train = train[M1].astype(float)

        X0_test = test[M0].astype(float)
        X1_test = test[M1].astype(float)

        y_train = train[TARGET].astype(int).to_numpy()
        y_test = test[TARGET].astype(int).to_numpy()

        model0 = make_model()
        model1 = make_model()

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model0.fit(X0_train, y_train)
            model1.fit(X1_train, y_train)

        p0 = model0.predict_proba(X0_test)[:, 1]
        p1 = model1.predict_proba(X1_test)[:, 1]

        metrics0 = {
            "average_precision": average_precision_score(
                y_test, p0
            ),
            "roc_auc": safe_auc(y_test, p0),
            "brier": brier_score_loss(y_test, p0),
            "log_loss": log_loss(
                y_test,
                p0,
                labels=[0, 1],
            ),
        }

        metrics1 = {
            "average_precision": average_precision_score(
                y_test, p1
            ),
            "roc_auc": safe_auc(y_test, p1),
            "brier": brier_score_loss(y_test, p1),
            "log_loss": log_loss(
                y_test,
                p1,
                labels=[0, 1],
            ),
        }

        fold_metrics.append({
            "fold_id": fold_id,
            "test_month": test_month,
            "n_train": len(train),
            "n_test": len(test),
            "test_positive": test_pos,
            "test_prevalence": test_pos / len(test),
            "m0_average_precision": metrics0["average_precision"],
            "m1_average_precision": metrics1["average_precision"],
            "delta_average_precision": (
                metrics1["average_precision"]
                - metrics0["average_precision"]
            ),
            "m0_roc_auc": metrics0["roc_auc"],
            "m1_roc_auc": metrics1["roc_auc"],
            "delta_roc_auc": metrics1["roc_auc"] - metrics0["roc_auc"],
            "m0_brier": metrics0["brier"],
            "m1_brier": metrics1["brier"],
            "delta_brier_m1_minus_m0": (
                metrics1["brier"] - metrics0["brier"]
            ),
            "m0_log_loss": metrics0["log_loss"],
            "m1_log_loss": metrics1["log_loss"],
            "delta_log_loss_m1_minus_m0": (
                metrics1["log_loss"] - metrics0["log_loss"]
            ),
        })

        for frac in TOP_FRACTIONS:
            r0, k0 = recall_at_fraction(
                y_test, p0, frac
            )
            r1, k1 = recall_at_fraction(
                y_test, p1, frac
            )

            if k0 != k1:
                raise RuntimeError(
                    "Paired Top-K cardinality mismatch."
                )

            recall_rows.append({
                "fold_id": fold_id,
                "test_month": test_month,
                "top_fraction": frac,
                "top_n": k0,
                "positives": test_pos,
                "m0_recall": r0,
                "m1_recall": r1,
                "delta_recall": r1 - r0,
            })

        for oid, yt, s0, s1 in zip(
            test["order_id"].astype(str),
            y_test,
            p0,
            p1,
        ):
            oof_rows.append({
                "order_id": oid,
                "fold_id": fold_id,
                "test_month": test_month,
                "y_true": int(yt),
                "m0_score": float(s0),
                "m1_score": float(s1),
            })

    fold_manifest_df = pd.DataFrame(fold_manifest)
    metrics = pd.DataFrame(fold_metrics)
    recall = pd.DataFrame(recall_rows)
    oof = pd.DataFrame(oof_rows)

    fold_manifest_df.to_csv(
        OUT / "03_temporal_fold_manifest.csv",
        index=False,
    )

    metrics.to_csv(
        OUT / "04_fold_metrics.csv",
        index=False,
    )

    recall.to_csv(
        OUT / "05_recall_at_top_fraction.csv",
        index=False,
    )

    oof.to_csv(
        OUT / "06_oof_predictions.csv",
        index=False,
    )

    if metrics.empty:
        raise RuntimeError("No valid paired folds.")

    if not (
        fold_manifest_df["purchase_time_violations"].eq(0).all()
        and fold_manifest_df["label_availability_violations"].eq(0).all()
    ):
        raise RuntimeError("Temporal leakage guardrail failure.")

    # Only READY folds should contribute metrics.
    ready_count = int(
        fold_manifest_df["status"].eq("READY").sum()
    )

    if ready_count != len(metrics):
        raise RuntimeError(
            "Fold manifest / metric count mismatch."
        )

    deltas = metrics[
        [
            "fold_id",
            "test_month",
            "delta_average_precision",
            "delta_roc_auc",
            "delta_brier_m1_minus_m0",
            "delta_log_loss_m1_minus_m0",
        ]
    ].copy()

    deltas.to_csv(
        OUT / "07_paired_metric_deltas.csv",
        index=False,
    )

    # Paired fold bootstrap of mean delta AP.
    rng = np.random.default_rng(BOOT_SEED)

    d = metrics["delta_average_precision"].to_numpy(
        dtype=float
    )

    boot = np.empty(N_BOOT, dtype=float)

    for i in range(N_BOOT):
        sample = rng.choice(
            d,
            size=len(d),
            replace=True,
        )
        boot[i] = sample.mean()

    ci_low, ci_high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    mean_delta = float(d.mean())
    median_delta = float(np.median(d))
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    losses = int((d < 0).sum())

    pooled_ap0 = average_precision_score(
        oof["y_true"],
        oof["m0_score"],
    )

    pooled_ap1 = average_precision_score(
        oof["y_true"],
        oof["m1_score"],
    )

    if mean_delta > 0 and ci_low > 0:
        classification = (
            "LOGISTIC_M1_INCREMENTAL_UTILITY_SUPPORTED"
        )
    elif mean_delta > 0:
        classification = (
            "LOGISTIC_M1_FAVORABLE_NOT_DECISIVE"
        )
    else:
        classification = (
            "LOGISTIC_M1_INCREMENTAL_UTILITY_NOT_SUPPORTED"
        )

    bootstrap_summary = {
        "n_valid_folds": int(len(d)),
        "bootstrap_iterations": N_BOOT,
        "seed": BOOT_SEED,
        "mean_delta_average_precision": mean_delta,
        "median_delta_average_precision": median_delta,
        "ci95_mean_delta_average_precision": [
            float(ci_low),
            float(ci_high),
        ],
        "m1_fold_wins": wins,
        "ties": ties,
        "m0_fold_wins": losses,
        "pooled_oof_m0_average_precision": float(
            pooled_ap0
        ),
        "pooled_oof_m1_average_precision": float(
            pooled_ap1
        ),
        "pooled_oof_delta_average_precision": float(
            pooled_ap1 - pooled_ap0
        ),
        "classification": classification,
    }

    (OUT / "08_bootstrap_ap_delta.json").write_text(
        json.dumps(
            bootstrap_summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    agg_rows = []

    for metric in [
        "average_precision",
        "roc_auc",
        "brier",
        "log_loss",
    ]:
        m0_col = f"m0_{metric}"
        m1_col = f"m1_{metric}"

        agg_rows.append({
            "metric": metric,
            "m0_mean": float(metrics[m0_col].mean()),
            "m1_mean": float(metrics[m1_col].mean()),
            "m0_median": float(metrics[m0_col].median()),
            "m1_median": float(metrics[m1_col].median()),
            "mean_delta_m1_minus_m0": float(
                (metrics[m1_col] - metrics[m0_col]).mean()
            ),
            "median_delta_m1_minus_m0": float(
                (metrics[m1_col] - metrics[m0_col]).median()
            ),
        })

    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(
        OUT / "09_aggregate_metrics.csv",
        index=False,
    )

    benchmark_contract = {
        "contract": "M0_VS_M1_PAIRED_LOGISTIC_BENCHMARK_V1",
        "status": classification,
        "prediction_time": "order_purchase_timestamp",
        "population": "M1_ELIGIBLE",
        "population_rows": int(len(eligible)),
        "m0_fallback_rows_outside_paired_estimand": int(
            len(data) - len(eligible)
        ),
        "minimum_prior_eligible_months": MIN_PRIOR_ELIGIBLE_MONTHS,
        "valid_temporal_folds": int(len(metrics)),
        "temporal_rule": {
            "test": "one purchase month",
            "training_purchase_time": "strictly before test-month cutoff",
            "training_label_available_time": (
                "order_delivered_customer_date strictly before test-month cutoff"
            ),
            "future_label_use": "FORBIDDEN",
        },
        "estimator": {
            "pipeline": [
                "StandardScaler",
                "LogisticRegression",
            ],
            "logistic_kwargs": LOGIT_KWARGS,
        },
        "m0_features": M0,
        "m1_incremental_features": M1_EXTRA,
        "primary_metric": "average_precision",
        "secondary_metrics": [
            "roc_auc",
            "brier",
            "log_loss",
            "recall_at_top_fraction",
        ],
        "top_fractions": TOP_FRACTIONS,
        "paired_bootstrap": bootstrap_summary,
        "decision": {
            "classification": classification,
            "m1_released": False,
            "algorithm_benchmark_authorized": True,
            "threshold_selected": False,
        },
        "provenance": {
            "m1_feature_contract_sha256": sha256(
                M1_CONTRACT
            ),
            "frozen_temporal_contract_sha256": sha256(
                FROZEN_TEMPORAL
            ),
        },
    }

    BENCHMARK_CONTRACT.write_text(
        json.dumps(
            benchmark_contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    with (OUT / "10_paired_benchmark_decision.txt").open(
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write("=" * 96 + "\n")
        fh.write(
            "DELIVERY RISK — BLOCK 10 PAIRED M0 VS M1 LOGISTIC BENCHMARK\n"
        )
        fh.write("=" * 96 + "\n\n")
        fh.write(f"PAIRED_POPULATION_ROWS       : {len(eligible)}\n")
        fh.write(
            f"M0_FALLBACK_OUTSIDE_ESTIMAND : {len(data) - len(eligible)}\n"
        )
        fh.write(f"VALID_TEMPORAL_FOLDS         : {len(metrics)}\n")
        fh.write("PURCHASE_TIME_VIOLATIONS     : 0\n")
        fh.write("LABEL_AVAILABILITY_VIOLATIONS: 0\n")
        fh.write(
            f"M0_MEAN_AP                   : "
            f"{metrics['m0_average_precision'].mean():.12f}\n"
        )
        fh.write(
            f"M1_MEAN_AP                   : "
            f"{metrics['m1_average_precision'].mean():.12f}\n"
        )
        fh.write(
            f"MEAN_DELTA_AP                : {mean_delta:.12f}\n"
        )
        fh.write(
            f"MEDIAN_DELTA_AP              : {median_delta:.12f}\n"
        )
        fh.write(
            f"BOOTSTRAP_95CI_DELTA_AP      : "
            f"[{ci_low:.12f}, {ci_high:.12f}]\n"
        )
        fh.write(
            f"M1_FOLD_WINS                 : {wins}\n"
        )
        fh.write(
            f"M0_FOLD_WINS                 : {losses}\n"
        )
        fh.write(
            f"TIES                         : {ties}\n"
        )
        fh.write(
            f"POOLED_OOF_M0_AP             : {pooled_ap0:.12f}\n"
        )
        fh.write(
            f"POOLED_OOF_M1_AP             : {pooled_ap1:.12f}\n"
        )
        fh.write(
            f"POOLED_OOF_DELTA_AP          : "
            f"{pooled_ap1 - pooled_ap0:.12f}\n"
        )
        fh.write("THRESHOLD_SELECTED           : NO\n")
        fh.write("M1_RELEASED                  : NO\n")
        fh.write("ALGORITHM_BENCHMARK_AUTHORIZED: YES\n\n")
        fh.write(
            f"STATUS = {classification}\n"
        )

    DOC.write_text(
        f"""# M0 vs M1 Paired Logistic Benchmark V1

## Purpose

This is the first production late-risk feature-utility experiment.

M0 and M1 are evaluated on the same {len(eligible):,} M1-eligible orders.

The {len(data) - len(eligible):,} remaining frozen supervised orders are not
discarded from the operational population; they remain governed by the M0
fallback.

## Models

Both arms use the same pipeline:

- StandardScaler
- LogisticRegression with L2 regularization
- C = 1.0
- solver = lbfgs
- max_iter = 2000
- random_state = 42
- sklearn API is version-aware: `penalty="l2"` before 1.8; `l1_ratio=0.0` from 1.8 onward

M0 uses ORDER_CORE_V1.

M1 uses ORDER_CORE_V1 plus:

- expected_freight_oot
- freight_burden
- freight_expected_ratio
- freight_log_ratio

## Temporal protocol

Each test fold is one purchase month.

Training rows must satisfy both:

1. purchase time strictly before the test-month cutoff;
2. label availability time strictly before the same cutoff.

The label-availability timestamp is the observed customer delivery timestamp.
This is a conservative rule: an order is not allowed into training merely
because it was purchased earlier.

Minimum prior M1-eligible months before the first test fold:
{MIN_PRIOR_ELIGIBLE_MONTHS}.

Valid folds: {len(metrics)}.

## Primary result

- M0 mean AP: {metrics["m0_average_precision"].mean():.12f}
- M1 mean AP: {metrics["m1_average_precision"].mean():.12f}
- Mean paired delta AP: {mean_delta:.12f}
- Median paired delta AP: {median_delta:.12f}
- Bootstrap 95% interval: [{ci_low:.12f}, {ci_high:.12f}]
- M1 fold wins: {wins}
- M0 fold wins: {losses}
- Ties: {ties}

Classification:

`{classification}`

## Governance

Average Precision is the primary metric.

No probability threshold is selected in this gate.

M1 is not production-released by this result alone.

The subsequent algorithm benchmark is authorized and must preserve temporal
governance and paired feature-set comparisons.
""",
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("PAIRED M0 VS M1 LOGISTIC BENCHMARK")
    print("=" * 96)
    print(metrics.to_string(index=False))
    print()
    print(json.dumps(
        bootstrap_summary,
        indent=2,
        ensure_ascii=False,
    ))
    print()
    print(
        (OUT / "10_paired_benchmark_decision.txt")
        .read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
