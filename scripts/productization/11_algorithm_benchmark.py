#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import math
import platform
import time
import traceback
import warnings

import duckdb
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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

OUT = REPO / "reports/productization/algorithm_benchmark"
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

BLOCK10_CONTRACT = (
    REPO
    / "configs/m0_vs_m1_paired_logistic_benchmark_v1.json"
)
BLOCK10_FOLDS = (
    REPO
    / "reports/productization/m0_vs_m1_paired_logistic/"
    / "03_temporal_fold_manifest.csv"
)
BLOCK10_OOF = (
    REPO
    / "reports/productization/m0_vs_m1_paired_logistic/"
    / "06_oof_predictions.csv"
)

CONTRACT = REPO / "configs/algorithm_benchmark_v1.json"
DOC = REPO / "docs/architecture/ALGORITHM_BENCHMARK_V1.md"

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

TARGET = "late_delivery_calendar_day"
PRED_TIME = "prediction_time"
LABEL_AVAILABLE_TIME = "order_delivered_customer_date"

TOP_FRACTIONS = [0.05, 0.10, 0.20]
N_BOOT = 20000
BOOT_SEED = 20260831


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

    k = max(1, int(math.ceil(fraction * len(y_true))))
    order = np.argsort(-score, kind="mergesort")
    chosen = order[:k]

    return float(y_true[chosen].sum() / positives), k


def logistic_factory(block10):
    kwargs = dict(
        block10["estimator"]["logistic_kwargs"]
    )

    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(**kwargs)),
    ])


def model_registry(block10):
    registry = {
        "LOGISTIC": {
            "available": True,
            "required": True,
            "family": "linear",
            "factory": lambda: logistic_factory(block10),
            "params": {
                "pipeline": "StandardScaler + LogisticRegression",
                **block10["estimator"]["logistic_kwargs"],
            },
        },
        "RANDOM_FOREST": {
            "available": True,
            "required": True,
            "family": "bagging_tree",
            "factory": lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight=None,
                random_state=42,
                n_jobs=2,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": None,
                "random_state": 42,
                "n_jobs": 2,
            },
        },
        "HIST_GRADIENT_BOOSTING": {
            "available": True,
            "required": True,
            "family": "boosted_tree",
            "factory": lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=42,
            ),
            "params": {
                "learning_rate": 0.05,
                "max_iter": 150,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": 42,
            },
        },
    }

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        registry["XGBOOST"] = {
            "available": True,
            "required": False,
            "family": "boosted_tree_external",
            "factory": lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1.0,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=2,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 2,
            },
        }

    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        registry["LIGHTGBM"] = {
            "available": True,
            "required": False,
            "family": "boosted_tree_external",
            "factory": lambda: LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            ),
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": 2,
                "verbosity": -1,
            },
        }

    if importlib.util.find_spec("catboost") is not None:
        from catboost import CatBoostClassifier

        registry["CATBOOST"] = {
            "available": True,
            "required": False,
            "family": "boosted_tree_external",
            "factory": lambda: CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=False,
                random_seed=42,
                thread_count=2,
                allow_writing_files=False,
            ),
            "params": {
                "iterations": 200,
                "depth": 6,
                "learning_rate": 0.05,
                "loss_function": "Logloss",
                "verbose": False,
                "random_seed": 42,
                "thread_count": 2,
                "allow_writing_files": False,
            },
        }

    return registry


def main():
    block10 = json.loads(
        BLOCK10_CONTRACT.read_text(encoding="utf-8")
    )

    if block10["population_rows"] != 91254:
        raise RuntimeError("Block 10 paired population drift.")

    if block10["valid_temporal_folds"] != 14:
        raise RuntimeError("Block 10 fold count drift.")

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

    label = pd.read_csv(
        LABEL_SOURCE,
        usecols=[
            "order_id",
            LABEL_AVAILABLE_TIME,
        ],
    )

    folds_ref = pd.read_csv(BLOCK10_FOLDS)
    oof_ref = pd.read_csv(BLOCK10_OOF)

    if gold["order_id"].duplicated().any():
        raise RuntimeError("Gold duplicate order_id.")

    if m1f["order_id"].duplicated().any():
        raise RuntimeError("M1 feature duplicate order_id.")

    if label["order_id"].duplicated().any():
        raise RuntimeError("Label-source duplicate order_id.")

    data = gold.merge(
        m1f[
            [
                "order_id",
                *M1_EXTRA,
                "m1_feature_vector_available",
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

    eligible = data[
        data["m1_feature_vector_available"]
        .fillna(False)
        .astype(bool)
    ].copy()

    if len(eligible) != 91254:
        raise RuntimeError(
            f"Paired population drift: {len(eligible)}"
        )

    if eligible[M0 + M1_EXTRA].isna().any().any():
        raise RuntimeError(
            "Missing model feature on paired population."
        )

    eligible["test_month"] = (
        eligible[PRED_TIME]
        .dt.to_period("M")
        .astype(str)
    )

    ready_folds = folds_ref[
        folds_ref["status"].eq("READY")
    ].copy()

    if len(ready_folds) != 14:
        raise RuntimeError(
            f"Expected 14 READY folds; found {len(ready_folds)}"
        )

    reference_oof_ids = set(
        oof_ref["order_id"].astype(str)
    )

    registry = model_registry(block10)

    model_rows = []

    for name, spec in registry.items():
        model_rows.append({
            "algorithm": name,
            "family": spec["family"],
            "available": spec["available"],
            "required": spec["required"],
            "params_json": json.dumps(
                spec["params"],
                sort_keys=True,
            ),
        })

    pd.DataFrame(model_rows).to_csv(
        OUT / "03_model_registry.csv",
        index=False,
    )

    fold_parity = []
    metrics_rows = []
    recall_rows = []
    runtime_errors = []

    # Wide OOF table initialized from the exact Block-10 OOF order set.
    oof = (
        oof_ref[
            [
                "order_id",
                "fold_id",
                "test_month",
                "y_true",
            ]
        ]
        .copy()
        .sort_values(["fold_id", "order_id"])
        .reset_index(drop=True)
    )

    oof_index = {
        oid: idx
        for idx, oid in enumerate(
            oof["order_id"].astype(str)
        )
    }

    active_algorithms = list(registry.keys())

    for fold in ready_folds.itertuples(index=False):
        fold_id = int(fold.fold_id)
        test_month = str(fold.test_month)
        cutoff = pd.Timestamp(f"{test_month}-01")

        train_mask = (
            (eligible[PRED_TIME] < cutoff)
            & (eligible[LABEL_AVAILABLE_TIME] < cutoff)
        )

        test_mask = eligible["test_month"].eq(test_month)

        train = eligible.loc[train_mask].copy()
        test = eligible.loc[test_mask].copy()

        test_ids = set(test["order_id"].astype(str))

        ref_ids = set(
            oof_ref.loc[
                oof_ref["test_month"].astype(str).eq(test_month),
                "order_id",
            ].astype(str)
        )

        parity_ok = (
            len(train) == int(fold.n_train)
            and len(test) == int(fold.n_test)
            and test_ids == ref_ids
        )

        fold_parity.append({
            "fold_id": fold_id,
            "test_month": test_month,
            "recomputed_n_train": len(train),
            "reference_n_train": int(fold.n_train),
            "recomputed_n_test": len(test),
            "reference_n_test": int(fold.n_test),
            "test_id_symmetric_difference": len(
                test_ids.symmetric_difference(ref_ids)
            ),
            "purchase_time_violations": int(
                (train[PRED_TIME] >= cutoff).sum()
            ),
            "label_availability_violations": int(
                (train[LABEL_AVAILABLE_TIME] >= cutoff).sum()
            ),
            "status": "PASS" if parity_ok else "FAIL",
        })

        if not parity_ok:
            raise RuntimeError(
                f"Fold {fold_id} no longer matches Block 10."
            )

        y_train = train[TARGET].astype(int).to_numpy()
        y_test = test[TARGET].astype(int).to_numpy()

        feature_sets = {
            "M0": M0,
            "M1": M0 + M1_EXTRA,
        }

        for algorithm in list(active_algorithms):
            spec = registry[algorithm]

            for arm, features in feature_sets.items():
                X_train = train[features].astype(float)
                X_test = test[features].astype(float)

                model = spec["factory"]()

                fit_start = time.perf_counter()

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("default")
                        model.fit(X_train, y_train)

                    fit_seconds = time.perf_counter() - fit_start

                    pred_start = time.perf_counter()
                    p = model.predict_proba(X_test)[:, 1]
                    predict_seconds = (
                        time.perf_counter() - pred_start
                    )
                except Exception as exc:
                    runtime_errors.append({
                        "algorithm": algorithm,
                        "arm": arm,
                        "fold_id": fold_id,
                        "test_month": test_month,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })

                    if spec["required"]:
                        raise

                    # Optional algorithm is dropped completely if runtime fails.
                    active_algorithms.remove(algorithm)
                    break

                ap = average_precision_score(y_test, p)
                auc = safe_auc(y_test, p)
                brier = brier_score_loss(y_test, p)
                ll = log_loss(y_test, p, labels=[0, 1])

                metrics_rows.append({
                    "algorithm": algorithm,
                    "arm": arm,
                    "fold_id": fold_id,
                    "test_month": test_month,
                    "n_train": len(train),
                    "n_test": len(test),
                    "test_positive": int(y_test.sum()),
                    "test_prevalence": float(y_test.mean()),
                    "average_precision": float(ap),
                    "roc_auc": float(auc),
                    "brier": float(brier),
                    "log_loss": float(ll),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                })

                for frac in TOP_FRACTIONS:
                    recall, top_n = recall_at_fraction(
                        y_test,
                        p,
                        frac,
                    )

                    recall_rows.append({
                        "algorithm": algorithm,
                        "arm": arm,
                        "fold_id": fold_id,
                        "test_month": test_month,
                        "top_fraction": frac,
                        "top_n": top_n,
                        "positives": int(y_test.sum()),
                        "recall": recall,
                    })

                score_col = f"{algorithm}__{arm}"

                if score_col not in oof.columns:
                    oof[score_col] = np.nan

                for oid, score in zip(
                    test["order_id"].astype(str),
                    p,
                ):
                    oof.at[
                        oof_index[oid],
                        score_col,
                    ] = float(score)

    # Remove all rows from optional algorithms that failed at runtime.
    failed_optional = {
        r["algorithm"]
        for r in runtime_errors
        if not registry[r["algorithm"]]["required"]
    }

    if failed_optional:
        metrics_rows = [
            r for r in metrics_rows
            if r["algorithm"] not in failed_optional
        ]
        recall_rows = [
            r for r in recall_rows
            if r["algorithm"] not in failed_optional
        ]
        drop_cols = [
            c for c in oof.columns
            if any(
                c.startswith(f"{alg}__")
                for alg in failed_optional
            )
        ]
        oof = oof.drop(columns=drop_cols)

    fold_parity_df = pd.DataFrame(fold_parity)
    metrics = pd.DataFrame(metrics_rows)
    recall = pd.DataFrame(recall_rows)

    fold_parity_df.to_csv(
        OUT / "02_fold_contract_parity.csv",
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

    if runtime_errors:
        pd.DataFrame(runtime_errors).to_csv(
            OUT / "11_optional_runtime_errors.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "algorithm",
                "arm",
                "fold_id",
                "test_month",
                "error_type",
                "error",
                "traceback",
            ]
        ).to_csv(
            OUT / "11_optional_runtime_errors.csv",
            index=False,
        )

    if not fold_parity_df["status"].eq("PASS").all():
        raise RuntimeError("Fold contract parity failure.")

    if not fold_parity_df[
        "purchase_time_violations"
    ].eq(0).all():
        raise RuntimeError("Purchase-time leakage.")

    if not fold_parity_df[
        "label_availability_violations"
    ].eq(0).all():
        raise RuntimeError("Label-availability leakage.")

    algorithms = sorted(
        metrics["algorithm"].unique().tolist()
    )

    # Every surviving algorithm must have 14 folds for both arms.
    completeness = (
        metrics.groupby(["algorithm", "arm"])
        .size()
        .rename("folds")
        .reset_index()
    )

    bad = completeness[
        completeness["folds"].ne(14)
    ]

    if len(bad):
        print(bad.to_string(index=False))
        raise RuntimeError("Incomplete algorithm benchmark.")

    # Exact OOF membership must remain Block-10 membership.
    if set(oof["order_id"].astype(str)) != reference_oof_ids:
        raise RuntimeError("OOF order-id population drift.")

    score_cols = [
        c for c in oof.columns
        if "__" in c
    ]

    if oof[score_cols].isna().any().any():
        missing = oof[score_cols].isna().sum()
        print(missing[missing > 0])
        raise RuntimeError("Missing OOF predictions.")

    # Same-algorithm M1-M0 paired deltas.
    delta_rows = []

    for algorithm in algorithms:
        m0 = metrics[
            (metrics["algorithm"] == algorithm)
            & (metrics["arm"] == "M0")
        ].sort_values("fold_id")

        m1 = metrics[
            (metrics["algorithm"] == algorithm)
            & (metrics["arm"] == "M1")
        ].sort_values("fold_id")

        if not m0["fold_id"].tolist() == m1["fold_id"].tolist():
            raise RuntimeError(
                f"Arm fold mismatch for {algorithm}"
            )

        for a, b in zip(
            m0.itertuples(index=False),
            m1.itertuples(index=False),
        ):
            delta_rows.append({
                "algorithm": algorithm,
                "fold_id": int(a.fold_id),
                "test_month": a.test_month,
                "delta_average_precision": (
                    b.average_precision - a.average_precision
                ),
                "delta_roc_auc": b.roc_auc - a.roc_auc,
                "delta_brier_m1_minus_m0": b.brier - a.brier,
                "delta_log_loss_m1_minus_m0": (
                    b.log_loss - a.log_loss
                ),
            })

    deltas = pd.DataFrame(delta_rows)

    deltas.to_csv(
        OUT / "07_same_algorithm_m1_minus_m0.csv",
        index=False,
    )

    # Paired fold bootstrap per algorithm.
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap = {}

    for algorithm in algorithms:
        d = deltas.loc[
            deltas["algorithm"].eq(algorithm),
            "delta_average_precision",
        ].to_numpy(dtype=float)

        boot = np.empty(N_BOOT, dtype=float)

        for i in range(N_BOOT):
            boot[i] = rng.choice(
                d,
                size=len(d),
                replace=True,
            ).mean()

        lo, hi = np.quantile(
            boot,
            [0.025, 0.975],
        )

        mean_delta = float(d.mean())

        if mean_delta > 0 and lo > 0:
            feature_status = "M1_SUPPORTED"
        elif mean_delta > 0:
            feature_status = "M1_FAVORABLE_NOT_DECISIVE"
        else:
            feature_status = "M1_NOT_SUPPORTED"

        bootstrap[algorithm] = {
            "n_folds": len(d),
            "mean_delta_ap": mean_delta,
            "median_delta_ap": float(np.median(d)),
            "ci95_mean_delta_ap": [
                float(lo),
                float(hi),
            ],
            "m1_wins": int((d > 0).sum()),
            "ties": int((d == 0).sum()),
            "m0_wins": int((d < 0).sum()),
            "feature_status": feature_status,
        }

    (OUT / "08_same_algorithm_bootstrap.json").write_text(
        json.dumps(
            bootstrap,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # Arm ranking.
    ranking_rows = []

    for algorithm in algorithms:
        for arm in ["M0", "M1"]:
            g = metrics[
                (metrics["algorithm"] == algorithm)
                & (metrics["arm"] == arm)
            ]

            score_col = f"{algorithm}__{arm}"

            pooled_ap = average_precision_score(
                oof["y_true"],
                oof[score_col],
            )

            pooled_auc = safe_auc(
                oof["y_true"],
                oof[score_col],
            )

            ranking_rows.append({
                "algorithm": algorithm,
                "arm": arm,
                "mean_average_precision": float(
                    g["average_precision"].mean()
                ),
                "median_average_precision": float(
                    g["average_precision"].median()
                ),
                "pooled_oof_average_precision": float(
                    pooled_ap
                ),
                "mean_roc_auc": float(
                    g["roc_auc"].mean()
                ),
                "pooled_oof_roc_auc": float(
                    pooled_auc
                ),
                "mean_brier": float(
                    g["brier"].mean()
                ),
                "mean_log_loss": float(
                    g["log_loss"].mean()
                ),
                "mean_fit_seconds": float(
                    g["fit_seconds"].mean()
                ),
                "mean_predict_seconds": float(
                    g["predict_seconds"].mean()
                ),
                "same_algorithm_m1_status": (
                    bootstrap[algorithm][
                        "feature_status"
                    ]
                ),
            })

    ranking = pd.DataFrame(ranking_rows)

    ranking = ranking.sort_values(
        [
            "mean_average_precision",
            "median_average_precision",
            "pooled_oof_average_precision",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    logistic_m0 = ranking[
        (ranking["algorithm"] == "LOGISTIC")
        & (ranking["arm"] == "M0")
    ]["mean_average_precision"].iloc[0]

    ranking["delta_mean_ap_vs_logistic_m0"] = (
        ranking["mean_average_precision"]
        - logistic_m0
    )

    ranking.to_csv(
        OUT / "09_algorithm_arm_ranking.csv",
        index=False,
    )

    winner = ranking.iloc[0].to_dict()

    any_m1_supported = any(
        v["feature_status"] == "M1_SUPPORTED"
        for v in bootstrap.values()
    )

    any_m1_favorable = any(
        v["feature_status"]
        in {
            "M1_SUPPORTED",
            "M1_FAVORABLE_NOT_DECISIVE",
        }
        for v in bootstrap.values()
    )

    if any_m1_supported:
        m1_evidence = (
            "NONLINEAR_OR_LINEAR_M1_INCREMENTAL_UTILITY_SUPPORTED"
        )
    elif any_m1_favorable:
        m1_evidence = (
            "M1_INCREMENTAL_UTILITY_MIXED_NOT_DECISIVE"
        )
    else:
        m1_evidence = (
            "M1_INCREMENTAL_UTILITY_NOT_SUPPORTED_ACROSS_BENCHMARK"
        )

    decision_status = "ALGORITHM_BENCHMARK_COMPLETE_CANDIDATE_IDENTIFIED"

    contract = {
        "contract": "ALGORITHM_BENCHMARK_V1",
        "status": decision_status,
        "prediction_time": "order_purchase_timestamp",
        "paired_population_rows": 91254,
        "m0_fallback_rows": 5216,
        "folds": 14,
        "selection_metric": "mean_average_precision_across_temporal_folds",
        "tiebreakers": [
            "median_average_precision",
            "pooled_oof_average_precision",
        ],
        "algorithms_benchmarked": algorithms,
        "optional_runtime_failures": sorted(
            failed_optional
        ),
        "winner": {
            "algorithm": winner["algorithm"],
            "arm": winner["arm"],
            "mean_average_precision": float(
                winner["mean_average_precision"]
            ),
            "median_average_precision": float(
                winner["median_average_precision"]
            ),
            "pooled_oof_average_precision": float(
                winner["pooled_oof_average_precision"]
            ),
        },
        "m1_feature_evidence": {
            "classification": m1_evidence,
            "same_algorithm_bootstrap": bootstrap,
        },
        "governance": {
            "same_block10_population": True,
            "same_block10_folds": True,
            "purchase_time_leakage": 0,
            "label_availability_leakage": 0,
            "hyperparameter_tuning": False,
            "threshold_selected": False,
            "m1_released": False,
            "production_model_released": False,
            "next_gate": (
                "candidate robustness + Recall@Top-K operational review"
            ),
        },
    }

    CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    with (OUT / "10_benchmark_decision.txt").open(
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write("=" * 100 + "\n")
        fh.write(
            "DELIVERY RISK — BLOCK 11 GOVERNED ALGORITHM BENCHMARK\n"
        )
        fh.write("=" * 100 + "\n\n")
        fh.write("PAIRED_POPULATION_ROWS       : 91254\n")
        fh.write("M0_FALLBACK_ROWS             : 5216\n")
        fh.write("VALID_TEMPORAL_FOLDS         : 14\n")
        fh.write("PURCHASE_TIME_LEAKAGE        : 0\n")
        fh.write("LABEL_AVAILABILITY_LEAKAGE   : 0\n")
        fh.write(
            f"ALGORITHMS_BENCHMARKED       : "
            f"{'|'.join(algorithms)}\n"
        )
        fh.write(
            f"OPTIONAL_RUNTIME_FAILURES     : "
            f"{'|'.join(sorted(failed_optional)) or 'NONE'}\n"
        )
        fh.write(
            f"WINNER_ALGORITHM             : "
            f"{winner['algorithm']}\n"
        )
        fh.write(
            f"WINNER_ARM                   : "
            f"{winner['arm']}\n"
        )
        fh.write(
            f"WINNER_MEAN_AP               : "
            f"{winner['mean_average_precision']:.12f}\n"
        )
        fh.write(
            f"WINNER_MEDIAN_AP             : "
            f"{winner['median_average_precision']:.12f}\n"
        )
        fh.write(
            f"WINNER_POOLED_OOF_AP         : "
            f"{winner['pooled_oof_average_precision']:.12f}\n"
        )
        fh.write(
            f"M1_FEATURE_EVIDENCE          : "
            f"{m1_evidence}\n"
        )
        fh.write("HYPERPARAMETER_TUNING        : NO\n")
        fh.write("THRESHOLD_SELECTED           : NO\n")
        fh.write("M1_RELEASED                  : NO\n")
        fh.write("PRODUCTION_MODEL_RELEASED    : NO\n\n")
        fh.write(
            "STATUS = "
            "ALGORITHM_BENCHMARK_COMPLETE_CANDIDATE_IDENTIFIED\n"
        )

    DOC.write_text(
        f"""# Governed Algorithm Benchmark V1

## Purpose

The logistic paired gate found no support for incremental M1 utility. This
benchmark tests whether nonlinear classifiers can extract useful signal from the
same Shipping Intelligence features.

## Frozen population and folds

- Paired M1-eligible population: 91,254 orders
- Operational M0 fallback outside estimand: 5,216 orders
- Temporal folds: 14
- Purchase-time leakage: 0
- Label-availability leakage: 0

The exact Block-10 fold counts and test order IDs are reproduced before any
model is fit.

## Algorithms

Benchmarked algorithms:

`{", ".join(algorithms)}`

Optional runtime failures:

`{", ".join(sorted(failed_optional)) if failed_optional else "NONE"}`

Each algorithm is evaluated with M0 and M1 on identical rows.

No hyperparameter optimization is performed in this gate.

## Selection

Primary ranking criterion:

`mean Average Precision across temporal folds`

Tie-breakers:

1. median temporal-fold Average Precision;
2. pooled OOF Average Precision.

Benchmark candidate:

- algorithm: `{winner["algorithm"]}`
- feature arm: `{winner["arm"]}`
- mean AP: `{winner["mean_average_precision"]:.12f}`
- median AP: `{winner["median_average_precision"]:.12f}`
- pooled OOF AP: `{winner["pooled_oof_average_precision"]:.12f}`

M1 evidence classification:

`{m1_evidence}`

## Governance

A benchmark winner is not yet a production release.

The next gate must review candidate robustness and operational Recall@Top-K
before production selection, thresholding, calibration, explainability or
deployment.
""",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("ALGORITHM ARM RANKING")
    print("=" * 100)
    print(ranking.to_string(index=False))

    print()
    print("=" * 100)
    print("SAME-ALGORITHM M1 VS M0")
    print("=" * 100)
    print(json.dumps(
        bootstrap,
        indent=2,
        ensure_ascii=False,
    ))

    print()
    print(
        (OUT / "10_benchmark_decision.txt")
        .read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    main()
