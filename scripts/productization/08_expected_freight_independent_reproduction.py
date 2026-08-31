#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Independent reproduction of Expected Freight F2.

Important:
- This script DOES NOT import or execute the historical producer.
- It implements only the frozen runtime contract from
  expected_freight_runtime_manifest_v1.json.
- It performs expanding-month OOT F2 reproduction and row-level parity.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import sys

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


REPO = Path(__file__).resolve().parents[2]

ROUTE = REPO / "artifacts/spatiotemporal_logistics/01_ROUTE_SELLER_ORDER.csv"
HIST = REPO / "artifacts/spatiotemporal_logistics/06_EXPECTED_FREIGHT_OOT.csv"
DB = REPO / "data/bronze/olist_medallion.duckdb"
RUNTIME_MANIFEST = REPO / "configs/expected_freight_runtime_manifest_v1.json"

OUT = REPO / "reports/productization/m1_expected_freight_runtime_reproduction"
CONTRACT = REPO / "configs/expected_freight_production_contract_v1.json"
DOC = REPO / "docs/architecture/EXPECTED_FREIGHT_RUNTIME_REPRODUCTION_V1.md"

OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "log_distance",
    "log_weight",
    "log_volume",
    "log_price",
    "seller_item_count",
]

MIN_TRAIN_MONTHS = 3
ATOL = 1e-10
RTOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")


def main() -> None:
    for path in [ROUTE, HIST, DB, RUNTIME_MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))

    if manifest["status"] != "READY_FOR_INDEPENDENT_RUNTIME_REPRODUCTION_V3":
        raise RuntimeError("Runtime manifest is not reproduction-ready.")

    if manifest["feature_vector"] != FEATURES:
        raise RuntimeError("Feature-vector drift.")

    if manifest["minimum_training_months"] != MIN_TRAIN_MONTHS:
        raise RuntimeError("MIN_TRAIN_MONTHS drift.")

    if manifest["model_constructor"] != (
        "HistGradientBoostingRegressor(max_iter=100, random_state=42)"
    ):
        raise RuntimeError("Model-constructor drift.")

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "duckdb": duckdb.__version__,
        "route_sha256": sha256_file(ROUTE),
        "historical_expected_freight_sha256": sha256_file(HIST),
        "runtime_manifest_sha256": sha256_file(RUNTIME_MANIFEST),
    }

    (OUT / "01_environment.json").write_text(
        json.dumps(env, indent=2) + "\n",
        encoding="utf-8",
    )

    rso = pd.read_csv(ROUTE)

    required_route = [
        "order_id",
        "year_month",
        "seller_freight",
        "great_circle_distance_km",
        "seller_weight_g",
        "seller_volume_proxy_cm3",
        "seller_price",
        "seller_item_count",
    ]

    require_columns(rso, required_route, "ROUTE")

    # Exact independent preprocessing from frozen manifest.
    rso["log_freight"] = np.log1p(
        rso["seller_freight"].where(rso["seller_freight"] >= 0)
    )
    rso["log_distance"] = np.log1p(
        rso["great_circle_distance_km"].where(
            rso["great_circle_distance_km"] >= 0
        )
    )
    rso["log_weight"] = np.log1p(
        rso["seller_weight_g"].where(rso["seller_weight_g"] >= 0)
    )
    rso["log_volume"] = np.log1p(
        rso["seller_volume_proxy_cm3"].where(
            rso["seller_volume_proxy_cm3"] >= 0
        )
    )
    rso["log_price"] = np.log1p(
        rso["seller_price"].where(rso["seller_price"] >= 0)
    )

    months = sorted(
        rso["year_month"].dropna().astype(str).unique().tolist()
    )

    if len(months) <= MIN_TRAIN_MONTHS:
        raise RuntimeError("Insufficient month history.")

    warmup_months = months[:MIN_TRAIN_MONTHS]

    rso["pred_log_f2_reproduced"] = np.nan

    fold_rows = []

    for i in range(MIN_TRAIN_MONTHS, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = (
            rso["year_month"].astype(str).isin(train_months)
            & rso["log_freight"].notna()
        )
        test_mask = rso["year_month"].astype(str).eq(test_month)

        if not train_mask.any() or not test_mask.any():
            continue

        X_train = rso.loc[train_mask, FEATURES]
        y_train = rso.loc[train_mask, "log_freight"]
        X_test = rso.loc[test_mask, FEATURES]

        f2 = HistGradientBoostingRegressor(
            max_iter=100,
            random_state=42,
        )

        f2.fit(X_train, y_train)

        pred = f2.predict(X_test)

        rso.loc[
            test_mask,
            "pred_log_f2_reproduced",
        ] = pred

        fold_rows.append({
            "test_month": test_month,
            "train_start": train_months[0],
            "train_end": train_months[-1],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_predicted": int(np.isfinite(pred).sum()),
            "model_n_iter": int(getattr(f2, "n_iter_", -1)),
        })

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(
        OUT / "02_seller_f2_fold_summary.csv",
        index=False,
    )

    rso["pred_freight_f2_reproduced"] = np.maximum(
        0.0,
        np.expm1(rso["pred_log_f2_reproduced"]),
    )

    rso["pred_freight_oot_reproduced"] = (
        rso["pred_freight_f2_reproduced"]
    )

    grouped = rso.groupby(
        "order_id",
        sort=False,
    )

    expected = grouped[
        "pred_freight_oot_reproduced"
    ].sum(
        min_count=1
    ).rename(
        "expected_freight_oot_reproduced"
    )

    order_month = grouped["year_month"].first().rename(
        "year_month_reproduced"
    )

    reproduced = pd.concat(
        [expected, order_month],
        axis=1,
    ).reset_index()

    if reproduced["order_id"].duplicated().any():
        raise RuntimeError("Reproduced order_id duplicates.")

    hist = pd.read_csv(
        HIST,
        usecols=[
            "order_id",
            "year_month",
            "expected_freight_oot",
        ],
    )

    if hist["order_id"].duplicated().any():
        raise RuntimeError("Historical order_id duplicates.")

    parity = hist.merge(
        reproduced,
        on="order_id",
        how="outer",
        validate="1:1",
        indicator=True,
    )

    parity["historical_present"] = parity["_merge"].ne("right_only")
    parity["reproduced_present"] = parity["_merge"].ne("left_only")

    hist_v = parity["expected_freight_oot"]
    repro_v = parity["expected_freight_oot_reproduced"]

    parity["null_pattern_match"] = hist_v.isna().eq(
        repro_v.isna()
    )

    both = hist_v.notna() & repro_v.notna()

    parity["abs_diff"] = np.nan
    parity.loc[both, "abs_diff"] = np.abs(
        hist_v.loc[both] - repro_v.loc[both]
    )

    parity["numeric_match"] = False
    parity.loc[
        hist_v.isna() & repro_v.isna(),
        "numeric_match",
    ] = True

    parity.loc[both, "numeric_match"] = np.isclose(
        hist_v.loc[both],
        repro_v.loc[both],
        atol=ATOL,
        rtol=RTOL,
    )

    left_only = int((parity["_merge"] == "left_only").sum())
    right_only = int((parity["_merge"] == "right_only").sum())
    null_mismatch = int((~parity["null_pattern_match"]).sum())
    numeric_mismatch = int((~parity["numeric_match"]).sum())

    max_abs_diff = (
        float(parity.loc[both, "abs_diff"].max())
        if both.any()
        else None
    )

    median_abs_diff = (
        float(parity.loc[both, "abs_diff"].median())
        if both.any()
        else None
    )

    strict_parity_pass = (
        left_only == 0
        and right_only == 0
        and null_mismatch == 0
        and numeric_mismatch == 0
    )

    reproduced.to_csv(
        OUT / "03_order_level_reproduction.csv",
        index=False,
    )

    mismatch_rows = parity.loc[
        (
            (parity["_merge"] != "both")
            | (~parity["null_pattern_match"])
            | (~parity["numeric_match"])
        )
    ].copy()

    mismatch_rows.to_csv(
        OUT / "04_order_level_parity_mismatches.csv",
        index=False,
    )

    # Monthly parity by historical year_month.
    parity["month_for_audit"] = (
        parity["year_month"]
        .fillna(parity["year_month_reproduced"])
        .astype("string")
    )

    month_rows = []

    for month, g in parity.groupby(
        "month_for_audit",
        dropna=False,
        sort=True,
    ):
        both_m = (
            g["expected_freight_oot"].notna()
            & g["expected_freight_oot_reproduced"].notna()
        )

        month_rows.append({
            "year_month": month,
            "rows": len(g),
            "historical_nonnull": int(
                g["expected_freight_oot"].notna().sum()
            ),
            "reproduced_nonnull": int(
                g["expected_freight_oot_reproduced"].notna().sum()
            ),
            "membership_mismatch": int(
                (g["_merge"] != "both").sum()
            ),
            "null_pattern_mismatch": int(
                (~g["null_pattern_match"]).sum()
            ),
            "numeric_mismatch": int(
                (~g["numeric_match"]).sum()
            ),
            "max_abs_diff": (
                float(g.loc[both_m, "abs_diff"].max())
                if both_m.any()
                else np.nan
            ),
        })

    pd.DataFrame(month_rows).to_csv(
        OUT / "05_historical_parity_by_month.csv",
        index=False,
    )

    parity_summary = {
        "historical_rows": int(len(hist)),
        "reproduced_rows": int(len(reproduced)),
        "left_only": left_only,
        "right_only": right_only,
        "historical_nonnull": int(
            hist["expected_freight_oot"].notna().sum()
        ),
        "reproduced_nonnull": int(
            reproduced["expected_freight_oot_reproduced"].notna().sum()
        ),
        "null_pattern_mismatch": null_mismatch,
        "numeric_mismatch": numeric_mismatch,
        "both_nonnull_compared": int(both.sum()),
        "max_abs_diff": max_abs_diff,
        "median_abs_diff": median_abs_diff,
        "atol": ATOL,
        "rtol": RTOL,
        "strict_parity_pass": strict_parity_pass,
    }

    (OUT / "06_historical_parity_summary.json").write_text(
        json.dumps(
            parity_summary,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    if not strict_parity_pass:
        print(json.dumps(parity_summary, indent=2))
        raise RuntimeError(
            "Independent Expected Freight reproduction failed parity."
        )

    # Gold supervised coverage protocol.
    con = duckdb.connect(str(DB), read_only=True)

    gold = con.execute(
        """
        SELECT order_id
        FROM gold.ml_delivery_supervised_core_v1
        ORDER BY order_id
        """
    ).df()

    con.close()

    if gold["order_id"].duplicated().any():
        raise RuntimeError("Gold supervised order_id duplicates.")

    coverage = gold.merge(
        reproduced,
        on="order_id",
        how="left",
        validate="1:1",
        indicator=True,
    )

    coverage["expected_freight_available"] = (
        coverage["expected_freight_oot_reproduced"].notna()
    )

    coverage["availability_reason"] = "AVAILABLE"

    coverage.loc[
        coverage["_merge"].eq("left_only"),
        "availability_reason",
    ] = "NO_ROUTE_ORDER_MEMBERSHIP"

    route_member = coverage["_merge"].eq("both")

    warmup = (
        route_member
        & coverage["year_month_reproduced"].astype("string").isin(
            warmup_months
        )
        & coverage["expected_freight_oot_reproduced"].isna()
    )

    coverage.loc[
        warmup,
        "availability_reason",
    ] = "WARMUP_MIN_TRAIN_MONTHS"

    missing_time = (
        route_member
        & coverage["year_month_reproduced"].isna()
        & coverage["expected_freight_oot_reproduced"].isna()
    )

    coverage.loc[
        missing_time,
        "availability_reason",
    ] = "MISSING_TIME_KEY"

    other_null = (
        route_member
        & coverage["expected_freight_oot_reproduced"].isna()
        & ~warmup
        & ~missing_time
    )

    coverage.loc[
        other_null,
        "availability_reason",
    ] = "UNEXPECTED_POST_WARMUP_MISSING"

    # Production routing rule.
    coverage["serving_model"] = np.where(
        coverage["expected_freight_available"],
        "M1_ELIGIBLE",
        "M0_FALLBACK",
    )

    coverage[
        [
            "order_id",
            "year_month_reproduced",
            "expected_freight_oot_reproduced",
            "expected_freight_available",
            "availability_reason",
            "serving_model",
        ]
    ].to_csv(
        OUT / "07_gold_expected_freight_coverage.csv",
        index=False,
    )

    summary = (
        coverage.groupby(
            ["availability_reason", "serving_model"],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
    )

    summary["share_of_gold"] = summary["rows"] / len(coverage)

    summary.to_csv(
        OUT / "08_gold_coverage_summary.csv",
        index=False,
    )

    available_rows = int(
        coverage["expected_freight_available"].sum()
    )

    unavailable_rows = int(
        (~coverage["expected_freight_available"]).sum()
    )

    unexpected_after_warmup = int(
        coverage["availability_reason"]
        .eq("UNEXPECTED_POST_WARMUP_MISSING")
        .sum()
    )

    missing_time_count = int(
        coverage["availability_reason"]
        .eq("MISSING_TIME_KEY")
        .sum()
    )

    release_pass = (
        strict_parity_pass
        and unexpected_after_warmup == 0
        and missing_time_count == 0
    )

    production_contract = {
        "contract": "EXPECTED_FREIGHT_PRODUCTION_CONTRACT_V1",
        "status": (
            "RELEASED_FOR_M1_CANDIDATE_WITH_M0_FALLBACK"
            if release_pass
            else "BLOCKED"
        ),
        "prediction_time": "order_purchase_timestamp",
        "source_runtime_manifest": (
            "configs/expected_freight_runtime_manifest_v1.json"
        ),
        "runtime_reproduction": {
            "strict_historical_parity_pass": strict_parity_pass,
            "historical_rows": int(len(hist)),
            "reproduced_rows": int(len(reproduced)),
            "historical_nonnull": int(
                hist["expected_freight_oot"].notna().sum()
            ),
            "reproduced_nonnull": int(
                reproduced["expected_freight_oot_reproduced"].notna().sum()
            ),
            "numeric_mismatch": numeric_mismatch,
            "null_pattern_mismatch": null_mismatch,
            "max_abs_diff": max_abs_diff,
        },
        "gold_supervised_coverage": {
            "rows": int(len(coverage)),
            "available_rows": available_rows,
            "unavailable_rows": unavailable_rows,
            "available_share": available_rows / len(coverage),
            "warmup_months": warmup_months,
            "unexpected_post_warmup_missing": unexpected_after_warmup,
            "missing_time_key": missing_time_count,
        },
        "serving_protocol": {
            "if_expected_freight_available": "M1_ELIGIBLE",
            "otherwise": "M0_FALLBACK",
            "silent_complete_case_drop_forbidden": True,
            "imputation_required": False,
        },
        "release": {
            "expected_freight_released_for_m1_candidate": release_pass,
            "m1_released": False,
            "late_risk_model_trained": False,
        },
    }

    CONTRACT.write_text(
        json.dumps(
            production_contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    DOC.write_text(
        f"""# Expected Freight Runtime Reproduction V1

## Independent implementation

This stage independently reproduced the frozen F2 runtime contract without
calling the historical producer.

### Model

- Features: `{FEATURES}`
- Target: `log1p(seller_freight)`
- Estimator: `HistGradientBoostingRegressor(max_iter=100, random_state=42)`
- Minimum training months: `{MIN_TRAIN_MONTHS}`
- Temporal protocol: expanding monthly out-of-time
- Output transform: `maximum(0, expm1(pred_log_f2))`
- Seller-to-order aggregation: sum with `min_count=1`

## Historical parity

- Historical rows: {len(hist)}
- Reproduced rows: {len(reproduced)}
- Historical non-null predictions: {int(hist["expected_freight_oot"].notna().sum())}
- Reproduced non-null predictions: {int(reproduced["expected_freight_oot_reproduced"].notna().sum())}
- Membership mismatch: {left_only + right_only}
- Null-pattern mismatch: {null_mismatch}
- Numeric mismatch: {numeric_mismatch}
- Maximum absolute difference: {max_abs_diff}
- Strict parity: {"PASS" if strict_parity_pass else "FAIL"}

## Gold supervised serving coverage

- Gold rows: {len(coverage)}
- Expected Freight available: {available_rows}
- Expected Freight unavailable: {unavailable_rows}
- Available share: {available_rows / len(coverage):.10f}
- Warm-up months: `{warmup_months}`
- Unexpected post-warmup missing: {unexpected_after_warmup}
- Missing time key: {missing_time_count}

No order is silently dropped.

Orders with Expected Freight available are eligible for M1.
Orders without Expected Freight use the M0 fallback.

## Release

Expected Freight candidate release:
`{"PASS" if release_pass else "BLOCKED"}`

M1 remains unreleased and no late-risk model is trained in this stage.
""",
        encoding="utf-8",
    )

    with (OUT / "09_runtime_reproduction_decision.txt").open(
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write("=" * 96 + "\n")
        fh.write("DELIVERY RISK — BLOCK 08 EXPECTED FREIGHT RUNTIME REPRODUCTION\n")
        fh.write("=" * 96 + "\n\n")
        fh.write(f"HISTORICAL_ROWS              : {len(hist)}\n")
        fh.write(f"REPRODUCED_ROWS              : {len(reproduced)}\n")
        fh.write(f"HISTORICAL_NONNULL           : {int(hist['expected_freight_oot'].notna().sum())}\n")
        fh.write(f"REPRODUCED_NONNULL           : {int(reproduced['expected_freight_oot_reproduced'].notna().sum())}\n")
        fh.write(f"MEMBERSHIP_MISMATCH          : {left_only + right_only}\n")
        fh.write(f"NULL_PATTERN_MISMATCH        : {null_mismatch}\n")
        fh.write(f"NUMERIC_MISMATCH             : {numeric_mismatch}\n")
        fh.write(f"MAX_ABS_DIFF                 : {max_abs_diff}\n")
        fh.write(f"STRICT_PARITY                : {'PASS' if strict_parity_pass else 'FAIL'}\n")
        fh.write(f"GOLD_ROWS                    : {len(coverage)}\n")
        fh.write(f"GOLD_EXPECTED_AVAILABLE      : {available_rows}\n")
        fh.write(f"GOLD_EXPECTED_UNAVAILABLE    : {unavailable_rows}\n")
        fh.write(f"GOLD_AVAILABLE_SHARE         : {available_rows / len(coverage):.10f}\n")
        fh.write(f"UNEXPECTED_POST_WARMUP_MISSING: {unexpected_after_warmup}\n")
        fh.write(f"MISSING_TIME_KEY             : {missing_time_count}\n")
        fh.write("SILENT_COMPLETE_CASE_DROP    : FORBIDDEN\n")
        fh.write("M0_FALLBACK                  : ENABLED\n")
        fh.write(f"EXPECTED_FREIGHT_RELEASED    : {'YES_WITH_FALLBACK' if release_pass else 'NO'}\n")
        fh.write("M1_RELEASED                  : NO\n")
        fh.write("LATE_RISK_MODEL_TRAINED      : NO\n\n")
        fh.write(
            "STATUS = EXPECTED_FREIGHT_RUNTIME_REPRODUCED_AND_RELEASED_FOR_M1_CANDIDATE\n"
            if release_pass
            else "STATUS = EXPECTED_FREIGHT_RUNTIME_REPRODUCTION_BLOCKED\n"
        )

    print()
    print("=" * 96)
    print("INDEPENDENT EXPECTED FREIGHT REPRODUCTION")
    print("=" * 96)
    print(json.dumps(parity_summary, indent=2))
    print()
    print("GOLD COVERAGE:")
    print(summary.to_string(index=False))
    print()
    print((OUT / "09_runtime_reproduction_decision.txt").read_text(
        encoding="utf-8"
    ))

    if not release_pass:
        raise RuntimeError(
            "Expected Freight production release conditions not met."
        )


if __name__ == "__main__":
    main()
