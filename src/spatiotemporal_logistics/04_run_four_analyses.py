#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf

from statsmodels.stats.sandwich_covariance import (
    cov_cluster_2groups,
)


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
ART = ROOT / "artifacts" / "spatiotemporal_logistics"
REP = ROOT / "reports" / "spatiotemporal_logistics"

MIN_TRAIN_MONTHS = 3

print("=" * 110)
print("STAGE 04 — FOUR ANALYSES")
print("=" * 110)


# ==================================================================================================
# A1 — COST STRUCTURE
# ==================================================================================================

master = pd.read_csv(
    ART / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv",
    dtype={
        "customer_id_municipio": "string",
    },
)

den = (
    master["total_price"]
    +
    master["total_freight"]
)

master["freight_burden"] = np.where(
    den > 0,
    master["total_freight"] / den,
    np.nan,
)

cost_cols = [
    "total_freight",
    "freight_burden",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "total_price",
]

desc = []

for c in cost_cols:

    s = pd.to_numeric(
        master[c],
        errors="coerce",
    ).dropna()

    desc.append(
        {
            "variable": c,
            "n": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "median": s.median(),
            "q1": s.quantile(.25),
            "q3": s.quantile(.75),
            "p05": s.quantile(.05),
            "p10": s.quantile(.10),
            "p90": s.quantile(.90),
            "p95": s.quantile(.95),
            "p99": s.quantile(.99),
            "min": s.min(),
            "max": s.max(),
        }
    )

pd.DataFrame(desc).to_csv(
    REP / "06_cost_descriptive.csv",
    index=False,
)

master[cost_cols].corr(
    method="pearson",
).to_csv(
    REP / "07a_cost_pearson.csv"
)

master[cost_cols].corr(
    method="spearman",
).to_csv(
    REP / "07b_cost_spearman.csv"
)

print("[PASS A1] COST STRUCTURE")


# ==================================================================================================
# A2 — EXPECTED / TYPICAL FREIGHT — OOT
# ==================================================================================================

rso = pd.read_csv(
    ART / "01_ROUTE_SELLER_ORDER.csv"
)

for c in [
    "seller_freight",
    "great_circle_distance_km",
    "seller_weight_g",
    "seller_volume_proxy_cm3",
    "seller_price",
]:
    rso[c] = pd.to_numeric(
        rso[c],
        errors="coerce",
    )

rso["log_freight"] = np.log1p(
    rso["seller_freight"].where(
        rso["seller_freight"] >= 0
    )
)

rso["log_distance"] = np.log1p(
    rso["great_circle_distance_km"].where(
        rso["great_circle_distance_km"] >= 0
    )
)

# Missing físico permanece NaN.
rso["log_weight"] = np.log1p(
    rso["seller_weight_g"].where(
        rso["seller_weight_g"] >= 0
    )
)

rso["log_volume"] = np.log1p(
    rso["seller_volume_proxy_cm3"].where(
        rso["seller_volume_proxy_cm3"] >= 0
    )
)

rso["log_price"] = np.log1p(
    rso["seller_price"].where(
        rso["seller_price"] >= 0
    )
)

features_f = [
    "log_distance",
    "log_weight",
    "log_volume",
    "log_price",
    "seller_item_count",
]

months = sorted(
    rso["year_month"]
    .dropna()
    .astype(str)
    .unique()
)

for name in ["f0", "f1", "f2"]:
    rso[f"pred_log_{name}"] = np.nan

folds = []

for i in range(
    MIN_TRAIN_MONTHS,
    len(months),
):

    train_months = months[:i]
    test_month = months[i]

    train_mask = (
        rso["year_month"]
        .isin(train_months)
        &
        rso["log_freight"].notna()
    )

    test_mask = (
        rso["year_month"]
        .eq(test_month)
    )

    if not train_mask.any() or not test_mask.any():
        continue

    X_train = rso.loc[
        train_mask,
        features_f,
    ]

    y_train = rso.loc[
        train_mask,
        "log_freight",
    ]

    X_test = rso.loc[
        test_mask,
        features_f,
    ]


    # F0 — historical median
    baseline = float(
        y_train.median()
    )

    rso.loc[
        test_mask,
        "pred_log_f0",
    ] = baseline


    # F1 — linear complete-case only.
    train_complete = (
        X_train.notna().all(axis=1)
    )

    test_complete = (
        X_test.notna().all(axis=1)
    )

    if train_complete.sum() >= 100:

        f1 = LinearRegression()

        f1.fit(
            X_train.loc[train_complete],
            y_train.loc[train_complete],
        )

        test_indices = (
            X_test.index[
                test_complete
            ]
        )

        if len(test_indices):

            rso.loc[
                test_indices,
                "pred_log_f1",
            ] = f1.predict(
                X_test.loc[
                    test_complete
                ]
            )


    # F2 — HistGradientBoosting with native NaN handling.
    f2 = HistGradientBoostingRegressor(
        max_iter=100,
        random_state=42,
    )

    f2.fit(
        X_train,
        y_train,
    )

    rso.loc[
        test_mask,
        "pred_log_f2",
    ] = f2.predict(
        X_test
    )

    folds.append(
        {
            "test_month": test_month,
            "train_start": train_months[0],
            "train_end": train_months[-1],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
    )

pd.DataFrame(folds).to_csv(
    REP / "08_freight_oot_folds.csv",
    index=False,
)

for name in ["f0", "f1", "f2"]:

    rso[f"pred_freight_{name}"] = np.maximum(
        0.0,
        np.expm1(
            rso[f"pred_log_{name}"]
        ),
    )

metrics_f = []

for name in ["f0", "f1", "f2"]:

    pred_col = f"pred_freight_{name}"

    valid = (
        rso[pred_col].notna()
        &
        rso["seller_freight"].notna()
    )

    y = rso.loc[
        valid,
        "seller_freight",
    ]

    p = rso.loc[
        valid,
        pred_col,
    ]

    metrics_f.append(
        {
            "model": name.upper(),
            "n_predictions": int(valid.sum()),
            "coverage_pct":
                100.0 * valid.mean(),
            "mae":
                mean_absolute_error(y, p),
            "median_ae":
                median_absolute_error(y, p),
            "rmse":
                root_mean_squared_error(y, p),
            "r2":
                r2_score(y, p),
        }
    )

pd.DataFrame(
    metrics_f
).to_csv(
    REP / "09_freight_model_comparison.csv",
    index=False,
)

# F2 = primary structural model for downstream.
rso["pred_freight_oot"] = (
    rso["pred_freight_f2"]
)

rso["freight_residual_oot"] = (
    rso["seller_freight"]
    -
    rso["pred_freight_oot"]
)

g = rso.groupby(
    "order_id",
    sort=False,
)

expected = (
    g["pred_freight_oot"]
    .sum(min_count=1)
    .rename("expected_freight_oot")
)

residual = (
    g["freight_residual_oot"]
    .sum(min_count=1)
    .rename("freight_residual_OOT")
)

order_resid = pd.concat(
    [
        expected,
        residual,
    ],
    axis=1,
).reset_index()

master = master.merge(
    order_resid,
    on="order_id",
    how="left",
    validate="1:1",
)

master["freight_oot_available"] = (
    master["freight_residual_OOT"]
    .notna()
    .astype("int8")
)

master["freight_ratio_to_expected"] = np.where(
    master["expected_freight_oot"] > 0,
    (
        master["total_freight"]
        /
        master["expected_freight_oot"]
    ),
    np.nan,
)

master.to_csv(
    ART / "06_EXPECTED_FREIGHT_OOT.csv",
    index=False,
)

print("[PASS A2] EXPECTED FREIGHT OOT")

del rso, order_resid
gc.collect()


# ==================================================================================================
# SHARED MODEL FEATURES
# ==================================================================================================

master["log_distance"] = np.log1p(
    master[
        "distance_freight_weighted_km"
    ].where(
        master[
            "distance_freight_weighted_km"
        ] >= 0
    )
)

master["log_weight"] = np.log1p(
    master["total_weight_g"].where(
        master["total_weight_g"] >= 0
    )
)

master["log_volume"] = np.log1p(
    master[
        "product_volume_sum_proxy_cm3"
    ].where(
        master[
            "product_volume_sum_proxy_cm3"
        ] >= 0
    )
)

master["log_freight"] = np.log1p(
    master["total_freight"].where(
        master["total_freight"] >= 0
    )
)

master["log_pop"] = np.log1p(
    master[
        "customer_population"
    ].where(
        master[
            "customer_population"
        ] >= 0
    )
)

master["log_gdp"] = np.log1p(
    master[
        "customer_gdp_per_capita"
    ].where(
        master[
            "customer_gdp_per_capita"
        ] >= 0
    )
)

master["log_diesel"] = np.log1p(
    master[
        "customer_diesel_common_municipal"
    ].where(
        master[
            "customer_diesel_common_municipal"
        ] > 0
    )
)


# ==================================================================================================
# A3 — DELIVERY SPEED
# ==================================================================================================

speed_features = {
    "S0_Physical": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
    ],

    "S1_Cost": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
        "log_freight",
        "freight_burden",
    ],

    "S2_Anomalies": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
        "log_freight",
        "freight_burden",
        "freight_residual_OOT",
    ],

    "S3A_Context_All": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
        "log_freight",
        "freight_burden",
        "freight_residual_OOT",
        "log_pop",
        "log_gdp",
        "truck_strike_2018",
    ],

    "S3B_ANP_Subset": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
        "log_freight",
        "freight_burden",
        "freight_residual_OOT",
        "log_pop",
        "log_gdp",
        "log_diesel",
        "truck_strike_2018",
    ],
}

speed_formulas = {
    name:
        "actual_delivery_days ~ "
        +
        " + ".join(features)
    for name, features
    in speed_features.items()
}

speed_in_sample = []

for name, formula in speed_formulas.items():

    try:
        fit = smf.ols(
            formula=formula,
            data=master,
        ).fit()

        speed_in_sample.append(
            {
                "model": name,
                "status": "PASS",
                "nobs": int(fit.nobs),
                "r2_in_sample": fit.rsquared,
                "adj_r2_in_sample":
                    fit.rsquared_adj,
            }
        )

    except Exception as exc:

        speed_in_sample.append(
            {
                "model": name,
                "status":
                    f"FAIL:{type(exc).__name__}",
            }
        )

pd.DataFrame(
    speed_in_sample
).to_csv(
    REP / "11a_speed_nested_in_sample.csv",
    index=False,
)


# OOT speed evaluation
speed_months = sorted(
    master["year_month"]
    .dropna()
    .astype(str)
    .unique()
)

speed_oot_metrics = []

for model_name, features in speed_features.items():

    pred = pd.Series(
        np.nan,
        index=master.index,
        dtype=float,
    )

    needed = (
        features
        +
        [
            "actual_delivery_days",
            "year_month",
        ]
    )

    for i in range(
        MIN_TRAIN_MONTHS,
        len(speed_months),
    ):

        train_months = speed_months[:i]
        test_month = speed_months[i]

        train = master[
            master["year_month"].isin(train_months)
        ].dropna(
            subset=needed
        )

        test = master[
            master["year_month"].eq(test_month)
        ].dropna(
            subset=needed
        )

        if (
            len(train) < 100
            or
            test.empty
        ):
            continue

        model = LinearRegression()

        model.fit(
            train[features],
            train["actual_delivery_days"],
        )

        pred.loc[test.index] = model.predict(
            test[features]
        )

    valid = (
        pred.notna()
        &
        master[
            "actual_delivery_days"
        ].notna()
    )

    y = master.loc[
        valid,
        "actual_delivery_days",
    ]

    p = pred.loc[valid]

    speed_oot_metrics.append(
        {
            "model": model_name,
            "n_oot": int(valid.sum()),
            "mae_oot":
                mean_absolute_error(y, p)
                if len(y)
                else np.nan,
            "median_ae_oot":
                median_absolute_error(y, p)
                if len(y)
                else np.nan,
            "rmse_oot":
                root_mean_squared_error(y, p)
                if len(y)
                else np.nan,
            "r2_oot":
                r2_score(y, p)
                if len(y) > 1
                else np.nan,
        }
    )

pd.DataFrame(
    speed_oot_metrics
).to_csv(
    REP / "11b_speed_nested_oot.csv",
    index=False,
)

print("[PASS A3] DELIVERY SPEED")


# ==================================================================================================
# A4 — DELIVERY RELIABILITY
# ==================================================================================================

rel_features = {
    "R_A_Observed": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
        "freight_residual_OOT",
    ],

    "R_B_Promise_Adjusted": [
        "log_distance",
        "log_weight",
        "log_volume",
        "route_sellers_total",
        "any_interstate_route",
        "freight_residual_OOT",
        "promised_delivery_days",
    ],
}

rel_formulas = {
    name:
        "late_delivery_calendar_day ~ "
        +
        " + ".join(features)
    for name, features
    in rel_features.items()
}


# Retrospective full-sample inference, clearly labeled.
rel_inference = []

for name, formula in rel_formulas.items():

    try:

        fit = smf.logit(
            formula=formula,
            data=master,
        ).fit(
            disp=False,
            maxiter=200,
        )

        rel_inference.append(
            {
                "model": name,
                "status": "PASS",
                "nobs": int(fit.nobs),
                "pseudo_r2_in_sample":
                    fit.prsquared,
            }
        )

    except Exception as exc:

        rel_inference.append(
            {
                "model": name,
                "status":
                    f"FAIL:{type(exc).__name__}",
            }
        )

pd.DataFrame(
    rel_inference
).to_csv(
    REP / "13a_reliability_inference.csv",
    index=False,
)


# Expanding-window OOT classification metrics.
rel_oot_metrics = []

for model_name, features in rel_features.items():

    pred = pd.Series(
        np.nan,
        index=master.index,
        dtype=float,
    )

    needed = (
        features
        +
        [
            "late_delivery_calendar_day",
            "year_month",
        ]
    )

    for i in range(
        MIN_TRAIN_MONTHS,
        len(speed_months),
    ):

        train_months = speed_months[:i]
        test_month = speed_months[i]

        train = master[
            master["year_month"].isin(train_months)
        ].dropna(
            subset=needed
        )

        test = master[
            master["year_month"].eq(test_month)
        ].dropna(
            subset=needed
        )

        if (
            len(train) < 200
            or
            test.empty
            or
            train[
                "late_delivery_calendar_day"
            ].nunique() < 2
        ):
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        )

        clf.fit(
            train[features],
            train[
                "late_delivery_calendar_day"
            ],
        )

        pred.loc[test.index] = (
            clf.predict_proba(
                test[features]
            )[:, 1]
        )

    valid = (
        pred.notna()
        &
        master[
            "late_delivery_calendar_day"
        ].notna()
    )

    y = master.loc[
        valid,
        "late_delivery_calendar_day",
    ].astype(int)

    p = pred.loc[valid]

    rel_oot_metrics.append(
        {
            "model": model_name,
            "n_oot": int(valid.sum()),
            "base_rate":
                y.mean()
                if len(y)
                else np.nan,
            "average_precision_oot":
                average_precision_score(y, p)
                if y.nunique() > 1
                else np.nan,
            "roc_auc_oot":
                roc_auc_score(y, p)
                if y.nunique() > 1
                else np.nan,
            "brier_oot":
                brier_score_loss(y, p)
                if len(y)
                else np.nan,
        }
    )

pd.DataFrame(
    rel_oot_metrics
).to_csv(
    REP / "13b_reliability_oot_metrics.csv",
    index=False,
)

print("[PASS A4] DELIVERY RELIABILITY")


# ==================================================================================================
# CLUSTERED INFERENCE
# ==================================================================================================

features_cluster = (
    speed_features[
        "S3A_Context_All"
    ]
)

needed_cluster = (
    features_cluster
    +
    [
        "actual_delivery_days",
        "customer_id_municipio",
    ]
)

cluster_df = master.dropna(
    subset=needed_cluster
).copy()

fit_one = smf.ols(
    formula=speed_formulas[
        "S3A_Context_All"
    ],
    data=cluster_df,
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":
            cluster_df[
                "customer_id_municipio"
            ]
    },
)

with open(
    REP / "18a_oneway_cluster_municipality.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(
        str(
            fit_one.summary()
        )
    )


# Two-way only in single-seller orders where seller ID is meaningful.
two_way_status = "NOT_RUN"

try:

    needed_ss = (
        features_cluster
        +
        [
            "actual_delivery_days",
            "customer_id_municipio",
            "seller_id",
            "single_seller_order",
        ]
    )

    cluster_ss = master[
        master[
            "single_seller_order"
        ].eq(1)
    ].dropna(
        subset=needed_ss
    ).copy()

    fit_ss = smf.ols(
        formula=speed_formulas[
            "S3A_Context_All"
        ],
        data=cluster_ss,
    ).fit()

    seller_groups = pd.factorize(
        cluster_ss["seller_id"]
    )[0]

    municipality_groups = pd.factorize(
        cluster_ss[
            "customer_id_municipio"
        ]
    )[0]

    cov_both, cov_seller, cov_muni = (
        cov_cluster_2groups(
            fit_ss,
            seller_groups,
            municipality_groups,
        )
    )

    se_two_way = np.sqrt(
        np.diag(cov_both)
    )

    table_two = pd.DataFrame(
        {
            "term": fit_ss.params.index,
            "coef": fit_ss.params.values,
            "se_two_way":
                se_two_way,
            "t_two_way":
                (
                    fit_ss.params.values
                    /
                    se_two_way
                ),
        }
    )

    table_two.to_csv(
        REP / "18b_twoway_cluster_single_seller.csv",
        index=False,
    )

    two_way_status = "PASS"

except Exception as exc:

    two_way_status = (
        f"NON_FATAL_FAIL:"
        f"{type(exc).__name__}:"
        f"{exc}"
    )

    with open(
        REP / "18b_twoway_cluster_status.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(two_way_status)


# ==================================================================================================
# SIX SENSITIVITIES
# ==================================================================================================

sens_specs = {
    "SENS_1_SINGLE_SELLER":
        master[
            "single_seller_order"
        ].eq(1),

    "SENS_2_FULL_ROUTE":
        master[
            "route_distance_coverage"
        ].eq(1.0),

    "SENS_3_WEIGHT_VOLUME_COMPLETE":
        (
            master["weight_complete"].eq(1)
            &
            master["volume_complete"].eq(1)
        ),

    "SENS_4_ANP_MUNICIPAL":
        master[
            "customer_diesel_common_municipal"
        ].notna(),

    "SENS_6_HAVERSINE_PROXY":
        master[
            "log_distance"
        ].notna(),
}

q_low = master[
    "actual_delivery_days"
].quantile(.005)

q_high = master[
    "actual_delivery_days"
].quantile(.995)

sens_specs[
    "SENS_5_TRIM_0_5_99_5"
] = master[
    "actual_delivery_days"
].between(
    q_low,
    q_high,
)

sens = []

for name, mask in sens_specs.items():

    formula = (
        "actual_delivery_days ~ log_distance"
        if name == "SENS_6_HAVERSINE_PROXY"
        else speed_formulas[
            "S3A_Context_All"
        ]
    )

    try:

        fit = smf.ols(
            formula=formula,
            data=master.loc[mask],
        ).fit()

        sens.append(
            {
                "sensitivity": name,
                "status": "PASS",
                "nobs": int(fit.nobs),
                "r2_in_sample":
                    fit.rsquared,
            }
        )

    except Exception as exc:

        sens.append(
            {
                "sensitivity": name,
                "status":
                    f"FAIL:{type(exc).__name__}",
            }
        )

pd.DataFrame(
    sens
).to_csv(
    REP / "15_sensitivity_summary.csv",
    index=False,
)

print(f"[INFO] two-way cluster = {two_way_status}")
print("[PASS 04] FOUR ANALYSES COMPLETE")
