#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import numpy as np
import pandas as pd

import statsmodels.api as sm


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

REP = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
    / "gnadmm_delivery"
)

INPUT = (
    REP
    / "20b_constraint_ablation_oot_predictions.csv"
)


def logit_clip(p):
    p = np.clip(
        np.asarray(
            p,
            dtype=float
        ),
        1e-8,
        1.0 - 1e-8
    )

    return np.log(
        p
        /
        (
            1.0 - p
        )
    )


def calibration_intercept_slope(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=float
    )

    lp = logit_clip(
        p
    )

    Xcal = sm.add_constant(
        lp
    )

    try:
        fit = sm.GLM(
            y,
            Xcal,
            family=sm.families.Binomial()
        ).fit()

        return (
            float(
                fit.params[0]
            ),
            float(
                fit.params[1]
            ),
        )

    except Exception:
        return (
            np.nan,
            np.nan
        )


def calibration_bins(
    y,
    p,
    n_bins=10,
):

    temp = pd.DataFrame(
        {
            "y":
                np.asarray(
                    y,
                    dtype=float
                ),

            "p":
                np.asarray(
                    p,
                    dtype=float
                ),
        }
    )

    try:
        temp[
            "bin"
        ] = pd.qcut(
            temp["p"],
            q=n_bins,
            duplicates="drop"
        )

    except Exception:
        temp[
            "bin"
        ] = pd.cut(
            temp["p"],
            bins=n_bins,
            include_lowest=True
        )

    grouped = (
        temp
        .groupby(
            "bin",
            observed=True
        )
        .agg(
            n=("y", "size"),
            observed_rate=("y", "mean"),
            mean_predicted=("p", "mean"),
            min_predicted=("p", "min"),
            max_predicted=("p", "max"),
        )
        .reset_index()
    )

    grouped[
        "bin"
    ] = grouped[
        "bin"
    ].astype(
        str
    )

    grouped[
        "abs_calibration_gap"
    ] = np.abs(
        grouped[
            "observed_rate"
        ]
        -
        grouped[
            "mean_predicted"
        ]
    )

    return grouped


def calibration_statistics(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=float
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=float
        ),
        1e-8,
        1.0 - 1e-8
    )

    n = len(
        y
    )

    prevalence = float(
        np.mean(
            y
        )
    )

    mean_prediction = float(
        np.mean(
            p
        )
    )

    brier = float(
        np.mean(
            (
                y
                -
                p
            ) ** 2
        )
    )

    baseline_brier = float(
        np.mean(
            (
                y
                -
                prevalence
            ) ** 2
        )
    )

    brier_skill = (
        1.0
        -
        brier
        /
        baseline_brier
        if baseline_brier > 0
        else np.nan
    )

    bins = calibration_bins(
        y,
        p,
        n_bins=10
    )

    weights = (
        bins[
            "n"
        ].to_numpy(
            dtype=float
        )
        /
        n
    )

    reliability = float(
        np.sum(
            weights
            *
            (
                bins[
                    "mean_predicted"
                ].to_numpy()
                -
                bins[
                    "observed_rate"
                ].to_numpy()
            ) ** 2
        )
    )

    resolution = float(
        np.sum(
            weights
            *
            (
                bins[
                    "observed_rate"
                ].to_numpy()
                -
                prevalence
            ) ** 2
        )
    )

    uncertainty = float(
        prevalence
        *
        (
            1.0
            -
            prevalence
        )
    )

    ece = float(
        np.sum(
            weights
            *
            bins[
                "abs_calibration_gap"
            ].to_numpy()
        )
    )

    cal_intercept, cal_slope = (
        calibration_intercept_slope(
            y,
            p
        )
    )

    return {
        "n":
            int(n),

        "prevalence":
            prevalence,

        "mean_prediction":
            mean_prediction,

        "calibration_in_the_large":
            float(
                mean_prediction
                -
                prevalence
            ),

        "brier":
            brier,

        "baseline_brier":
            baseline_brier,

        "brier_skill_score":
            float(
                brier_skill
            ),

        "ECE_quantile10":
            ece,

        "calibration_intercept":
            cal_intercept,

        "calibration_slope":
            cal_slope,

        "murphy_reliability_binned":
            reliability,

        "murphy_resolution_binned":
            resolution,

        "murphy_uncertainty":
            uncertainty,
    }, bins


print("=" * 110)
print("MODULE 21 — CALIBRATION / RESOLUTION AUDIT")
print("=" * 110)

df = pd.read_csv(
    INPUT
)

y = df[
    "y_true"
].astype(
    float
).to_numpy()


pred_cols = [
    c
    for c in df.columns
    if c.startswith(
        "pred_"
    )
]


overall_rows = []

bin_frames = []

monthly_rows = []


for pred_col in pred_cols:

    model = pred_col.replace(
        "pred_",
        "",
        1
    )

    p = df[
        pred_col
    ].astype(
        float
    ).to_numpy()

    stats, bins = (
        calibration_statistics(
            y,
            p
        )
    )

    stats[
        "model"
    ] = model

    overall_rows.append(
        stats
    )

    bins[
        "model"
    ] = model

    bin_frames.append(
        bins
    )


    for month, g in df.groupby(
        "test_month"
    ):

        y_m = g[
            "y_true"
        ].astype(
            float
        ).to_numpy()

        p_m = g[
            pred_col
        ].astype(
            float
        ).to_numpy()

        if (
            len(y_m) < 100
            or
            np.unique(
                y_m
            ).size < 2
        ):
            continue

        stat_m, _ = (
            calibration_statistics(
                y_m,
                p_m
            )
        )

        monthly_rows.append(
            {
                "test_month":
                    month,

                "model":
                    model,

                "n":
                    stat_m[
                        "n"
                    ],

                "prevalence":
                    stat_m[
                        "prevalence"
                    ],

                "mean_prediction":
                    stat_m[
                        "mean_prediction"
                    ],

                "brier":
                    stat_m[
                        "brier"
                    ],

                "ECE_quantile10":
                    stat_m[
                        "ECE_quantile10"
                    ],

                "calibration_intercept":
                    stat_m[
                        "calibration_intercept"
                    ],

                "calibration_slope":
                    stat_m[
                        "calibration_slope"
                    ],
            }
        )


overall = pd.DataFrame(
    overall_rows
)

overall = overall[
    [
        "model",
        "n",
        "prevalence",
        "mean_prediction",
        "calibration_in_the_large",
        "brier",
        "baseline_brier",
        "brier_skill_score",
        "ECE_quantile10",
        "calibration_intercept",
        "calibration_slope",
        "murphy_reliability_binned",
        "murphy_resolution_binned",
        "murphy_uncertainty",
    ]
]


overall.to_csv(
    REP
    / "21a_oot_calibration_summary.csv",
    index=False
)


pd.concat(
    bin_frames,
    ignore_index=True
).to_csv(
    REP
    / "21b_oot_calibration_bins.csv",
    index=False
)


monthly = pd.DataFrame(
    monthly_rows
)

monthly.to_csv(
    REP
    / "21c_monthly_calibration.csv",
    index=False
)


best_brier = str(
    overall.loc[
        overall[
            "brier"
        ].idxmin(),
        "model"
    ]
)

best_ece = str(
    overall.loc[
        overall[
            "ECE_quantile10"
        ].idxmin(),
        "model"
    ]
)


overall[
    "calibration_parameter_distance"
] = (
    np.abs(
        overall[
            "calibration_intercept"
        ]
    )
    +
    np.abs(
        overall[
            "calibration_slope"
        ]
        -
        1.0
    )
)


best_cal_params = str(
    overall.loc[
        overall[
            "calibration_parameter_distance"
        ].idxmin(),
        "model"
    ]
)


decision = {
    "status":
        "PASS",

    "best_pooled_OOT_Brier":
        best_brier,

    "best_pooled_OOT_ECE":
        best_ece,

    "best_calibration_intercept_slope_proximity":
        best_cal_params,

    "interpretation":
        (
            "Brier is not interpreted as calibration "
            "alone. Reliability diagrams/bins, ECE, "
            "calibration intercept/slope and a binned "
            "Murphy-style decomposition are reported "
            "jointly."
        ),

    "murphy_decomposition_note":
        (
            "Reliability and resolution are computed "
            "after quantile binning and should be "
            "interpreted as binned approximations."
        ),
}


with open(
    REP
    / "21d_CALIBRATION_DECISION.json",
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
print(
    overall.to_string(
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
    "[PASS 21] CALIBRATION / "
    "RESOLUTION AUDIT COMPLETE."
)
