#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, binomtest


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

REP = (
    ROOT
    / "reports"
    / "spatiotemporal_logistics"
    / "scientific"
    / "gnadmm_delivery"
)

INPUT = REP / "18b_gnadmm_oot_fold_metrics.csv"

RNG = np.random.default_rng(20260831)

N_BOOT = 20000
BLOCK_LENGTH = 3


def percentile_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)

    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def paired_month_bootstrap(x, n_boot=N_BOOT):
    x = np.asarray(x, dtype=float)

    n = len(x)

    out = np.empty(
        n_boot,
        dtype=float
    )

    for b in range(n_boot):
        idx = RNG.integers(
            0,
            n,
            size=n
        )

        out[b] = float(
            np.mean(
                x[idx]
            )
        )

    return out


def moving_block_bootstrap(
    x,
    block_length=BLOCK_LENGTH,
    n_boot=N_BOOT,
):
    x = np.asarray(
        x,
        dtype=float
    )

    n = len(x)

    if n <= block_length:
        return paired_month_bootstrap(
            x,
            n_boot=n_boot
        )

    possible_starts = np.arange(
        0,
        n - block_length + 1
    )

    n_blocks = int(
        np.ceil(
            n / block_length
        )
    )

    out = np.empty(
        n_boot,
        dtype=float
    )

    for b in range(n_boot):

        starts = RNG.choice(
            possible_starts,
            size=n_blocks,
            replace=True
        )

        sample_parts = [
            x[
                s:
                s + block_length
            ]
            for s in starts
        ]

        sample = np.concatenate(
            sample_parts
        )[:n]

        out[b] = float(
            np.mean(
                sample
            )
        )

    return out


def safe_wilcoxon(x):
    x = np.asarray(
        x,
        dtype=float
    )

    if np.allclose(
        x,
        0.0
    ):
        return {
            "statistic": 0.0,
            "p_two_sided": 1.0,
            "p_greater": 1.0,
        }

    try:
        w2 = wilcoxon(
            x,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        wg = wilcoxon(
            x,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )

        return {
            "statistic":
                float(w2.statistic),

            "p_two_sided":
                float(w2.pvalue),

            "p_greater":
                float(wg.pvalue),
        }

    except Exception as exc:

        return {
            "statistic": None,
            "p_two_sided": None,
            "p_greater": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def sign_test(x):
    x = np.asarray(
        x,
        dtype=float
    )

    nz = x[
        ~np.isclose(
            x,
            0.0
        )
    ]

    if len(nz) == 0:
        return {
            "positive": 0,
            "negative": 0,
            "n_nonzero": 0,
            "p_two_sided": 1.0,
        }

    positive = int(
        np.sum(
            nz > 0
        )
    )

    negative = int(
        np.sum(
            nz < 0
        )
    )

    test = binomtest(
        positive,
        len(nz),
        p=0.5,
        alternative="two-sided",
    )

    return {
        "positive": positive,
        "negative": negative,
        "n_nonzero": int(len(nz)),
        "p_two_sided": float(test.pvalue),
    }


def leave_one_month_out_means(x):
    x = np.asarray(
        x,
        dtype=float
    )

    return np.array(
        [
            np.mean(
                np.delete(
                    x,
                    i
                )
            )
            for i in range(
                len(x)
            )
        ],
        dtype=float,
    )


print("=" * 110)
print("MODULE 19 — PAIRED TEMPORAL INFERENCE")
print("=" * 110)

df = pd.read_csv(
    INPUT
)

required_models = [
    "LOGIT_MLE",
    "GN_BRIER",
    "GNADMM_BRIER_CONSTRAINED",
]

missing_models = (
    set(required_models)
    -
    set(df["model"].unique())
)

if missing_models:
    raise RuntimeError(
        f"Missing models: {missing_models}"
    )


metric_orientation = {
    # Higher is better.
    "average_precision": "higher",
    "roc_auc": "higher",

    # Lower is better.
    "brier": "lower",
    "log_loss": "lower",
}


candidates = [
    "GN_BRIER",
    "GNADMM_BRIER_CONSTRAINED",
]


rows = []

delta_rows = []

lomo_rows = []


for metric, orientation in metric_orientation.items():

    pivot = df.pivot(
        index="test_month",
        columns="model",
        values=metric
    ).sort_index()

    for candidate in candidates:

        pair = pivot[
            [
                "LOGIT_MLE",
                candidate
            ]
        ].dropna()

        baseline = pair[
            "LOGIT_MLE"
        ].to_numpy()

        cand = pair[
            candidate
        ].to_numpy()

        if orientation == "higher":
            delta = (
                cand
                -
                baseline
            )

        else:
            # Positive always means candidate is better.
            delta = (
                baseline
                -
                cand
            )

        ordinary_boot = (
            paired_month_bootstrap(
                delta
            )
        )

        block_boot = (
            moving_block_bootstrap(
                delta
            )
        )

        ci_ord = percentile_ci(
            ordinary_boot
        )

        ci_block = percentile_ci(
            block_boot
        )

        wil = safe_wilcoxon(
            delta
        )

        sign = sign_test(
            delta
        )

        lomo = (
            leave_one_month_out_means(
                delta
            )
        )

        rows.append(
            {
                "candidate":
                    candidate,

                "metric":
                    metric,

                "months":
                    int(
                        len(delta)
                    ),

                "mean_improvement":
                    float(
                        np.mean(
                            delta
                        )
                    ),

                "median_improvement":
                    float(
                        np.median(
                            delta
                        )
                    ),

                "std_improvement":
                    float(
                        np.std(
                            delta,
                            ddof=1
                        )
                    ),

                "months_better":
                    int(
                        np.sum(
                            delta > 0
                        )
                    ),

                "months_worse":
                    int(
                        np.sum(
                            delta < 0
                        )
                    ),

                "bootstrap_ci_lower":
                    ci_ord[0],

                "bootstrap_ci_upper":
                    ci_ord[1],

                "block_bootstrap_ci_lower":
                    ci_block[0],

                "block_bootstrap_ci_upper":
                    ci_block[1],

                "wilcoxon_p_two_sided":
                    wil.get(
                        "p_two_sided"
                    ),

                "wilcoxon_p_greater":
                    wil.get(
                        "p_greater"
                    ),

                "sign_test_p_two_sided":
                    sign[
                        "p_two_sided"
                    ],

                "lomo_min_mean":
                    float(
                        np.min(
                            lomo
                        )
                    ),

                "lomo_max_mean":
                    float(
                        np.max(
                            lomo
                        )
                    ),

                "all_lomo_positive":
                    bool(
                        np.all(
                            lomo > 0
                        )
                    ),
            }
        )

        for (
            month,
            d
        ) in zip(
            pair.index,
            delta
        ):
            delta_rows.append(
                {
                    "test_month":
                        month,

                    "candidate":
                        candidate,

                    "metric":
                        metric,

                    "improvement_vs_logit":
                        float(d),
                }
            )

        for i, (
            month,
            val
        ) in enumerate(
            zip(
                pair.index,
                lomo
            )
        ):
            lomo_rows.append(
                {
                    "left_out_month":
                        month,

                    "candidate":
                        candidate,

                    "metric":
                        metric,

                    "mean_improvement_without_month":
                        float(val),
                }
            )


res = pd.DataFrame(
    rows
)

res.to_csv(
    REP
    / "19a_paired_temporal_inference.csv",
    index=False
)

pd.DataFrame(
    delta_rows
).to_csv(
    REP
    / "19b_monthly_metric_deltas.csv",
    index=False
)

pd.DataFrame(
    lomo_rows
).to_csv(
    REP
    / "19c_leave_one_month_out.csv",
    index=False
)


ap_row = res[
    (
        res["candidate"]
        ==
        "GNADMM_BRIER_CONSTRAINED"
    )
    &
    (
        res["metric"]
        ==
        "average_precision"
    )
].iloc[0]


if (
    ap_row[
        "mean_improvement"
    ] > 0
    and
    ap_row[
        "block_bootstrap_ci_lower"
    ] > 0
    and
    ap_row[
        "wilcoxon_p_two_sided"
    ] < 0.05
):

    ap_evidence = (
        "SUPPORTED"
    )

elif (
    ap_row[
        "mean_improvement"
    ] > 0
    and
    ap_row[
        "months_better"
    ] >= 8
):

    ap_evidence = (
        "FAVORABLE_NOT_DECISIVE"
    )

else:

    ap_evidence = (
        "NO_CONSISTENT_ADVANTAGE"
    )


decision = {
    "status":
        "PASS",

    "primary_comparison":
        (
            "GNADMM_BRIER_CONSTRAINED "
            "vs LOGIT_MLE"
        ),

    "primary_metric":
        "average_precision",

    "mean_AP_improvement":
        float(
            ap_row[
                "mean_improvement"
            ]
        ),

    "months_AP_better":
        int(
            ap_row[
                "months_better"
            ]
        ),

    "block_bootstrap_AP_CI95":
        [
            float(
                ap_row[
                    "block_bootstrap_ci_lower"
                ]
            ),

            float(
                ap_row[
                    "block_bootstrap_ci_upper"
                ]
            ),
        ],

    "wilcoxon_AP_p_two_sided":
        float(
            ap_row[
                "wilcoxon_p_two_sided"
            ]
        ),

    "sign_test_AP_p_two_sided":
        float(
            ap_row[
                "sign_test_p_two_sided"
            ]
        ),

    "all_leave_one_month_out_AP_means_positive":
        bool(
            ap_row[
                "all_lomo_positive"
            ]
        ),

    "AP_evidence_classification":
        ap_evidence,

    "important_guardrail":
        (
            "Months are the paired evaluation units. "
            "With only 14 OOT months, uncertainty "
            "intervals and sensitivity analyses are "
            "reported rather than treating a small "
            "mean delta as definitive."
        ),
}

with open(
    REP
    / "19d_PAIRED_TEMPORAL_DECISION.json",
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
    res.to_string(
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
    "[PASS 19] PAIRED TEMPORAL "
    "INFERENCE COMPLETE."
)
