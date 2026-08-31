#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
ART = ROOT / "artifacts" / "spatiotemporal_logistics"
REP = ROOT / "reports" / "spatiotemporal_logistics"
REG = ROOT / "data" / "external" / "registry"

master = pd.read_csv(
    ART / "06_EXPECTED_FREIGHT_OOT.csv",
    dtype={
        "customer_id_municipio": "string",
    },
)

rso = pd.read_csv(
    ART / "01_ROUTE_SELLER_ORDER.csv"
)

muni = pd.read_csv(
    ART / "03_MUNICIPAL_CONTEXT.csv",
    dtype={
        "id_municipio": "string",
    },
)


# G01
g01 = not master[
    "order_id"
].duplicated().any()


# G02
g02 = master[
    "year"
].isin(
    [2017, 2018]
).all()


# G03
g03 = not rso.duplicated(
    [
        "order_id",
        "seller_id",
    ]
).any()


# G04
g04 = (
    rso[
        "great_circle_distance_km"
    ]
    .dropna()
    .ge(0)
    .all()
)


# G05 — missing is allowed; impossible nonmissing coordinates are not.
seller_lat_ok = (
    rso["seller_lat"]
    .dropna()
    .between(-35.5, 6.5)
    .all()
)

customer_lat_ok = (
    rso["customer_lat"]
    .dropna()
    .between(-35.5, 6.5)
    .all()
)

seller_lon_ok = (
    rso["seller_lon"]
    .dropna()
    .between(-75.5, -32.0)
    .all()
)

customer_lon_ok = (
    rso["customer_lon"]
    .dropna()
    .between(-75.5, -32.0)
    .all()
)

g05 = bool(
    seller_lat_ok
    and
    customer_lat_ok
    and
    seller_lon_ok
    and
    customer_lon_ok
)


# G06
g06 = (
    master[
        "total_freight"
    ]
    .dropna()
    .ge(0)
    .all()
)


# G07
g07 = (
    muni[
        "population"
    ]
    .dropna()
    .gt(0)
    .all()
)


# G08 / G09
g08 = (
    len(
        muni[
            muni["year"].eq(2017)
        ]
    )
    ==
    5570
)

g09 = (
    len(
        muni[
            muni["year"].eq(2018)
        ]
    )
    ==
    5570
)


# G10
g10 = not muni.duplicated(
    [
        "year",
        "id_municipio",
    ]
).any()


# G11 — state fallback is separate;
# municipal missing remains actually missing.
municipal_cols = [
    "customer_diesel_common_municipal",
    "customer_diesel_s10_municipal",
]

missing_rows = (
    master[
        "customer_diesel_geo_level"
    ]
    .isin(
        [
            "STATE_ONLY",
            "MISSING",
        ]
    )
)

g11 = bool(
    master.loc[
        missing_rows,
        municipal_cols,
    ]
    .isna()
    .all(axis=None)
    and
    (
        master[
            municipal_cols
        ]
        .stack()
        .gt(0)
        .all()
    )
)


# G12 — GDP registry must explicitly be retrospective.
temporal_path = (
    REG / "temporal_eligibility.csv"
)

g12 = False

if temporal_path.exists():

    temporal = pd.read_csv(
        temporal_path
    )

    row = temporal[
        temporal["source_id"]
        .eq(
            "IBGE_GDP_2010_2018"
        )
    ]

    if not row.empty:

        g12 = (
            row[
                "pit_status"
            ]
            .astype(str)
            .str.upper()
            .eq(
                "RETROSPECTIVE_ONLY"
            )
            .all()
        )


# G13 — residual OOT exists, availability flag is coherent,
# and at least one early segment remains unavailable.
oot_na = master[
    "freight_residual_OOT"
].isna()

oot_flag_na = master[
    "freight_oot_available"
].eq(0)

g13 = bool(
    master[
        "freight_residual_OOT"
    ]
    .notna()
    .any()
    and
    oot_na.equals(
        oot_flag_na
    )
    and
    oot_na.any()
)


# G14 — exact RAW SHA256 before/after.
before = REP / "raw_before_v21.sha256"
after = REP / "raw_after_v21.sha256"

g14 = (
    before.exists()
    and
    after.exists()
    and
    before.read_text(
        encoding="utf-8"
    )
    ==
    after.read_text(
        encoding="utf-8"
    )
)


required_outputs = [
    "06_cost_descriptive.csv",
    "07a_cost_pearson.csv",
    "07b_cost_spearman.csv",
    "09_freight_model_comparison.csv",
    "11a_speed_nested_in_sample.csv",
    "11b_speed_nested_oot.csv",
    "13a_reliability_inference.csv",
    "13b_reliability_oot_metrics.csv",
    "15_sensitivity_summary.csv",
]

g15 = all(
    (REP / x).exists()
    for x in required_outputs
)


gates = {
    "G01_order_id_unique": bool(g01),
    "G02_year_2017_2018_only": bool(g02),
    "G03_order_seller_unique": bool(g03),
    "G04_nonnegative_distance": bool(g04),
    "G05_coordinates_valid_if_observed": bool(g05),
    "G06_nonnegative_freight": bool(g06),
    "G07_population_positive": bool(g07),
    "G08_ibge_2017_exact_5570": bool(g08),
    "G09_ibge_2018_exact_5570": bool(g09),
    "G10_municipality_year_unique": bool(g10),
    "G11_anp_missing_preserved": bool(g11),
    "G12_gdp_retrospective_registry": bool(g12),
    "G13_residual_oot_integrity": bool(g13),
    "G14_raw_sha256_unchanged": bool(g14),
    "G15_analysis_outputs_exist": bool(g15),
}

status = (
    "PASS"
    if all(
        gates.values()
    )
    else
    "FAIL"
)

coverage = pd.read_csv(
    REP / "02_join_coverage.csv"
).to_dict(
    orient="records"
)

summary = {
    "status": status,
    "scope":
        "SPATIOTEMPORAL_LOGISTICS_AUDIT_V2_1_STAGED",
    "total_orders":
        int(len(master)),
    "order_seller_rows":
        int(len(rso)),
    "freight_oot_orders":
        int(
            master[
                "freight_oot_available"
            ].sum()
        ),
    "gates": gates,
    "coverage": coverage,
}

with open(
    REP / "SPATIOTEMPORAL_LOGISTICS_SUMMARY.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

lines = [
    "=" * 100,
    "SPATIOTEMPORAL LOGISTICS AUDIT V2.1",
    "=" * 100,
    "",
    f"STATUS                    : {status}",
    f"ORDERS                    : {len(master):,}",
    f"ORDER × SELLER            : {len(rso):,}",
    f"OOT FREIGHT ORDERS        : {int(master['freight_oot_available'].sum()):,}",
    "",
    "GATES",
    "-" * 100,
]

for k, v in gates.items():
    lines.append(
        f"{k:<45} : {'PASS' if v else 'FAIL'}"
    )

lines.extend(
    [
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 100,
        "Great-circle distance is a geographic proxy, not road-network distance.",
        "GDP 2017/2018 is retrospective context, not point-in-time predictive information.",
        "Municipal ANP missing values were not replaced by state values.",
        "State diesel context is retained as a separate variable.",
        "Freight residual used downstream is generated out-of-time only.",
        "Speed predictive metrics are out-of-time; full-sample R² is descriptive/inferential.",
        "Reliability AP/AUC/Brier metrics are out-of-time.",
        "",
        "=" * 100,
        f"FINAL STATUS = {status}",
        "=" * 100,
    ]
)

report = "\n".join(lines)

(
    REP
    / "SPATIOTEMPORAL_LOGISTICS_REPORT.txt"
).write_text(
    report,
    encoding="utf-8",
)

print(report)

sys.exit(
    0
    if status == "PASS"
    else 1
)
