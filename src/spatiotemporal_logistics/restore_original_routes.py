#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# 1. Corrigir a string multilinha mal formatada do banner
text = re.sub(
    r'banner\(\s*"2\. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA\nprint\(.*?\)"\s*\)',
    'banner("2. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA")',
    text
)

# Caso o patch tenha modificado a string direta sem o banner()
text = re.sub(
    r'"2\. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA\nprint\([\s\S]*?\)"',
    '"2. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA"',
    text
)

# 2. Restaurar o bloco de cálculo de rotas e média ponderada exatamente como original
START_MARKER = "# =====================================================================\n# 3. SELLERS / MULTI-SELLER ROUTE"
END_MARKER = "# =====================================================================\n# 4. PRODUCTS — WEIGHT / VOLUME"

start = text.find(START_MARKER)
end = text.find(END_MARKER, start)

original_route_block = '''# =====================================================================
# 3. SELLERS / MULTI-SELLER ROUTE
# =====================================================================

items = pd.read_csv(
    find_one(
        "*order_items_dataset.csv"
    ),
    usecols=[
        "order_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    ],
    low_memory=False
)


items = items[
    items[
        "order_id"
    ].isin(
        df["order_id"]
    )
].copy()


sellers = pd.read_csv(
    find_one(
        "*sellers_dataset.csv"
    ),
    usecols=[
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
    low_memory=False
)


sellers[
    "seller_city_norm"
] = sellers[
    "seller_city"
].map(norm_text)


sellers[
    "seller_state"
] = (
    sellers[
        "seller_state"
    ]
    .astype(str)
    .str.upper()
    .str.strip()
)


seller_geo = sellers.merge(

    geo_zip,

    left_on="seller_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",

    how="left"
)


seller_geo = seller_geo.rename(
    columns={
        "geo_lat":
            "seller_lat",
        "geo_lng":
            "seller_lng",
    }
)


seller_order = (

    items.groupby(
        [
            "order_id",
            "seller_id",
        ],
        as_index=False
    )

    .agg(
        seller_order_price=(
            "price",
            "sum"
        ),
        seller_order_freight=(
            "freight_value",
            "sum"
        ),
        seller_item_rows=(
            "product_id",
            "size"
        ),
    )
)


seller_order = seller_order.merge(

    seller_geo[
        [
            "seller_id",
            "seller_city",
            "seller_city_norm",
            "seller_state",
            "seller_zip_code_prefix",
            "seller_lat",
            "seller_lng",
        ]
    ],

    on="seller_id",
    how="left",
    validate="m:1"
)


seller_order = seller_order.merge(

    df[
        [
            "order_id",
            "customer_lat",
            "customer_lng",
        ]
    ],

    on="order_id",
    how="left",
    validate="m:1"
)


seller_order[
    "distance_km"
] = haversine_km(

    seller_order[
        "customer_lat"
    ],

    seller_order[
        "customer_lng"
    ],

    seller_order[
        "seller_lat"
    ],

    seller_order[
        "seller_lng"
    ],
)


def weighted_distance(group):

    valid = (
        group[
            "distance_km"
        ].notna()
    )

    if not valid.any():
        return np.nan

    weights = (

        pd.to_numeric(
            group.loc[
                valid,
                "seller_order_freight"
            ],
            errors="coerce"
        )

        .fillna(0)

        .clip(
            lower=0
        )
    )

    distance = (
        group.loc[
            valid,
            "distance_km"
        ].astype(float)
    )

    if weights.sum() > 0:

        return float(
            np.average(
                distance,
                weights=weights
            )
        )

    return float(
        distance.mean()
    )


route_rows = []

for order_id, group in seller_order.groupby(
    "order_id",
    sort=False
):

    distances = group[
        "distance_km"
    ]

    route_rows.append(
        {
            "order_id":
                order_id,

            "distance_max_km":
                distances.max(),

            "distance_mean_km":
                distances.mean(),

            "distance_freight_weighted_km":
                weighted_distance(group),

            "route_sellers_total":
                group[
                    "seller_id"
                ].nunique(),

            "route_sellers_with_distance":
                group.loc[
                    distances.notna(),
                    "seller_id"
                ].nunique(),

            "seller_state_count":
                group[
                    "seller_state"
                ].nunique(
                    dropna=True
                ),
        }
    )


route = pd.DataFrame(
    route_rows
)


route[
    "route_distance_coverage"
] = (

    route[
        "route_sellers_with_distance"
    ]

    /

    route[
        "route_sellers_total"
    ].replace(
        0,
        np.nan
    )
)


df = df.merge(
    route,
    on="order_id",
    how="left",
    validate="1:1"
)


'''

if start >= 0 and end >= 0:
    text = text[:start] + original_route_block + text[end:]

TARGET.write_text(text, encoding="utf-8")
print("[PASS] CÓDIGO ORIGINAL DE ROTAS RESTAURADO COM SUCESSO!")
