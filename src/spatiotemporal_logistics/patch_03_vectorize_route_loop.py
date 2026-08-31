#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

START_MARKER = "route_rows = []"
END_MARKER = "route = pd.DataFrame(\n    route_rows\n)"

start = text.find(START_MARKER)
end = text.find(END_MARKER, start)

if start < 0 or end < 0:
    raise SystemExit("[FAIL] Marcadores do laço de rota não encontrados.")

# Substitui o laço 'for' lento por agregação vetorizada nativa do Pandas
vectorized_code = r'''# Vetorização nativa da rota multi-seller
seller_order["freight_clean"] = pd.to_numeric(seller_order["seller_order_freight"], errors="coerce").fillna(0).clip(lower=0)
seller_order["weighted_dist"] = seller_order["distance_km"] * seller_order["freight_clean"]

route_agg = seller_order.groupby("order_id", as_index=False).agg(
    distance_max_km=("distance_km", "max"),
    distance_mean_km=("distance_km", "mean"),
    sum_weighted_dist=("weighted_dist", "sum"),
    sum_freight=("freight_clean", "sum"),
    route_sellers_total=("seller_id", "nunique"),
    seller_state_count=("seller_state", lambda x: x.dropna().nunique())
)

# Cálculo de vendedores com distância válida vetorizado
valid_dist = seller_order[seller_order["distance_km"].notna()].groupby("order_id")["seller_id"].nunique().reset_index()
valid_dist.columns = ["order_id", "route_sellers_with_distance"]

route = route_agg.merge(valid_dist, on="order_id", how="left")
route["route_sellers_with_distance"] = route["route_sellers_with_distance"].fillna(0).astype(int)

# Cálculo da distância média ponderada pelo frete vetorizada
route["distance_freight_weighted_km"] = np.where(
    route["sum_freight"] > 0,
    route["sum_weighted_dist"] / route["sum_freight"],
    route["distance_mean_km"]
)

route = route.drop(columns=["sum_weighted_dist", "sum_freight"])'''

new_text = text[:start] + vectorized_code + text[end + len(END_MARKER):]
TARGET.write_text(new_text, encoding="utf-8")
print("[PASS] LAÇO DE ROTA VETORIZADO COM SUCESSO")
