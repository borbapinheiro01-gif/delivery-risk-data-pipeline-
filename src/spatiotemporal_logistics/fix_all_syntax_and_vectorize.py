#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# 1. Corrigir a string quebrada pelo patch anterior
text = re.sub(
    r'"2\. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA\s*print\(\'[^\']*\', flush=True\)"',
    '"2. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA"',
    text
)

# 2. Localizar o bloco do laço for lento (Etapa 2) e aplicar vetorização limpa
START_MARKER = "route_rows = []"
END_MARKER = "df = df.merge(\n    route,\n    on=\"order_id\",\n    how=\"left\",\n    validate=\"1:1\"\n)"

start = text.find(START_MARKER)
end = text.find(END_MARKER, start)

if start >= 0 and end >= 0:
    vectorized_code = r'''# Vetorização nativa da rota multi-seller (Pandas Cython)
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

valid_dist = seller_order[seller_order["distance_km"].notna()].groupby("order_id")["seller_id"].nunique().reset_index()
valid_dist.columns = ["order_id", "route_sellers_with_distance"]

route = route_agg.merge(valid_dist, on="order_id", how="left")
route["route_sellers_with_distance"] = route["route_sellers_with_distance"].fillna(0).astype(int)

route["distance_freight_weighted_km"] = np.where(
    route["sum_freight"] > 0,
    route["sum_weighted_dist"] / route["sum_freight"],
    route["distance_mean_km"]
)

route = route.drop(columns=["sum_weighted_dist", "sum_freight"])

route["route_distance_coverage"] = (
    route["route_sellers_with_distance"] / route["route_sellers_total"].replace(0, np.nan)
)

'''
    text = text[:start] + vectorized_code + text[end:]

TARGET.write_text(text, encoding="utf-8")
print("[PASS] CORREÇÃO DE SINTAXE E VETORIZAÇÃO APLICADAS COM SUCESSO!")
