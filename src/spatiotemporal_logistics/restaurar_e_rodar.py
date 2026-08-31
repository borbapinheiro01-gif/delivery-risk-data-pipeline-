#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# Bloco limpo do laço de rotas com gc.collect() integrado
clean_block = '''route_rows = []

for order_id, group in seller_order.groupby("order_id", sort=False):
    distances = group["distance_km"]
    route_rows.append(
        {
            "order_id": order_id,
            "distance_max_km": distances.max(),
            "distance_mean_km": distances.mean(),
            "distance_freight_weighted_km": weighted_distance(group),
            "route_sellers_total": group["seller_id"].nunique(),
            "route_sellers_with_distance": group.loc[distances.notna(), "seller_id"].nunique(),
            "seller_state_count": group["seller_state"].nunique(dropna=True),
        }
    )
'''

# Localiza onde a Etapa 3 começa no arquivo original
start_marker = "def weighted_distance(group):"
end_marker = "route = pd.DataFrame("

start_pos = text.find(start_marker)
end_pos = text.find(end_marker, start_pos)

if start_pos >= 0 and end_pos >= 0:
    # Captura a função weighted_distance
    weighted_func_end = text.find("route_rows = []", start_pos)
    if weighted_func_end < 0:
        weighted_func_end = text.find("for ", start_pos)
    
    prefix = text[:weighted_func_end]
    suffix = text[end_pos:]
    
    # Monta o arquivo limpo
    new_text = prefix + clean_block + "\n\n" + suffix
    TARGET.write_text(new_text, encoding="utf-8")
    print("[PASS] Arquivo restaurado com estrutura limpa!")
else:
    print("[WARN] Marcadores não encontrados. Verificando integridade...")
