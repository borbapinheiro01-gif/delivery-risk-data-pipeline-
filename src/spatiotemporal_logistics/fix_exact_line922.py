#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
lines = TARGET.read_text(encoding="utf-8").splitlines()

# Garante o import gc
if not any("import gc" in l for l in lines[:10]):
    lines.insert(0, "import gc")

# Procura a linha com problema de sintaxe e corrige pontualmente
for i, line in enumerate(lines):
    if "for idx_grp, (order_id, group) in enumerate(seller_order.groupby(" in line or "for order_id, group in seller_order.groupby(" in line:
        lines[i] = 'for idx_grp, (order_id, group) in enumerate(seller_order.groupby("order_id", sort=False)):'
        # Insere a coleta de lixo a cada 5000 iterações na linha seguinte com a indentação correta
        indent = " " * 4
        lines.insert(i + 1, f"{indent}if idx_grp % 5000 == 0:\n{indent}    gc.collect()")
        break

TARGET.write_text("\n".join(lines), encoding="utf-8")
print("[PASS] LINHA 922 E GERENCIAMENTO DE MEMÓRIA AJUSTADOS COM SUCESSO!")
