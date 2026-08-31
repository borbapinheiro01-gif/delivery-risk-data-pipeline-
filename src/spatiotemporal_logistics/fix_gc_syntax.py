#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# 1. Garantir import gc
if "import gc" not in text:
    text = "import gc\n" + text

# 2. Corrigir a linha do groupby com fechamento exato dos parênteses
pattern_loop = r"for\s+(?:idx_grp,\s*\()?(?:order_id,\s*group\)?)\s+in\s+enumerate\(seller_order\.groupby\(\s*\"order_id\",\s*sort=False\s*\)\):|for\s+order_id,\s*group\s+in\s+seller_order\.groupby\(\s*\"order_id\",\s*sort=False\s*\):"

correct_loop = "for idx_grp, (order_id, group) in enumerate(seller_order.groupby(\"order_id\", sort=False)):"

text = re.sub(pattern_loop, correct_loop, text)

# 3. Injetar gc.collect() a cada 5.000 iterações mantendo o laço original intacto
if "if idx_grp % 5000 == 0:" not in text:
    text = text.replace(
        "    route_rows.append(",
        "    if idx_grp % 5000 == 0:\n        gc.collect()\n    route_rows.append("
    )

TARGET.write_text(text, encoding="utf-8")
print("[PASS] LAÇO DE ROTAS E GERENCIAMENTO DE MEMÓRIA CORRIGIDOS COM SUCESSO!")
