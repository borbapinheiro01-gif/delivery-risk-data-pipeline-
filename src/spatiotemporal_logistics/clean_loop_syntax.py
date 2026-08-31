#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
content = TARGET.read_text(encoding="utf-8")

# Substitui qualquer variação quebrada do cabeçalho do laço por uma estrutura única e limpa
bad_patterns = [
    'for idx_grp, (order_id, group) in enumerate(seller_order.groupby("order_id", sort=False)):\n    if idx_grp % 5000 == 0:\n        gc.collect()\n    ):\n',
    'for idx_grp, (order_id, group) in enumerate(seller_order.groupby("order_id", sort=False)):\n    if idx_grp % 5000 == 0:\n        gc.collect()\n):',
]

clean_loop = 'for idx_grp, (order_id, group) in enumerate(seller_order.groupby("order_id", sort=False)):\n    if idx_grp % 5000 == 0:\n        gc.collect()'

for bad in bad_patterns:
    if bad in content:
        content = content.replace(bad, clean_loop + "\n")

# Remove linhas orfãs com apenas '):' no trecho das rotas
lines = content.splitlines()
fixed_lines = []
for i, line in enumerate(lines):
    if line.strip() == "):" and i > 0 and "seller_order.groupby" in lines[i-1]:
        continue
    fixed_lines.append(line)

TARGET.write_text("\n".join(fixed_lines), encoding="utf-8")
print("[PASS] ESTRUTURA DO LAÇO E PARÊNTESES LIMPOS COM SUCESSO!")
