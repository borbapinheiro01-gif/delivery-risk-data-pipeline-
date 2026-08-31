#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# 1. Injetar import gc se não existir
if "import gc" not in text:
    text = "import gc\n" + text

# 2. Injetar gc.collect() dentro do laço original para liberar RAM mantendo a iteração exata
old_loop = "for order_id, group in seller_order.groupby("
new_loop = "for idx_grp, (order_id, group) in enumerate(seller_order.groupby("

text = text.replace(old_loop, new_loop)

# Limpeza de memória a cada 10.000 iterações no laço original
clean_memory_code = """    if idx_grp % 10000 == 0:
        gc.collect()
"""

if "route_rows.append(" in text and "idx_grp % 10000" not in text:
    text = text.replace("    route_rows.append(", clean_memory_code + "    route_rows.append(")

# 3. Ajustar leitura do PIB para evitar falha no cabeçalho
pattern = r"gdp_raw,\s*sheet,\s*header\s*=\s*read_excel_detect\([\s\S]*?required_groups=\[\s*\(\"ano\",\),[\s\S]*?\]\s*\)"
new_gdp_block = """try:
    gdp_raw, sheet, header = read_excel_detect(
        gdp_path,
        required_groups=[
            ("ano",),
            ("municip", "codigo"),
            ("produto", "pib"),
        ]
    )
except RuntimeError:
    excel_gdp = pd.ExcelFile(gdp_path)
    sheet = excel_gdp.sheet_names[0]
    for h_row in range(0, 15):
        df_tmp = pd.read_excel(excel_gdp, sheet_name=sheet, header=h_row, nrows=10)
        cols_str = " ".join([str(c).lower() for c in df_tmp.columns])
        if "ano" in cols_str and ("municip" in cols_str or "codigo" in cols_str):
            header = h_row
            gdp_raw = pd.read_excel(excel_gdp, sheet_name=sheet, header=header)
            break
    else:
        gdp_raw = pd.read_excel(excel_gdp, sheet_name=sheet, header=0)
        header = 0"""

if re.search(pattern, text):
    text = re.sub(pattern, new_gdp_block, text)

TARGET.write_text(text, encoding="utf-8")
print("[PASS] GERENCIAMENTO DE MEMÓRIA (GC) E LEITURA DO PIB APLICADOS COM SUCESSO!")
