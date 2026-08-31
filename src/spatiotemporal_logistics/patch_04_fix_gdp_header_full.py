#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# Adjust GDP detection logic to safely inspect first rows without failing
old_gdp_block = """gdp_raw, sheet, header = read_excel_detect(

        gdp_path,

        required_groups=[
            ("ano",),
            ("municip",),
            (
                "produto",
                "interno",
                "bruto",
            ),
        ]
    )"""

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
    # Fallback para planilhas históricas com cabeçalhos mesclados nas primeiras linhas
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

if old_gdp_block in text:
    text = text.replace(old_gdp_block, new_gdp_block)
    TARGET.write_text(text, encoding="utf-8")
    print("[PASS] PATCH COMPLETO DO CABEÇALHO DO PIB APLICADO!")
else:
    print("[INFO] Buscando padrão por expressão regular...")
    pattern = r"gdp_raw,\s*sheet,\s*header\s*=\s*read_excel_detect\([\s\S]*?required_groups=\[\s*\(\"ano\",\),[\s\S]*?\]\s*\)"
    if re.search(pattern, text):
        text = re.sub(pattern, new_gdp_block, text)
        TARGET.write_text(text, encoding="utf-8")
        print("[PASS] PATCH COMPLETO DO CABEÇALHO DO PIB APLICADO VIA REGEX!")
    else:
        print("[WARN] Bloco exato não encontrado. Verifique a estrutura do arquivo.")
