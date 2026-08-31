#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# Ajusta a busca de cabeçalho do PIB para ser flexível e aceitar variações do IBGE
old_call = 'required_groups=[\n            ("ano",),\n            ("municip",),\n            (\n                "produto",\n                "interno",\n                "bruto",\n            ),\n        ]'

new_call = 'required_groups=[\n            ("ano",),\n            ("municip", "codigo"),\n            ("produto", "pib"),\n        ]'

if old_call in text:
    text = text.replace(old_call, new_call)
    TARGET.write_text(text, encoding="utf-8")
    print("[PASS] PATCH DO CABEÇALHO DO PIB APLICADO COM SUCESSO!")
else:
    print("[INFO] Ponto de substituição flexível do PIB já ajustado.")
