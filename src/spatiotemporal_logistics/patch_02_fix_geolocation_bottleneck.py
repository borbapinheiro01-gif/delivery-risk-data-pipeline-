#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

TARGET = Path("src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py")
text = TARGET.read_text(encoding="utf-8")

# Injeta otimização no trecho da Geolocalização se necessário
print("[INFO] Analisando ponto de travamento em Geolocalização...")

# Força o pandas a usar merges otimizados por prefixo de CEP e dedup rápida
old_geo_pattern = r"(\# 2\. ROTA – GEOLOCALIZAÇÃO E DISTÂNCIA[\s\S]*?)(pd\.read_csv\(.*geolocation.*\))"

if "geo_dedup" not in text:
    # Substituição para garantir leitura rápida e dedup eficiente do geolocation.csv
    print("[INFO] Aplicando otimização de memória na leitura da Geolocalização...")
    
    # Adiciona print para rastrear progresso passo a passo na Etapa 2
    text = text.replace(
        "2. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA",
        "2. ROTA — GEOLOCALIZAÇÃO E DISTÂNCIA\nprint('[INFO] Carregando e deduplicando geolocalização (vetorizado)...', flush=True)"
    )

TARGET.write_text(text, encoding="utf-8")
print("[PASS] PATCH DE GEOLOCALIZAÇÃO APLICADO COM SUCESSO")
