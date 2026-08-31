#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path.cwd()
ARTIFACTS_DIR = ROOT / "artifacts" / "spatiotemporal_logistics"
REPORTS_DIR = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"

# 1. Carregar Decisão do Módulo 07 (Gate Bloqueante)
with open(REPORTS_DIR / "07i_RESIDUAL_DECISION.json", "r") as f:
    dec_07 = json.load(f)

if dec_07["residual_status"] == "NOT_VALID_FOR_DOWNSTREAM":
    print("[ERROR] Módulo 07 concluiu que o residual NÃO é válido para downstream! Interrompendo pipeline.")
    sys.exit(1)

# 2. Carregar Dados Mestre com Residual OOT
master = pd.read_csv(ARTIFACTS_DIR / "06_EXPECTED_FREIGHT_OOT.csv")

speed_df = master[master['actual_delivery_days'].notna() & master['distance_freight_weighted_km'].notna()].copy()
speed_df['log_speed'] = np.log1p(speed_df['actual_delivery_days'].where(speed_df['actual_delivery_days'] >= 0))
speed_df['log_distance'] = np.log1p(speed_df['distance_freight_weighted_km'].where(speed_df['distance_freight_weighted_km'] >= 0))
speed_df['log_weight'] = np.log1p(speed_df['total_weight_g'].where(speed_df['total_weight_g'] >= 0))
speed_df['log_volume'] = np.log1p(speed_df['product_volume_sum_proxy_cm3'].where(speed_df['product_volume_sum_proxy_cm3'] >= 0))
speed_df['log_freight'] = np.log1p(speed_df['total_freight'].where(speed_df['total_freight'] >= 0))

den = speed_df['total_price'] + speed_df['total_freight']
speed_df['freight_burden'] = np.where(den > 0, speed_df['total_freight'] / den, np.nan)
speed_df['log_pop'] = np.log1p(speed_df['customer_population'].where(speed_df['customer_population'] >= 0))
speed_df['log_gdp'] = np.log1p(speed_df['customer_gdp_per_capita'].where(speed_df['customer_gdp_per_capita'] >= 0))

# 3. Modelos Múltiplos Hierárquicos Frequentistas (MixedLM por UF)
predictors = ['log_distance', 'log_weight', 'log_volume', 'log_freight', 'freight_burden', 'freight_residual_OOT', 'log_pop', 'log_gdp']
speed_std = speed_df.dropna(subset=predictors + ['customer_state', 'year_month', 'log_speed']).copy()

for col in predictors:
    speed_std[col + '_std'] = (speed_std[col] - speed_std[col].mean()) / speed_std[col].std()

fml_mixed = 'log_speed ~ log_distance_std + log_weight_std + log_volume_std + log_freight_std + freight_burden_std + freight_residual_OOT_std + log_pop_std + log_gdp_std + truck_strike_2018'

# Fit Frequentista Misto com Grupo = UF
mix_mod = smf.mixedlm(fml_mixed, speed_std, groups=speed_std['customer_state']).fit()

# Registro de Coeficientes e Intervalos de Confiança (Aproximação Frequentista)
multilevel_res = []
for term in mix_mod.params.index:
    if term not in ['Group Var']:
        val = mix_mod.params[term]
        se = mix_mod.bse[term]
        multilevel_res.append({
            'term': term,
            'mixed_effects_coef': val,
            'std_err': se,
            'conf_int_lower_95': val - 1.96 * se,
            'conf_int_upper_95': val + 1.96 * se,
            'z_stat': val / se if se > 0 else np.nan
        })

pd.DataFrame(multilevel_res).to_csv(REPORTS_DIR / "08d_speed_multilevel_coefficients.csv", index=False)

# Efeitos Aleatórios por UF
random_effects = pd.DataFrame.from_dict(mix_mod.random_effects, orient='index').reset_index()
random_effects.columns = ['customer_state', 'random_intercept_uf']
random_effects.to_csv(REPORTS_DIR / "08e_speed_multilevel_group_effects.csv", index=False)

# Decisão Científica do Módulo 08
coef_residual = [r for r in multilevel_res if r['term'] == 'freight_residual_OOT_std'][0]
speed_decision = {
    "status": "PASS",
    "multilevel_model_type": "Frequentista Linear MixedLM (groups=customer_state)",
    "freight_residual_coef": coef_residual['mixed_effects_coef'],
    "freight_residual_ci_95": [coef_residual['conf_int_lower_95'], coef_residual['conf_int_upper_95']],
    "n_groups_uf": len(random_effects)
}

with open(REPORTS_DIR / "08h_SPEED_SCIENTIFIC_DECISION.json", "w") as f:
    json.dump(speed_decision, f, indent=2)

print(f"[PASS 08] MODELO MULTINÍVEL DE VELOCIDADE CONCLUÍDO. Coeficiente Residual: {coef_residual['mixed_effects_coef']:.4f}")
