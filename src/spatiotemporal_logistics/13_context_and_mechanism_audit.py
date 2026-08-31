#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

ROOT = Path.cwd()
ARTIFACTS_DIR = ROOT / "artifacts" / "spatiotemporal_logistics"
REPORTS_DIR = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"

# 1. Ler Decisão Não Linear do Módulo 11 para Alinhamento Adaptativo da Forma
with open(REPORTS_DIR / "11d_NONLINEAR_RELIABILITY_DECISION.json", "r") as f:
    dec_11 = json.load(f)

shape_choice = dec_11.get("classified_relationship_shape", "LINEAR_OR_INCONCLUSIVE")

if "CONVEX" in shape_choice or "QUADRATIC" in shape_choice:
    z_expr = "Z_freight_log_ratio + I(Z_freight_log_ratio**2)"
else:
    z_expr = "Z_freight_log_ratio"

# 2. Carregar Dados Mestre e Formatar
USECOLS = [
    "total_freight",
    "expected_freight_oot",
    "late_delivery_calendar_day",
    "distance_freight_weighted_km",
    "total_weight_g",
    "product_volume_sum_proxy_cm3",
    "total_price",
    "route_sellers_total",
    "any_interstate_route",
    "customer_population",
    "customer_gdp_per_capita",
    "promised_delivery_days",
    "customer_diesel_common_municipal",
    "truck_strike_2018",
]

master = pd.read_csv(
    ARTIFACTS_DIR / "06_EXPECTED_FREIGHT_OOT.csv",
    usecols=USECOLS,
)

valid_mask = (master['total_freight'] >= 0) & (master['expected_freight_oot'] > 0)
df_valid = master[valid_mask].copy()

eps = 1e-4
df_valid['Z_freight_log_ratio'] = np.log((df_valid['total_freight'] + eps) / (df_valid['expected_freight_oot'] + eps))
df_valid['Q_freight_ratio'] = df_valid['total_freight'] / df_valid['expected_freight_oot']

q10, q90 = df_valid['Q_freight_ratio'].quantile(0.10), df_valid['Q_freight_ratio'].quantile(0.90)
df_valid['residual_group'] = np.where(df_valid['Q_freight_ratio'] <= q10, 'LOW', np.where(df_valid['Q_freight_ratio'] >= q90, 'HIGH', 'NORMAL'))

# SMD Ajustado para Recursos Contínuos e Binários
smd_list = []
norm_grp = df_valid[df_valid['residual_group'] == 'NORMAL']

cont_features = ['distance_freight_weighted_km', 'total_weight_g', 'product_volume_sum_proxy_cm3', 'total_price', 'customer_population', 'promised_delivery_days']
bin_features = ['any_interstate_route']

for grp_name in ['LOW', 'HIGH']:
    comp_grp = df_valid[df_valid['residual_group'] == grp_name]
    for feat in cont_features:
        s1, s2 = comp_grp[feat].dropna(), norm_grp[feat].dropna()
        m1, m2 = s1.mean(), s2.mean()
        v1, v2 = s1.var(), s2.var()
        s_pooled = np.sqrt(((len(s1) - 1) * v1 + (len(s2) - 1) * v2) / (len(s1) + len(s2) - 2))
        smd = (m1 - m2) / s_pooled if s_pooled > 0 else np.nan
        smd_list.append({'group_vs_normal': grp_name, 'feature': feat, 'type': 'continuous', 'smd': smd})
        
    for feat in bin_features:
        p1, p2 = comp_grp[feat].mean(), norm_grp[feat].mean()
        s_pooled_bin = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        smd_bin = (p1 - p2) / s_pooled_bin if s_pooled_bin > 0 else np.nan
        smd_list.append({'group_vs_normal': grp_name, 'feature': feat, 'type': 'binary', 'smd': smd_bin})

pd.DataFrame(smd_list).to_csv(REPORTS_DIR / "13a_group_composition_smd.csv", index=False)

# 3. Cadeia de Decomposição H0 -> H4 com Amostra ANP Emparelhada
df_sub_A = df_valid.dropna(subset=['late_delivery_calendar_day', 'Z_freight_log_ratio', 'distance_freight_weighted_km', 'total_weight_g', 'product_volume_sum_proxy_cm3', 'route_sellers_total', 'promised_delivery_days', 'customer_population', 'customer_gdp_per_capita']).copy()

df_sub_A['log_distance'] = np.log1p(df_sub_A['distance_freight_weighted_km'].where(df_sub_A['distance_freight_weighted_km'] >= 0))
df_sub_A['log_weight'] = np.log1p(df_sub_A['total_weight_g'].where(df_sub_A['total_weight_g'] >= 0))
df_sub_A['log_volume'] = np.log1p(df_sub_A['product_volume_sum_proxy_cm3'].where(df_sub_A['product_volume_sum_proxy_cm3'] >= 0))
df_sub_A['log_pop'] = np.log1p(df_sub_A['customer_population'].where(df_sub_A['customer_population'] >= 0))
df_sub_A['log_gdp'] = np.log1p(df_sub_A['customer_gdp_per_capita'].where(df_sub_A['customer_gdp_per_capita'] >= 0))

# Amostra B com Cobertura ANP
df_sub_B = df_sub_A[df_sub_A['customer_diesel_common_municipal'].notna()].copy()
df_sub_B['log_diesel'] = np.log1p(df_sub_B['customer_diesel_common_municipal'].where(df_sub_B['customer_diesel_common_municipal'] >= 0))

models_seq = {
    'H0_Unadjusted': f'late_delivery_calendar_day ~ {z_expr}',
    'H1_Physical': f'late_delivery_calendar_day ~ {z_expr} + log_distance + log_weight + log_volume + route_sellers_total',
    'H2_Promise_Adjusted': f'late_delivery_calendar_day ~ {z_expr} + log_distance + log_weight + log_volume + route_sellers_total + promised_delivery_days',
    'H3_Context_Added': f'late_delivery_calendar_day ~ {z_expr} + log_distance + log_weight + log_volume + route_sellers_total + promised_delivery_days + log_pop + log_gdp',
}

seq_res = []
for name, fml in models_seq.items():
    fit = smf.logit(fml, data=df_sub_A).fit(disp=False)
    seq_res.append({'specification': name, 'sample': 'A_Full', 'llf': fit.llf, 'aic': fit.aic})

# H4 na Amostra B
fit_h4 = smf.logit(f'late_delivery_calendar_day ~ {z_expr} + log_distance + log_weight + log_volume + route_sellers_total + promised_delivery_days + log_pop + log_gdp + log_diesel + truck_strike_2018', data=df_sub_B).fit(disp=False)
seq_res.append({'specification': 'H4_ANP_Covered_Full', 'sample': 'B_ANP_Covered', 'llf': fit_h4.llf, 'aic': fit_h4.aic})

pd.DataFrame(seq_res).to_csv(REPORTS_DIR / "13b_sequential_decomposition.csv", index=False)

# 4. Heterogeneidade e Correção Benjamini-Hochberg (FDR)
df_sub_A['dist_tercile'] = pd.qcut(df_sub_A['distance_freight_weighted_km'], q=3, labels=['SHORT', 'MEDIUM', 'LONG'])
het_pvals = []
het_rows = []

for t_label, t_df in df_sub_A.groupby('dist_tercile'):
    fit_t = smf.logit(f'late_delivery_calendar_day ~ {z_expr} + log_weight + log_volume + promised_delivery_days', data=t_df).fit(disp=False)
    p_v = float(fit_t.pvalues.iloc[1])
    het_pvals.append(p_v)
    het_rows.append({'subgroup': f'Distance_{t_label}', 'p_value_raw': p_v})

rejected, p_adjusted = fdrcorrection(het_pvals, alpha=0.05)
for idx, r_item in enumerate(het_rows):
    r_item['fdr_rejected'] = bool(rejected[idx])
    r_item['p_value_fdr'] = float(p_adjusted[idx])

pd.DataFrame(het_rows).to_csv(REPORTS_DIR / "13c_heterogeneity_fdr.csv", index=False)

dec_13 = {
    "status": "PASS",
    "adapted_z_expression": z_expr,
    "sample_A_size": int(len(df_sub_A)),
    "sample_B_anp_size": int(len(df_sub_B)),
    "heterogeneity_fdr_rejected_count": int(np.sum(rejected))
}

with open(REPORTS_DIR / "13d_MECHANISM_AUDIT_DECISION.json", "w") as f:
    json.dump(dec_13, f, indent=2)

print(f"[PASS 13] MECANISMOS AUDITADOS COM ALINHAMENTO ADAPTATIVO DA FORMA. Expressão Z: {z_expr}")
