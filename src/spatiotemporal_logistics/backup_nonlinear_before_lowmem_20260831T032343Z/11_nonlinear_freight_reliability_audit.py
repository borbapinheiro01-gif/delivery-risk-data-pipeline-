#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import average_precision_score, brier_score_loss
from patsy import dmatrix

ROOT = Path.cwd()
ARTIFACTS_DIR = ROOT / "artifacts" / "spatiotemporal_logistics"
REPORTS_DIR = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"

# 1. Carregar Dados e Aplicar Trava de Validade do Ratio
master = pd.read_csv(ARTIFACTS_DIR / "06_EXPECTED_FREIGHT_OOT.csv")

valid_mask = (master['total_freight'] >= 0) & (master['expected_freight_oot'] > 0)
df_valid = master[valid_mask].copy()

eps = 1e-4
df_valid['Z_freight_log_ratio'] = np.log((df_valid['total_freight'] + eps) / (df_valid['expected_freight_oot'] + eps))
df_valid['Q_freight_ratio'] = df_valid['total_freight'] / df_valid['expected_freight_oot']

# Congelar Amostra Analítica Única
req_cols = ['late_delivery_calendar_day', 'Z_freight_log_ratio', 'distance_freight_weighted_km', 
            'total_weight_g', 'product_volume_sum_proxy_cm3', 'route_sellers_total', 
            'any_interstate_route', 'promised_delivery_days', 'year_month']

df_sample = df_valid.dropna(subset=req_cols).copy()

df_sample['log_distance'] = np.log1p(df_sample['distance_freight_weighted_km'].where(df_sample['distance_freight_weighted_km'] >= 0))
df_sample['log_weight'] = np.log1p(df_sample['total_weight_g'].where(df_sample['total_weight_g'] >= 0))
df_sample['log_volume'] = np.log1p(df_sample['product_volume_sum_proxy_cm3'].where(df_sample['product_volume_sum_proxy_cm3'] >= 0))

q10, q90 = df_sample['Q_freight_ratio'].quantile(0.10), df_sample['Q_freight_ratio'].quantile(0.90)
df_sample['cat_LOW'] = (df_sample['Q_freight_ratio'] <= q10).astype(int)
df_sample['cat_HIGH'] = (df_sample['Q_freight_ratio'] >= q90).astype(int)

# 2. Ajuste In-Sample: Famílias A (sem Promessa) e B (com Promessa)
fml_controls_A = ' + log_distance + log_weight + log_volume + route_sellers_total + any_interstate_route'
fml_controls_B = fml_controls_A + ' + promised_delivery_days'

models_specs = {
    'N1_Linear': 'Z_freight_log_ratio',
    'N2_Quadratic': 'Z_freight_log_ratio + I(Z_freight_log_ratio**2)',
    'N3_Natural_Spline_df4': 'cr(Z_freight_log_ratio, df=4)',
    'N4_Categorical': 'cat_LOW + cat_HIGH'
}

fits_A, fits_B = {}, {}
comp_rows = []

for m_name, expr in models_specs.items():
    fit_a = smf.logit(f'late_delivery_calendar_day ~ {expr}' + fml_controls_A, data=df_sample).fit(disp=False)
    fit_b = smf.logit(f'late_delivery_calendar_day ~ {expr}' + fml_controls_B, data=df_sample).fit(disp=False)
    
    fits_A[m_name] = fit_a
    fits_B[m_name] = fit_b
    
    comp_rows.append({
        'model': m_name,
        'aic_A': fit_a.aic, 'bic_A': fit_a.bic, 'llf_A': fit_a.llf,
        'aic_B': fit_b.aic, 'bic_B': fit_b.bic, 'llf_B': fit_b.llf,
        'nobs': int(fit_a.nobs)
    })

pd.DataFrame(comp_rows).to_csv(REPORTS_DIR / "11a_nonlinear_model_comparison.csv", index=False)

# Ponto Estacionário Quadrático N2_B
b1 = fits_B['N2_Quadratic'].params['Z_freight_log_ratio']
b2 = fits_B['N2_Quadratic'].params['I(Z_freight_log_ratio ** 2)']
z_stat_quad = -b1 / (2 * b2) if b2 != 0 else np.nan

# Likelihood Ratio Test N1_B vs N2_B
lr_stat = 2 * (fits_B['N2_Quadratic'].llf - fits_B['N1_Linear'].llf)
p_lr_quad = stats.chi2.sf(lr_stat, df=1)

# 3. Avaliação Preditiva Temporal Expansiva OOT
months_sorted = sorted(df_sample['year_month'].unique())
min_train_months = 3
oot_scores = []

for i in range(min_train_months, len(months_sorted)):
    train_m = months_sorted[:i]
    test_m = months_sorted[i]
    
    tr_df = df_sample[df_sample['year_month'].isin(train_m)]
    te_df = df_sample[df_sample['year_month'] == test_m]
    
    if te_df.empty or tr_df['late_delivery_calendar_day'].nunique() < 2:
        continue
        
    for m_name, expr in models_specs.items():
        m_oot = smf.logit(f'late_delivery_calendar_day ~ {expr}' + fml_controls_B, data=tr_df).fit(disp=False)
        preds = m_oot.predict(te_df)
        
        ap = average_precision_score(te_df['late_delivery_calendar_day'], preds)
        brier = brier_score_loss(te_df['late_delivery_calendar_day'], preds)
        
        oot_scores.append({'test_month': test_m, 'model': m_name, 'ap_oot': ap, 'brier_oot': brier})

oot_df = pd.DataFrame(oot_scores)
oot_df.to_csv(REPORTS_DIR / "11b_nonlinear_oot_performance.csv", index=False)

# 4. Curva de Predição Marginal Média com Intervalos de Confiança (N3_B)
z_grid = np.linspace(df_sample['Z_freight_log_ratio'].quantile(0.01), df_sample['Z_freight_log_ratio'].quantile(0.99), 50)
best_spline_fit = fits_B['N3_Natural_Spline_df4']
cov_mat = best_spline_fit.cov_params()

pred_curves = []
for z_val in z_grid:
    df_temp = df_sample.copy()
    df_temp['Z_freight_log_ratio'] = z_val
    
    # Matriz de design
    dmat = dmatrix(best_spline_fit.model.data.orig_exog.design_info, df_temp, return_type='dataframe')
    lin_pred = dmat.values @ best_spline_fit.params.values
    prob_pred = 1.0 / (1.0 + np.exp(-lin_pred))
    
    mean_prob = float(np.mean(prob_pred))
    var_lin = np.diag(dmat.values @ cov_mat.values @ dmat.values.T)
    se_prob = float(np.mean(np.sqrt(var_lin) * prob_pred * (1.0 - prob_pred)))
    
    pred_curves.append({
        'Z_grid': z_val, 'freight_ratio_equivalent': float(np.exp(z_val)),
        'predicted_late_prob_mean': mean_prob,
        'ci_95_lower': max(0.0, mean_prob - 1.96 * se_prob),
        'ci_95_upper': min(1.0, mean_prob + 1.96 * se_prob)
    })

pd.DataFrame(pred_curves).to_csv(REPORTS_DIR / "11c_spline_marginal_prediction_curve.csv", index=False)

# 5. Classificação da Relação
best_model_aic = str(pd.DataFrame(comp_rows).loc[pd.DataFrame(comp_rows)['aic_B'].idxmin(), 'model'])
shape_class = "CONVEX_U_LIKE" if (b2 > 0 and p_lr_quad < 0.01 and "Spline" in best_model_aic) else ("QUADRATIC_CURVED" if p_lr_quad < 0.05 else "LINEAR_OR_INCONCLUSIVE")

dec_11 = {
    "status": "PASS",
    "sample_n_frozen": int(len(df_sample)),
    "best_aic_model_family_B": best_model_aic,
    "quadratic_lr_p_value": float(p_lr_quad),
    "quadratic_stationary_point_z": float(z_stat_quad) if np.isfinite(z_stat_quad) else None,
    "classified_relationship_shape": shape_class
}

with open(REPORTS_DIR / "11d_NONLINEAR_RELIABILITY_DECISION.json", "w") as f:
    json.dump(dec_11, f, indent=2)

print(f"[PASS 11] AUDITORIA NÃO LINEAR CONCLUÍDA. Forma Detectada: {shape_class} | Melhor Modelo: {best_model_aic}")
