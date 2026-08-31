#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT = Path.cwd()
ARTIFACTS_DIR = ROOT / "artifacts" / "spatiotemporal_logistics"
REPORTS_DIR = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"

# 1. Carregar artefatos
rso = pd.read_csv(ARTIFACTS_DIR / "01_ROUTE_SELLER_ORDER.csv")
oot_master = pd.read_csv(ARTIFACTS_DIR / "06_EXPECTED_FREIGHT_OOT.csv")

# 2. Avaliação de Métricas Globais OOT de F2
valid_res = oot_master[oot_master['freight_residual_OOT'].notna()].copy()

# Materializar Formalmente as três versões do residual
valid_res['residual_abs'] = valid_res['freight_residual_OOT']
valid_res['residual_rel'] = np.where(valid_res['expected_freight_oot'] > 0, valid_res['residual_abs'] / valid_res['expected_freight_oot'], np.nan)
valid_res['freight_ratio_Q'] = np.where(valid_res['expected_freight_oot'] > 0, valid_res['total_freight'] / valid_res['expected_freight_oot'], np.nan)

mae_f2 = mean_absolute_error(valid_res['total_freight'], valid_res['expected_freight_oot'])
rmse_f2 = root_mean_squared_error(valid_res['total_freight'], valid_res['expected_freight_oot'])
r2_f2 = r2_score(valid_res['total_freight'], valid_res['expected_freight_oot'])

# Leitura/preservação da comparação F0 vs F1 vs F2 se disponível do Stage 04
stage4_comp_path = ROOT / "reports" / "spatiotemporal_logistics" / "09_freight_model_comparison.csv"
if stage4_comp_path.exists():
    f0_f1_comp = pd.read_csv(stage4_comp_path)
    f0_f1_comp.to_csv(REPORTS_DIR / "07a_freight_model_global_metrics.csv", index=False)
else:
    global_metrics = pd.DataFrame([{'model': 'F2_HistGradientBoosting_OOT', 'MAE': mae_f2, 'RMSE': rmse_f2, 'R2': r2_f2, 'nobs': len(valid_res)}])
    global_metrics.to_csv(REPORTS_DIR / "07a_freight_model_global_metrics.csv", index=False)

# 3. Métricas Mensais OOT e Block Bootstrap sobre Meses
monthly_list = []
for ym, grp in valid_res.groupby('year_month'):
    monthly_list.append({
        'year_month': ym,
        'MAE_F2': mean_absolute_error(grp['total_freight'], grp['expected_freight_oot']),
        'RMSE_F2': root_mean_squared_error(grp['total_freight'], grp['expected_freight_oot']),
        'R2_F2': r2_score(grp['total_freight'], grp['expected_freight_oot']),
        'nobs': len(grp)
    })
monthly_df = pd.DataFrame(monthly_list)
monthly_df.to_csv(REPORTS_DIR / "07b_freight_model_monthly_metrics.csv", index=False)

# Block Bootstrap sobre meses (B=1000) para resiliência temporal
np.random.seed(42)
months_arr = monthly_df['MAE_F2'].values
b_mae_means = [np.mean(np.random.choice(months_arr, size=len(months_arr), replace=True)) for _ in range(1000)]
bootstrap_df = pd.DataFrame([{
    'metric': 'MAE_F2_Monthly_Mean',
    'mean': np.mean(b_mae_means),
    'ci_2.5': np.percentile(b_mae_means, 2.5),
    'ci_97.5': np.percentile(b_mae_means, 97.5)
}])
bootstrap_df.to_csv(REPORTS_DIR / "07d_freight_block_bootstrap.csv", index=False)

# 4. Calibração por Decis da Previsão
valid_res['pred_decile'] = pd.qcut(valid_res['expected_freight_oot'], q=10, labels=False, duplicates='drop')
calib = valid_res.groupby('pred_decile').agg(
    median_observed=('total_freight', 'median'),
    median_predicted=('expected_freight_oot', 'median'),
    calibration_ratio=('total_freight', lambda x: x.sum() / valid_res.loc[x.index, 'expected_freight_oot'].sum())
).reset_index()
calib.to_csv(REPORTS_DIR / "07e_freight_calibration.csv", index=False)

# 5. Correlação Estrutural (Pearson e Spearman) para Testar Desacoplamento do Residual
assoc_res = []
for col in ['distance_freight_weighted_km', 'total_weight_g', 'product_volume_sum_proxy_cm3', 'total_price']:
    sub = valid_res[['residual_abs', col]].dropna()
    assoc_res.append({
        'var': col,
        'pearson_r': sub['residual_abs'].corr(sub[col], method='pearson'),
        'spearman_rho': sub['residual_abs'].corr(sub[col], method='spearman')
    })
res_corr = pd.DataFrame(assoc_res)
res_corr.to_csv(REPORTS_DIR / "07f_residual_structural_associations.csv", index=False)

# 6. Drift Temporal do Residual
drift = valid_res.groupby('year_month').agg(
    residual_mean=('residual_abs', 'mean'),
    residual_median=('residual_abs', 'median'),
    residual_std=('residual_abs', 'std')
).reset_index()
drift.to_csv(REPORTS_DIR / "07g_residual_temporal_drift.csv", index=False)

# 7. Decisão Científica Baseada em Evidência Relativa
max_spearman = float(res_corr['spearman_rho'].abs().max())
is_valid_downstream = bool(r2_f2 > 0.25 and max_spearman < 0.45)

decision = {
    "residual_status": "VALID_FOR_DOWNSTREAM" if is_valid_downstream else "NOT_VALID_FOR_DOWNSTREAM",
    "r2_oot_global": r2_f2,
    "max_residual_spearman_correlation": max_spearman,
    "n_valid_oot_orders": len(valid_res)
}

with open(REPORTS_DIR / "07i_RESIDUAL_DECISION.json", "w") as f:
    json.dump(decision, f, indent=2)

print(f"[PASS 07] MODELO DE FRETE OOT AUDITADO. DECISÃO: {decision['residual_status']}")
