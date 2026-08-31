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

# 1. Carregar Dados Mestre
master = pd.read_csv(ARTIFACTS_DIR / "06_EXPECTED_FREIGHT_OOT.csv")

rel_df = master[master['actual_delivery_days'].notna() & master['promised_delivery_days'].notna()].copy()
rel_df['log_distance'] = np.log1p(rel_df['distance_freight_weighted_km'].where(rel_df['distance_freight_weighted_km'] >= 0))
rel_df['log_weight'] = np.log1p(rel_df['total_weight_g'].where(rel_df['total_weight_g'] >= 0))
rel_df['log_volume'] = np.log1p(rel_df['product_volume_sum_proxy_cm3'].where(rel_df['product_volume_sum_proxy_cm3'] >= 0))

rel_df['freight_ratio_Q'] = np.where(rel_df['expected_freight_oot'] > 0, rel_df['total_freight'] / rel_df['expected_freight_oot'], np.nan)

# 2. Diagnóstico Bayesiano Conjugado Beta-Binomial
q10 = rel_df['freight_ratio_Q'].quantile(0.10)
q90 = rel_df['freight_ratio_Q'].quantile(0.90)

rel_df['residual_group'] = np.where(
    rel_df['freight_ratio_Q'] <= q10, 'LOW_FREIGHT',
    np.where(rel_df['freight_ratio_Q'] >= q90, 'HIGH_FREIGHT', 'NORMAL_FREIGHT')
)

beta_binom_dict = {}
alpha_prior, beta_prior = 1.0, 1.0  # Beta(1,1) Não-Informativo

beta_binom_list = []
for grp_name, grp_data in rel_df.groupby('residual_group'):
    n_total = len(grp_data)
    k_late = int(grp_data['late_delivery_calendar_day'].sum())
    
    alpha_post = alpha_prior + k_late
    beta_post = beta_prior + n_total - k_late
    
    post_mean = alpha_post / (alpha_post + beta_post)
    ci_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
    ci_upper = stats.beta.ppf(0.975, alpha_post, beta_post)
    
    beta_binom_dict[grp_name] = (alpha_post, beta_post)
    
    beta_binom_list.append({
        'group': grp_name, 'N': n_total, 'late_count': k_late,
        'observed_rate': k_late / n_total,
        'posterior_beta_mean': post_mean,
        'ci_95_lower': ci_lower, 'ci_95_upper': ci_upper
    })

beta_binom_df = pd.DataFrame(beta_binom_list)
beta_binom_df.to_csv(REPORTS_DIR / "09c_beta_binomial_residual_groups.csv", index=False)

# Comparação Probabilística Posterior via Monte Carlo (100.000 draws)
np.random.seed(42)
draws_low = stats.beta.rvs(beta_binom_dict['LOW_FREIGHT'][0], beta_binom_dict['LOW_FREIGHT'][1], size=100000)
draws_normal = stats.beta.rvs(beta_binom_dict['NORMAL_FREIGHT'][0], beta_binom_dict['NORMAL_FREIGHT'][1], size=100000)
draws_high = stats.beta.rvs(beta_binom_dict['HIGH_FREIGHT'][0], beta_binom_dict['HIGH_FREIGHT'][1], size=100000)

p_low_gt_normal = float(np.mean(draws_low > draws_normal))
p_high_gt_normal = float(np.mean(draws_high > draws_normal))

# 3. Modelos Frequentistas Logísticos R-A (Observed) vs R-B (Promise-Adjusted)
predictors = ['log_distance', 'log_weight', 'log_volume', 'freight_residual_OOT']
rel_std = rel_df.dropna(subset=predictors + ['promised_delivery_days', 'late_delivery_calendar_day']).copy()

for col in predictors:
    rel_std[col + '_std'] = (rel_std[col] - rel_std[col].mean()) / rel_std[col].std()

fit_ra = smf.logit('late_delivery_calendar_day ~ log_distance_std + log_weight_std + log_volume_std + freight_residual_OOT_std', data=rel_std).fit(disp=False)
fit_rb = smf.logit('late_delivery_calendar_day ~ log_distance_std + log_weight_std + log_volume_std + freight_residual_OOT_std + promised_delivery_days', data=rel_std).fit(disp=False)

or_rows = []
for term in ['log_distance_std', 'log_weight_std', 'log_volume_std', 'freight_residual_OOT_std']:
    coef_a, se_a = fit_ra.params[term], fit_ra.bse[term]
    coef_b, se_b = fit_rb.params[term], fit_rb.bse[term]
    
    or_rows.append({
        'predictor': term,
        'OR_R_A_observed': float(np.exp(coef_a)),
        'OR_R_A_ci_lower': float(np.exp(coef_a - 1.96 * se_a)),
        'OR_R_A_ci_upper': float(np.exp(coef_a + 1.96 * se_a)),
        'OR_R_B_promise_adjusted': float(np.exp(coef_b)),
        'OR_R_B_ci_lower': float(np.exp(coef_b - 1.96 * se_b)),
        'OR_R_B_ci_upper': float(np.exp(coef_b + 1.96 * se_b))
    })

or_df = pd.DataFrame(or_rows)
or_df.to_csv(REPORTS_DIR / "09h_frequentist_odds_ratios.csv", index=False)

# Decisão Científica
rel_decision = {
    "status": "PASS",
    "P_p_low_gt_p_normal": p_low_gt_normal,
    "P_p_high_gt_p_normal": p_high_gt_normal,
    "odds_ratio_residual_observed_R_A": float(or_df.loc[or_df['predictor'] == 'freight_residual_OOT_std', 'OR_R_A_observed'].values[0]),
    "odds_ratio_residual_promise_adjusted_R_B": float(or_df.loc[or_df['predictor'] == 'freight_residual_OOT_std', 'OR_R_B_promise_adjusted'].values[0])
}

with open(REPORTS_DIR / "09m_RELIABILITY_DECISION.json", "w") as f:
    json.dump(rel_decision, f, indent=2)

print(f"[PASS 09] AUDITORIA DE CONFIABILIDADE E BETA-BINOMIAL CONCLUÍDAS. P(p_low > p_normal): {p_low_gt_normal:.4f}")
