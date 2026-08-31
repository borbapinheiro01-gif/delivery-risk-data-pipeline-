#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path.cwd()
ARTIFACTS_DIR = ROOT / "artifacts" / "spatiotemporal_logistics"
REPORTS_DIR = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"

master = pd.read_csv(ARTIFACTS_DIR / "06_EXPECTED_FREIGHT_OOT.csv")

valid_mask = (master['total_freight'] >= 0) & (master['expected_freight_oot'] > 0)
df_valid = master[valid_mask].copy()

eps = 1e-4
df_valid['Z_freight_log_ratio'] = np.log((df_valid['total_freight'] + eps) / (df_valid['expected_freight_oot'] + eps))

speed_df = df_valid[df_valid['actual_delivery_days'].notna() & df_valid['distance_freight_weighted_km'].notna()].copy()
speed_df['log_speed'] = np.log1p(speed_df['actual_delivery_days'].where(speed_df['actual_delivery_days'] >= 0))
speed_df['log_distance'] = np.log1p(speed_df['distance_freight_weighted_km'].where(speed_df['distance_freight_weighted_km'] >= 0))
speed_df['log_weight'] = np.log1p(speed_df['total_weight_g'].where(speed_df['total_weight_g'] >= 0))
speed_df['log_volume'] = np.log1p(speed_df['product_volume_sum_proxy_cm3'].where(speed_df['product_volume_sum_proxy_cm3'] >= 0))
speed_df['log_freight'] = np.log1p(speed_df['total_freight'].where(speed_df['total_freight'] >= 0))

den = speed_df['total_price'] + speed_df['total_freight']
speed_df['freight_burden'] = np.where(den > 0, speed_df['total_freight'] / den, np.nan)
speed_df['log_pop'] = np.log1p(speed_df['customer_population'].where(speed_df['customer_population'] >= 0))
speed_df['log_gdp'] = np.log1p(speed_df['customer_gdp_per_capita'].where(speed_df['customer_gdp_per_capita'] >= 0))

predictors = ['log_distance', 'log_weight', 'log_volume', 'log_freight', 'freight_burden', 'Z_freight_log_ratio', 'log_pop', 'log_gdp']
speed_std = speed_df.dropna(subset=predictors + ['customer_state', 'year_month', 'log_speed']).copy()

for col in predictors:
    speed_std[col + '_std'] = (speed_std[col] - speed_std[col].mean()) / speed_std[col].std()

# Tríade de Especificações: S_L (Linear), S_Q (Quadrático) e S_S (Spline Cúbica)
fml_SL = 'log_speed ~ Z_freight_log_ratio_std + log_distance_std + log_weight_std + log_volume_std + log_freight_std + freight_burden_std + log_pop_std + log_gdp_std + C(year_month)'
fml_SQ = 'log_speed ~ Z_freight_log_ratio_std + I(Z_freight_log_ratio_std**2) + log_distance_std + log_weight_std + log_volume_std + log_freight_std + freight_burden_std + log_pop_std + log_gdp_std + C(year_month)'
fml_SS = 'log_speed ~ cr(Z_freight_log_ratio_std, df=4) + log_distance_std + log_weight_std + log_volume_std + log_freight_std + freight_burden_std + log_pop_std + log_gdp_std + C(year_month)'

specs = {'S_L_Linear': fml_SL, 'S_Q_Quadratic': fml_SQ, 'S_S_Spline': fml_SS}
optimizers = ['cg', 'bfgs', 'lbfgs']

opt_audit = []
fitted_models = {}

for s_name, fml in specs.items():
    model = smf.mixedlm(fml, speed_std, groups=speed_std['customer_state'])
    best_spec_fit = None
    
    for opt in optimizers:
        caught_warns = []
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                fit = model.fit(method=opt, reml=False, maxiter=1000, disp=False)
                
                caught_warns = [str(item.message) for item in w if issubclass(item.category, ConvergenceWarning)]
                
                is_conv = bool(getattr(fit, 'converged', False))
                is_fin_llf = bool(np.isfinite(fit.llf))
                is_fin_scale = bool(np.isfinite(fit.scale) and fit.scale > 0)
                is_fin_params = bool(np.isfinite(fit.params).all())
                is_fin_bse = bool(np.isfinite(fit.bse).all())
                
                valid_fit = is_conv and is_fin_llf and is_fin_scale and is_fin_params and is_fin_bse
                
                opt_audit.append({
                    'spec': s_name, 'optimizer': opt, 'valid_fit': valid_fit,
                    'converged': is_conv, 'isfinite_llf': is_fin_llf,
                    'llf': float(fit.llf) if is_fin_llf else np.nan,
                    'warning_count': len(caught_warns), 'warning_text': " | ".join(caught_warns)
                })
                
                if valid_fit and (best_spec_fit is None or fit.llf > best_spec_fit.llf):
                    best_spec_fit = fit
        except Exception as exc:
            opt_audit.append({
                'spec': s_name, 'optimizer': opt, 'valid_fit': False,
                'converged': False, 'isfinite_llf': False, 'llf': np.nan,
                'warning_count': 1, 'warning_text': f"{type(exc).__name__}: {exc}"
            })
            
    if best_spec_fit is not None:
        fitted_models[s_name] = best_spec_fit

pd.DataFrame(opt_audit).to_csv(REPORTS_DIR / "12a_mixedlm_optimizer_audit.csv", index=False)

if 'S_L_Linear' not in fitted_models:
    raise RuntimeError("MixedLM S_L Linear não convergiu adequadamente em nenhum otimizador!")

best_sl = fitted_models['S_L_Linear']
coef_z = float(best_sl.params['Z_freight_log_ratio_std'])
se_z = float(best_sl.bse['Z_freight_log_ratio_std'])
pct_impact = float(100 * (np.exp(coef_z) - 1))

dec_12 = {
    "status": "PASS",
    "valid_fits_count": len(fitted_models),
    "sl_llf": float(best_sl.llf),
    "z_freight_std_coef": coef_z,
    "z_freight_std_se": se_z,
    "effect_percent_per_SD": pct_impact,
    "practical_magnitude": "NEGLIGIBLE" if abs(pct_impact) < 0.5 else ("SMALL" if abs(pct_impact) < 2.0 else "MODERATE")
}

with open(REPORTS_DIR / "12b_SPEED_CORRECTED_DECISION.json", "w") as f:
    json.dump(dec_12, f, indent=2)

print(f"[PASS 12] VELOCIDADE AUDITADA COM VALIDAÇÃO RIGOROSA. Impacto % por DP: {pct_impact:.4f}% | Magnitude: {dec_12['practical_magnitude']}")
