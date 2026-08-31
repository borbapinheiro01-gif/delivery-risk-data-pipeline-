#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.2
SENSITIVITY / SVD / RANK / CONDITIONING AUDIT
===============================================================================

Objetivo:
- auditar a geometria das 13 features do ORDER_CORE_V1;
- medir posto, nulidade e condicionamento;
- confirmar dependências lineares já conhecidas;
- verificar se a estrutura de posto é persistente ao longo do tempo;
- cruzar o condicionamento com o regime temporal do C3.

NÃO:
- remove features;
- cria Silver;
- congela threshold;
- congela boundaries;
- congela folds;
- treina modelo;
- altera RAW.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

MATRIX = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "pretraining"
    / "ORDER_CORE_V1_AUDIT_MATRIX.csv"
)

CONTRACT = (
    PROJECT
    / "configs"
    / "order_core_v1_feature_contract.json"
)

OUT = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

C3 = OUT / "01o_temporal_regime_evidence_matrix.csv"

OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OUTPUTS
# =============================================================================

P_GLOBAL = OUT / "01s_sensitivity_rank_global_summary.csv"
P_SINGULAR = OUT / "01t_singular_values_standardized.csv"
P_NULL = OUT / "01u_nullspace_basis_standardized.csv"
P_IDENTITIES = OUT / "01v_known_linear_identity_validation.csv"
P_TEMPORAL = OUT / "01w_temporal_rank_conditioning.csv"
P_VALIDATION = OUT / "01x_sensitivity_rank_validation.csv"
P_SUMMARY = OUT / "01y_sensitivity_rank_summary.json"
P_REPORT = OUT / "01z_sensitivity_rank_report.txt"


# =============================================================================
# HELPERS
# =============================================================================

def standardize(x):
    x = np.asarray(x, dtype=np.float64)

    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)

    zero_variance = scale == 0

    safe_scale = scale.copy()
    safe_scale[zero_variance] = 1.0

    z = (x - mean) / safe_scale

    return z, mean, scale, zero_variance


def svd_diagnostics(z):
    z = np.asarray(z, dtype=np.float64)

    _, s, vt = np.linalg.svd(
        z,
        full_matrices=False
    )

    eps = np.finfo(s.dtype).eps

    tol = (
        s.max()
        * max(z.shape)
        * eps
    )

    rank = int(
        np.sum(s > tol)
    )

    nullity = int(
        z.shape[1] - rank
    )

    if rank > 0:
        effective_condition = float(
            s[0] / s[rank - 1]
        )
    else:
        effective_condition = np.nan

    if s[-1] > tol:
        full_condition = float(
            s[0] / s[-1]
        )
    else:
        full_condition = np.inf

    return {
        "s": s,
        "vt": vt,
        "tol": float(tol),
        "rank": rank,
        "nullity": nullity,
        "effective_condition": effective_condition,
        "full_condition": full_condition,
    }


def finite_or_none(value):
    value = float(value)

    if np.isfinite(value):
        return value

    return None


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

print()
print("=" * 110)
print("MODEL 01.0-C3.2 — SENSITIVITY / SVD / RANK / CONDITIONING")
print("=" * 110)

if not MATRIX.exists():
    raise SystemExit(f"[FAIL] Matriz ausente: {MATRIX}")

if not CONTRACT.exists():
    raise SystemExit(f"[FAIL] Contrato ausente: {CONTRACT}")

print("[PASS] Matriz encontrada.")
print("[PASS] Contrato encontrado.")


# =============================================================================
# 2. FEATURE CONTRACT
# =============================================================================

with CONTRACT.open(encoding="utf-8") as f:
    contract = json.load(f)

features = [
    row["feature_name"]
    for row in contract["features"]
]

if len(features) != 13:
    raise RuntimeError(
        f"Esperadas 13 features; obtidas {len(features)}."
    )


# =============================================================================
# 3. LOAD MATRIX
# =============================================================================

df = pd.read_csv(
    MATRIX,
    parse_dates=["order_purchase_timestamp"]
)

required = (
    ["order_id", "order_purchase_timestamp", "late_delivery_calendar_day"]
    + features
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Colunas ausentes: {missing}"
    )

Xdf = df[features].apply(
    pd.to_numeric,
    errors="coerce"
)

X = Xdf.to_numpy(dtype=np.float64)

na_count = int(np.isnan(X).sum())
inf_count = int(np.isinf(X).sum())

if na_count:
    raise RuntimeError(f"NA nas features: {na_count}")

if inf_count:
    raise RuntimeError(f"Inf nas features: {inf_count}")


# =============================================================================
# 4. GLOBAL STANDARDIZED MATRIX
# =============================================================================

Z, means, scales, zero_variance = standardize(X)

diag = svd_diagnostics(Z)

s = diag["s"]
vt = diag["vt"]
tol = diag["tol"]
rank = diag["rank"]
nullity = diag["nullity"]

numpy_rank = int(
    np.linalg.matrix_rank(Z)
)

print()
print("GLOBAL MATRIX")
print("-" * 110)
print("Rows               :", len(df))
print("Features           :", len(features))
print("Numerical rank     :", rank)
print("NumPy matrix_rank  :", numpy_rank)
print("Nullity            :", nullity)
print("SVD tolerance      :", f"{tol:.8e}")
print(
    "Effective condition:",
    f"{diag['effective_condition']:.8e}"
)


# =============================================================================
# 5. SINGULAR VALUES
# =============================================================================

energy = s ** 2

energy_ratio = (
    energy / energy.sum()
)

singular = pd.DataFrame({
    "component":
        np.arange(1, len(s) + 1),

    "singular_value":
        s,

    "relative_to_max":
        s / s[0],

    "energy_ratio":
        energy_ratio,

    "cumulative_energy_ratio":
        np.cumsum(energy_ratio),

    "above_rank_tolerance":
        s > tol
})

singular.to_csv(
    P_SINGULAR,
    index=False
)


# =============================================================================
# 6. NUMERICAL NULLSPACE
# =============================================================================

if nullity > 0:
    null_basis = vt[rank:, :].T
else:
    null_basis = np.empty(
        (len(features), 0)
    )

null_rows = []

for component in range(null_basis.shape[1]):

    vector = null_basis[:, component]

    order = np.argsort(
        -np.abs(vector)
    )

    for loading_rank, index in enumerate(
        order,
        start=1
    ):
        null_rows.append({
            "null_component":
                component + 1,

            "loading_rank":
                loading_rank,

            "feature":
                features[index],

            "loading":
                float(vector[index]),

            "abs_loading":
                float(abs(vector[index]))
        })

pd.DataFrame(
    null_rows,
    columns=[
        "null_component",
        "loading_rank",
        "feature",
        "loading",
        "abs_loading"
    ]
).to_csv(
    P_NULL,
    index=False
)


# =============================================================================
# 7. KNOWN EXACT IDENTITIES
# =============================================================================

index = {
    feature: i
    for i, feature in enumerate(features)
}

relations = {
    "PRICE_RANGE": {
        "price_range": 1.0,
        "max_price": -1.0,
        "min_price": 1.0,
    },

    "MERCHANDISE_PLUS_FREIGHT": {
        "merchandise_plus_freight": 1.0,
        "total_price": -1.0,
        "total_freight": -1.0,
    },
}

identity_rows = []

for name, relation in relations.items():

    coeff = np.zeros(
        len(features),
        dtype=np.float64
    )

    for feature, value in relation.items():
        coeff[index[feature]] = value

    residual = X @ coeff

    max_abs = float(
        np.max(np.abs(residual))
    )

    rms = float(
        np.sqrt(
            np.mean(residual ** 2)
        )
    )

    allclose = bool(
        np.allclose(
            residual,
            0.0,
            atol=1e-10,
            rtol=1e-10
        )
    )

    # Relação equivalente após padronização.
    c_std = scales * coeff

    norm = np.linalg.norm(c_std)

    if norm > 0:
        c_std = c_std / norm

    standardized_residual = Z @ c_std

    if null_basis.shape[1] > 0:

        projection = (
            null_basis
            @
            (
                null_basis.T
                @
                c_std
            )
        )

        projection_norm = float(
            np.linalg.norm(projection)
        )

        projection_norm = min(
            1.0,
            max(0.0, projection_norm)
        )

        projection_energy = (
            100.0
            * projection_norm ** 2
        )

        angle = float(
            np.degrees(
                np.arccos(
                    projection_norm
                )
            )
        )

    else:

        projection_energy = 0.0
        angle = np.nan

    identity_rows.append({
        "relation":
            name,

        "formula":
            " + ".join(
                f"{coef:+g}*{feature}"
                for feature, coef
                in relation.items()
            )
            + " = 0",

        "max_abs_residual":
            max_abs,

        "rms_residual":
            rms,

        "allclose_zero":
            allclose,

        "standardized_max_abs_residual":
            float(
                np.max(
                    np.abs(
                        standardized_residual
                    )
                )
            ),

        "projection_energy_in_nullspace_pct":
            projection_energy,

        "angle_to_nullspace_deg":
            angle,
    })

identity_df = pd.DataFrame(
    identity_rows
)

identity_df.to_csv(
    P_IDENTITIES,
    index=False
)


# =============================================================================
# 8. GLOBAL SUMMARY
# =============================================================================

global_summary = pd.DataFrame([{
    "rows":
        len(df),

    "features":
        len(features),

    "numerical_rank":
        rank,

    "numpy_matrix_rank":
        numpy_rank,

    "nullity":
        nullity,

    "known_exact_relations":
        len(relations),

    "known_relation_rank_upper_bound":
        len(features) - len(relations),

    "svd_tolerance":
        tol,

    "largest_singular_value":
        float(s[0]),

    "smallest_singular_value":
        float(s[-1]),

    "smallest_retained_singular_value":
        float(s[rank - 1]),

    "effective_condition_number":
        diag["effective_condition"],

    "full_condition_number":
        diag["full_condition"],

    "zero_variance_features":
        int(zero_variance.sum()),

    "pc1_energy_pct":
        100.0 * float(energy_ratio[0]),

    "pc3_cumulative_energy_pct":
        100.0 * float(
            energy_ratio[:3].sum()
        ),

    "pc5_cumulative_energy_pct":
        100.0 * float(
            energy_ratio[:5].sum()
        ),
}])

global_summary.to_csv(
    P_GLOBAL,
    index=False
)


# =============================================================================
# 9. TEMPORAL RANK / CONDITIONING
# =============================================================================

tmp = df[
    ["order_purchase_timestamp"]
    + features
].copy()

tmp["purchase_month"] = (
    tmp["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

temporal_rows = []

p = len(features)

for month, group in tmp.groupby(
    "purchase_month",
    sort=True
):

    Xm = group[features].to_numpy(
        dtype=np.float64
    )

    n = len(Xm)

    # Muito poucas linhas para diagnóstico geométrico.
    if n < p + 1:

        temporal_rows.append({
            "purchase_month":
                month,

            "orders":
                n,

            "computable":
                False,

            "numerical_rank":
                np.nan,

            "nullity":
                np.nan,

            "zero_variance_features":
                np.nan,

            "effective_condition_number":
                np.nan,

            "log10_effective_condition":
                np.nan,

            "smallest_retained_relative_singular_value":
                np.nan,

            "nullspace_max_angle_deg_vs_global":
                np.nan,

            "nullspace_mean_angle_deg_vs_global":
                np.nan,
        })

        continue

    Zm, _, _, zero_m = standardize(Xm)

    dm = svd_diagnostics(Zm)

    sm = dm["s"]
    rm = dm["rank"]
    nm = dm["nullity"]

    if rm > 0:
        smallest_relative = float(
            sm[rm - 1] / sm[0]
        )
    else:
        smallest_relative = np.nan

    cond = dm["effective_condition"]

    if np.isfinite(cond) and cond > 0:
        log_cond = float(np.log10(cond))
    else:
        log_cond = np.nan

    # -------------------------------------------------------------
    # Espaço nulo mensal vs espaço nulo global
    # -------------------------------------------------------------

    if null_basis.shape[1] > 0 and nm > 0:

        null_month = (
            dm["vt"][rm:, :]
            .T
        )

        angles = np.degrees(
            subspace_angles(
                null_basis,
                null_month
            )
        )

        max_angle = float(
            angles.max()
        )

        mean_angle = float(
            angles.mean()
        )

    else:

        max_angle = np.nan
        mean_angle = np.nan

    temporal_rows.append({
        "purchase_month":
            month,

        "orders":
            n,

        "computable":
            True,

        "numerical_rank":
            rm,

        "nullity":
            nm,

        "zero_variance_features":
            int(zero_m.sum()),

        "effective_condition_number":
            cond,

        "log10_effective_condition":
            log_cond,

        "smallest_retained_relative_singular_value":
            smallest_relative,

        "nullspace_max_angle_deg_vs_global":
            max_angle,

        "nullspace_mean_angle_deg_vs_global":
            mean_angle,
    })

temporal_df = pd.DataFrame(
    temporal_rows
)


# =============================================================================
# 10. JOIN WITH C3
# =============================================================================

c3_joined = False

if C3.exists():

    c3 = pd.read_csv(C3)

    columns = [
        "current_month",
        "local_covariate_score",
        "persistent_covariate_score",
        "target_change_score",
        "consensus_regime_score",
        "consensus_rank",
        "dominant_feature",
    ]

    columns = [
        c for c in columns
        if c in c3.columns
    ]

    c3 = c3[columns].rename(
        columns={
            "current_month":
                "purchase_month"
        }
    )

    temporal_df = temporal_df.merge(
        c3,
        on="purchase_month",
        how="left",
        validate="one_to_one"
    )

    c3_joined = True


# =============================================================================
# 11. CORRELATION WITH TEMPORAL REGIME
# =============================================================================

spearman_consensus = np.nan
spearman_persistent = np.nan

if c3_joined:

    valid = temporal_df.loc[
        temporal_df["computable"].eq(True)
        &
        temporal_df["log10_effective_condition"].notna()
        &
        temporal_df["consensus_regime_score"].notna()
    ].copy()

    if len(valid) >= 3:

        spearman_consensus = float(
            valid[
                "consensus_regime_score"
            ].corr(
                valid[
                    "log10_effective_condition"
                ],
                method="spearman"
            )
        )

    valid2 = temporal_df.loc[
        temporal_df["computable"].eq(True)
        &
        temporal_df["log10_effective_condition"].notna()
        &
        temporal_df[
            "persistent_covariate_score"
        ].notna()
    ].copy()

    if len(valid2) >= 3:

        spearman_persistent = float(
            valid2[
                "persistent_covariate_score"
            ].corr(
                valid2[
                    "log10_effective_condition"
                ],
                method="spearman"
            )
        )


temporal_df.to_csv(
    P_TEMPORAL,
    index=False
)


# =============================================================================
# 12. VALIDATION
# =============================================================================

checks = []


def add(check, condition, observed, expected):

    checks.append({
        "check":
            check,

        "status":
            "PASS"
            if bool(condition)
            else "FAIL",

        "observed":
            observed,

        "expected":
            expected,
    })


add(
    "rows",
    len(df) == 96470,
    len(df),
    96470
)

add(
    "features",
    len(features) == 13,
    len(features),
    13
)

add(
    "feature_na",
    na_count == 0,
    na_count,
    0
)

add(
    "feature_inf",
    inf_count == 0,
    inf_count,
    0
)

add(
    "zero_variance_global",
    int(zero_variance.sum()) == 0,
    int(zero_variance.sum()),
    0
)

add(
    "numpy_rank_equals_explicit_svd_rank",
    numpy_rank == rank,
    numpy_rank,
    rank
)

add(
    "known_linear_identities",
    bool(
        identity_df[
            "allclose_zero"
        ].all()
    ),
    int(
        identity_df[
            "allclose_zero"
        ].sum()
    ),
    len(identity_df)
)

add(
    "nullity_at_least_two",
    nullity >= 2,
    nullity,
    ">=2"
)

add(
    "rank_respects_known_upper_bound",
    rank <= 11,
    rank,
    "<=11"
)


validation = pd.DataFrame(
    checks
)

validation.to_csv(
    P_VALIDATION,
    index=False
)

failures = int(
    validation["status"]
    .eq("FAIL")
    .sum()
)


# =============================================================================
# 13. SUMMARY
# =============================================================================

computable = temporal_df.loc[
    temporal_df["computable"].eq(True)
].copy()


def json_number(x):

    if x is None:
        return None

    x = float(x)

    if np.isfinite(x):
        return x

    return None


summary = {
    "step":
        "MODEL_01_0_C3_2_SENSITIVITY_SVD_RANK",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "path_assessment":
        "VIABLE_DIAGNOSTIC"
        if failures == 0
        else "REVIEW_REQUIRED",

    "rows":
        int(len(df)),

    "features":
        int(len(features)),

    "global_rank":
        int(rank),

    "global_nullity":
        int(nullity),

    "known_exact_relations":
        int(len(relations)),

    "known_rank_upper_bound":
        11,

    "effective_condition_number":
        json_number(
            diag["effective_condition"]
        ),

    "full_condition_number":
        json_number(
            diag["full_condition"]
        ),

    "temporal_months_total":
        int(len(temporal_df)),

    "temporal_months_computable":
        int(len(computable)),

    "temporal_rank_min":
        int(
            computable[
                "numerical_rank"
            ].min()
        )
        if len(computable)
        else None,

    "temporal_rank_max":
        int(
            computable[
                "numerical_rank"
            ].max()
        )
        if len(computable)
        else None,

    "c3_joined":
        bool(c3_joined),

    "spearman_consensus_vs_log_condition":
        json_number(
            spearman_consensus
        ),

    "spearman_persistent_vs_log_condition":
        json_number(
            spearman_persistent
        ),

    "automatic_feature_removal":
        False,

    "threshold_frozen":
        False,

    "boundaries_frozen":
        False,

    "folds_frozen":
        False,

    "model_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures,
}

P_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 14. REPORT TXT
# =============================================================================

lines = [
    "=" * 100,
    "MODEL 01.0-C3.2 — SENSITIVITY / SVD / RANK / CONDITIONING",
    "=" * 100,
    "",
    f"STATUS                       : {summary['status']}",
    f"PATH ASSESSMENT              : {summary['path_assessment']}",
    f"ROWS                         : {len(df):,}",
    f"FEATURES                     : {len(features)}",
    f"GLOBAL RANK                  : {rank}",
    f"GLOBAL NULLITY               : {nullity}",
    f"KNOWN EXACT RELATIONS        : {len(relations)}",
    f"KNOWN RANK UPPER BOUND       : 11",
    f"EFFECTIVE CONDITION NUMBER   : {diag['effective_condition']:.8e}",
    "",
    "IMPORTANT",
    "-" * 100,
    "Diagnostic only.",
    "No feature was removed.",
    "No threshold was frozen.",
    "No temporal boundary was frozen.",
    "No fold was frozen.",
    "No model was trained.",
    "RAW was not modified.",
    "",
    "KNOWN IDENTITIES",
    "-" * 100,
]

for row in identity_rows:

    lines.append(
        f"{row['relation']} | "
        f"max residual={row['max_abs_residual']:.8e} | "
        f"nullspace projection="
        f"{row['projection_energy_in_nullspace_pct']:.8f}%"
    )

lines += [
    "",
    "TEMPORAL",
    "-" * 100,
    f"Months total      : {len(temporal_df)}",
    f"Months computable : {len(computable)}",
]

if len(computable):

    lines.append(
        "Temporal rank     : "
        f"{int(computable['numerical_rank'].min())}"
        " .. "
        f"{int(computable['numerical_rank'].max())}"
    )

lines += [
    "",
    f"C3 joined                    : {c3_joined}",
    f"Spearman consensus/condition : {spearman_consensus}",
    f"Spearman persistent/condition: {spearman_persistent}",
    "",
    "=" * 100,
]

P_REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# =============================================================================
# 15. PRINT RESULTS
# =============================================================================

print()
print("=" * 110)
print("GLOBAL SUMMARY")
print("=" * 110)

print(
    global_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.8e}"
    )
)

print()
print("=" * 110)
print("SINGULAR VALUES")
print("=" * 110)

print(
    singular.to_string(
        index=False,
        float_format=lambda x: f"{x:.8e}"
    )
)

print()
print("=" * 110)
print("KNOWN LINEAR IDENTITIES")
print("=" * 110)

print(
    identity_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.10e}"
    )
)

print()
print("=" * 110)
print("NULLSPACE — TOP LOADINGS")
print("=" * 110)

if null_rows:

    null_df = pd.DataFrame(
        null_rows
    )

    top_null = (
        null_df
        .groupby(
            "null_component",
            group_keys=False
        )
        .head(6)
    )

    print(
        top_null.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}"
        )
    )

else:

    print("No numerical nullspace.")


print()
print("=" * 110)
print("TEMPORAL RANK / CONDITIONING")
print("=" * 110)

columns = [
    "purchase_month",
    "orders",
    "computable",
    "numerical_rank",
    "nullity",
    "zero_variance_features",
    "effective_condition_number",
    "smallest_retained_relative_singular_value",
    "nullspace_max_angle_deg_vs_global",
]

if c3_joined:

    for c in [
        "consensus_regime_score",
        "consensus_rank",
        "dominant_feature",
    ]:
        if c in temporal_df.columns:
            columns.append(c)

print(
    temporal_df[
        columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


print()
print("=" * 110)
print("VALIDATION")
print("=" * 110)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 110)
print("RESULTADO C3.2")
print("=" * 110)

print("FAILURES                 =", failures)
print("GLOBAL RANK              =", rank)
print("GLOBAL NULLITY           =", nullity)
print("KNOWN RELATIONS          =", len(relations))
print(
    "EFFECTIVE CONDITION      =",
    diag["effective_condition"]
)
print("C3 JOINED                =", c3_joined)
print(
    "SPEARMAN CONSENSUS/COND  =",
    spearman_consensus
)
print(
    "SPEARMAN PERSISTENT/COND =",
    spearman_persistent
)

print()
print(
    "PATH ASSESSMENT          =",
    summary["path_assessment"]
)

print()
print("FEATURE REMOVAL          = NÃO")
print("THRESHOLD FROZEN         = NÃO")
print("BOUNDARIES FROZEN        = NÃO")
print("FOLDS FROZEN             = NÃO")
print("MODEL TRAINED            = NÃO")
print("SILVER CREATED           = NÃO")
print("RAW MODIFIED             = NÃO")

if failures:
    raise SystemExit(2)

print()
print("[PASS] MODEL 01.0-C3.2 concluído.")
print("[PASS] Nenhum tratamento automático aplicado.")
