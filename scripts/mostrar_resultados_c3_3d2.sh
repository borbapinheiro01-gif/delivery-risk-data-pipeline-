#!/usr/bin/env bash

PROJECT="$HOME/workspace/Delivery_Risk_Intelligence"
DIR="$PROJECT/reports/modeling/model_01_order_logistic"

SUMMARY_TABLE="$DIR/03l_raw_vs_smooth_temporal_summary.csv"
PAIRED="$DIR/03m_raw_vs_smooth_paired_comparison.csv"
VALIDATION="$DIR/03n_raw_vs_smooth_validation.csv"
SUMMARY_JSON="$DIR/03o_raw_vs_smooth_summary.json"
REPORT="$DIR/03p_raw_vs_smooth_report.txt"

OUT="$DIR/03q_D2_RESULTADOS_PARA_AUDITORIA.txt"

cd "$PROJECT" || exit 1

echo
echo "================================================================================"
echo "C3.3-D2 — RECUPERAÇÃO DOS RESULTADOS"
echo "================================================================================"
echo
echo "IMPORTANTE:"
echo "- NÃO roda PCA novamente."
echo "- NÃO roda smoothing novamente."
echo "- NÃO altera RAW."
echo "- NÃO treina modelo."
echo "- Apenas lê os resultados já produzidos."
echo


# =============================================================================
# 1. VERIFICAR SE O D2 SOBREVIVEU AO RESTART
# =============================================================================

echo "================================================================================"
echo "1. VERIFICANDO ARTEFATOS"
echo "================================================================================"

MISSING=0

for arquivo in \
    "$SUMMARY_TABLE" \
    "$PAIRED" \
    "$VALIDATION" \
    "$SUMMARY_JSON" \
    "$REPORT"
do
    if [[ -s "$arquivo" ]]; then
        echo "[OK] $(basename "$arquivo")"
    else
        echo "[AUSENTE] $(basename "$arquivo")"
        MISSING=$((MISSING + 1))
    fi
done

echo
echo "Arquivos ausentes: $MISSING"
echo


if [[ "$MISSING" -gt 0 ]]; then

    echo "================================================================================"
    echo "[STOP] O D2 NÃO ESTÁ COMPLETO NO DISCO"
    echo "================================================================================"
    echo
    echo "Não vou recalcular nada automaticamente."
    echo
    echo "Arquivos disponíveis atualmente:"
    echo

    find "$DIR" \
        -maxdepth 1 \
        -type f \
        \( -name '03*.csv' -o -name '03*.json' -o -name '03*.txt' \) \
        -printf '%TY-%Tm-%Td %TH:%TM:%TS  %f\n' \
        2>/dev/null \
        | sort

    echo
    echo "[OK] Shell continua vivo."
    exit 0
fi


# =============================================================================
# 2. GERAR UM ARQUIVO ÚNICO PARA COLAR NO CHAT
# =============================================================================

rm -f "$OUT"

python3 - <<'PY' | tee "$OUT"
from pathlib import Path
import json
import pandas as pd
import numpy as np

DIR = Path(
    "reports/modeling/model_01_order_logistic"
)

SUMMARY_TABLE = (
    DIR / "03l_raw_vs_smooth_temporal_summary.csv"
)

PAIRED = (
    DIR / "03m_raw_vs_smooth_paired_comparison.csv"
)

VALIDATION = (
    DIR / "03n_raw_vs_smooth_validation.csv"
)

SUMMARY_JSON = (
    DIR / "03o_raw_vs_smooth_summary.json"
)

REPORT = (
    DIR / "03p_raw_vs_smooth_report.txt"
)


def sep(title):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


# =============================================================================
# A. SUMMARY JSON
# =============================================================================

sep("A. SUMMARY JSON — ESTADO FORMAL DO D2")

with SUMMARY_JSON.open(
    encoding="utf-8"
) as f:
    summary_json = json.load(f)

print(
    json.dumps(
        summary_json,
        indent=4,
        ensure_ascii=False
    )
)


# =============================================================================
# B. VALIDATION
# =============================================================================

sep("B. VALIDAÇÃO DO D2")

validation = pd.read_csv(
    VALIDATION
)

print(
    validation.to_string(
        index=False,
        max_rows=None,
        max_cols=None
    )
)

failures = int(
    validation["status"]
    .astype(str)
    .str.upper()
    .eq("FAIL")
    .sum()
)

print()
print(
    "VALIDATION FAILURES =",
    failures
)


# =============================================================================
# C. SUMMARY PRINCIPAL
# =============================================================================

sep("C. RAW 30D vs SMOOTH 30D EDF=0.75 — RESUMO PRINCIPAL")

df = pd.read_csv(
    SUMMARY_TABLE
)

preferred_cols = [
    "channel",
    "temporal_tests",

    "raw_k90_median",
    "raw_k90_min",
    "raw_k90_max",

    "smooth_k90_median",
    "smooth_k90_min",
    "smooth_k90_max",

    "k90_reduction_mean",

    "raw_future_error_mean",
    "raw_future_error_median",
    "raw_future_error_p95",

    "smooth_internal_future_error_mean",
    "smooth_internal_future_error_median",

    "smooth_end_to_end_raw_error_mean",
    "smooth_end_to_end_raw_error_median",
    "smooth_end_to_end_raw_error_p95",

    "paired_end_to_end_delta_mean",
    "paired_end_to_end_delta_median",
    "paired_end_to_end_ratio_mean",

    "smooth_better_end_to_end_folds",
    "smooth_better_end_to_end_fold_pct",

    "smooth_test_distortion_mean",
    "smooth_test_distortion_p95",

    "raw_basis_mean_angle_deg_mean",
    "smooth_basis_mean_angle_deg_mean",

    "raw_basis_max_angle_deg_max",
    "smooth_basis_max_angle_deg_max",

    "smooth_negative_cell_pct_mean",
    "smooth_negative_cell_pct_max",
]

cols = [
    c
    for c in preferred_cols
    if c in df.columns
]

print(
    df[cols].to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.10f}"
    )
)


# =============================================================================
# D. DECISÃO NUMÉRICA BÁSICA
# =============================================================================

sep("D. LEITURA NUMÉRICA AUTOMÁTICA — SEM PROMOVER NENHUM MÉTODO")

for _, row in df.iterrows():

    channel = row["channel"]

    raw_error = float(
        row["raw_future_error_mean"]
    )

    smooth_error = float(
        row["smooth_end_to_end_raw_error_mean"]
    )

    delta = float(
        row["paired_end_to_end_delta_mean"]
    )

    ratio = float(
        row["paired_end_to_end_ratio_mean"]
    )

    raw_k = float(
        row["raw_k90_median"]
    )

    smooth_k = float(
        row["smooth_k90_median"]
    )

    wins = int(
        row["smooth_better_end_to_end_folds"]
    )

    tests = int(
        row["temporal_tests"]
    )

    print()
    print(channel)
    print("-" * 80)

    print(
        f"RAW future error mean             : {raw_error:.10f}"
    )

    print(
        f"SMOOTH -> RAW future error mean   : {smooth_error:.10f}"
    )

    print(
        f"Delta (smooth - raw)              : {delta:.10f}"
    )

    print(
        f"Ratio (smooth / raw)              : {ratio:.10f}"
    )

    print(
        f"RAW K90 median                    : {raw_k:.2f}"
    )

    print(
        f"SMOOTH K90 median                 : {smooth_k:.2f}"
    )

    print(
        f"SMOOTH better folds               : {wins}/{tests}"
    )

    if delta < 0:
        print(
            "SINAL END-TO-END                  : "
            "SMOOTH teve erro médio menor."
        )

    elif delta > 0:
        print(
            "SINAL END-TO-END                  : "
            "RAW teve erro médio menor."
        )

    else:
        print(
            "SINAL END-TO-END                  : "
            "empate numérico na média."
        )


# =============================================================================
# E. COMPARAÇÃO MÊS A MÊS
# =============================================================================

sep("E. COMPARAÇÃO PAREADA MÊS A MÊS")

paired = pd.read_csv(
    PAIRED
)

paired_cols = [
    "channel",
    "current_month",
    "train_orders",
    "test_orders",

    "raw_k90_train",
    "smooth_k90_train",
    "k90_delta_smooth_minus_raw",

    "raw_future_relative_error_k90",
    "smooth_internal_future_error_k90",
    "smooth_end_to_end_raw_error_k90",

    "smooth_test_distortion_from_raw",

    "end_to_end_error_delta_smooth_minus_raw",
    "end_to_end_error_ratio_smooth_over_raw",

    "smooth_lower_end_to_end_error",

    "raw_basis_mean_angle_deg",
    "smooth_basis_mean_angle_deg",

    "smooth_negative_cell_pct_test",
]

paired_cols = [
    c
    for c in paired_cols
    if c in paired.columns
]

print(
    paired[paired_cols].to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.10f}"
    )
)


# =============================================================================
# F. MESES EM QUE SMOOTH MAIS AJUDOU
# =============================================================================

sep("F. TOP MESES — SMOOTH MAIS AJUDOU END-TO-END")

if (
    "end_to_end_error_delta_smooth_minus_raw"
    in paired.columns
):

    best = (
        paired.sort_values(
            "end_to_end_error_delta_smooth_minus_raw",
            ascending=True
        )
        .head(10)
    )

    show = [
        "channel",
        "current_month",
        "raw_k90_train",
        "smooth_k90_train",
        "raw_future_relative_error_k90",
        "smooth_end_to_end_raw_error_k90",
        "end_to_end_error_delta_smooth_minus_raw",
        "end_to_end_error_ratio_smooth_over_raw",
    ]

    show = [
        c
        for c in show
        if c in best.columns
    ]

    print(
        best[show].to_string(
            index=False,
            float_format=lambda x: f"{x:.10f}"
        )
    )


# =============================================================================
# G. MESES EM QUE SMOOTH MAIS PIOROU
# =============================================================================

sep("G. TOP MESES — SMOOTH MAIS PIOROU END-TO-END")

if (
    "end_to_end_error_delta_smooth_minus_raw"
    in paired.columns
):

    worst = (
        paired.sort_values(
            "end_to_end_error_delta_smooth_minus_raw",
            ascending=False
        )
        .head(10)
    )

    show = [
        "channel",
        "current_month",
        "raw_k90_train",
        "smooth_k90_train",
        "raw_future_relative_error_k90",
        "smooth_end_to_end_raw_error_k90",
        "end_to_end_error_delta_smooth_minus_raw",
        "end_to_end_error_ratio_smooth_over_raw",
    ]

    show = [
        c
        for c in show
        if c in worst.columns
    ]

    print(
        worst[show].to_string(
            index=False,
            float_format=lambda x: f"{x:.10f}"
        )
    )


# =============================================================================
# H. CONSISTÊNCIA PAREADA
# =============================================================================

sep("H. CONSISTÊNCIA DA COMPARAÇÃO")

for channel, g in paired.groupby(
    "channel",
    sort=True
):

    delta = pd.to_numeric(
        g[
            "end_to_end_error_delta_smooth_minus_raw"
        ],
        errors="coerce"
    )

    print()
    print(channel)
    print("-" * 80)

    print(
        "Temporal tests          :",
        len(g)
    )

    print(
        "SMOOTH wins             :",
        int(
            (
                delta < 0
            ).sum()
        )
    )

    print(
        "RAW wins                :",
        int(
            (
                delta > 0
            ).sum()
        )
    )

    print(
        "Ties                    :",
        int(
            (
                delta == 0
            ).sum()
        )
    )

    print(
        "Mean delta              :",
        f"{delta.mean():.10f}"
    )

    print(
        "Median delta            :",
        f"{delta.median():.10f}"
    )

    print(
        "Minimum delta           :",
        f"{delta.min():.10f}"
    )

    print(
        "Maximum delta           :",
        f"{delta.max():.10f}"
    )


# =============================================================================
# I. RELATÓRIO ORIGINAL
# =============================================================================

sep("I. RELATÓRIO FORMAL GERADO PELO D2")

print(
    REPORT.read_text(
        encoding="utf-8"
    )
)


# =============================================================================
# J. ESTADO DA PESQUISA
# =============================================================================

sep("J. ESTADO APÓS O D2")

print(
    "D2 executado                     : SIM"
)

print(
    "Comparação RAW vs SMOOTH          : SIM"
)

print(
    "Comparação temporal pareada       : SIM"
)

print(
    "Erro end-to-end contra RAW futuro : SIM"
)

print()

print(
    "Smoothing promovido               : NÃO"
)

print(
    "Janela final congelada            : NÃO"
)

print(
    "K final congelado                 : NÃO"
)

print(
    "Target usado nesta análise        : NÃO"
)

print(
    "Clipping aplicado                 : NÃO"
)

print(
    "Feature funcional criada          : NÃO"
)

print(
    "Modelo treinado                   : NÃO"
)

print(
    "Silver criada                     : NÃO"
)

print(
    "RAW alterado                      : NÃO"
)

print()
print(
    "VALIDATION FAILURES               :",
    failures
)

print()
print("=" * 118)
print("FIM DOS RESULTADOS C3.3-D2")
print("=" * 118)
PY


STATUS=${PIPESTATUS[0]}

echo
echo "================================================================================"
echo "RESULTADO DO LEITOR"
echo "================================================================================"

echo "Python exit code: $STATUS"

if [[ "$STATUS" -eq 0 ]]; then
    echo "[PASS] Resultados do D2 recuperados."
else
    echo "[FAIL] Houve erro ao ler os resultados."
fi

echo
echo "Arquivo salvo para não depender da tela do Shell:"
ls -lh "$OUT" 2>/dev/null || true

echo
echo "Caminho:"
echo "$OUT"

echo
echo "================================================================================"
echo "[OK] Shell continua vivo."
echo "================================================================================"
