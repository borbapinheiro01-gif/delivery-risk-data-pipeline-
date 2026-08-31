#!/usr/bin/env bash

PROJECT="$HOME/workspace/Delivery_Risk_Intelligence"
DIR="$PROJECT/reports/modeling/model_01_order_logistic"

FOLDS="$DIR/04a_raw30_k_policy_folds.csv"
SUMMARY="$DIR/04b_raw30_k_policy_summary.csv"
ONESE="$DIR/04c_raw30_k_policy_one_se.csv"
VALIDATION="$DIR/04d_raw30_k_policy_validation.csv"
SUMMARY_JSON="$DIR/04e_raw30_k_policy_summary.json"
REPORT="$DIR/04f_raw30_k_policy_report.txt"

OUT="$DIR/04l_C34A_RESULTADOS_PARA_AUDITORIA.txt"

cd "$PROJECT" || exit 1

{
echo
echo "================================================================================================"
echo "C3.4-A — RECUPERAÇÃO DOS RESULTADOS JÁ CALCULADOS"
echo "================================================================================================"
echo
echo "IMPORTANTE:"
echo "- NÃO executa PCA."
echo "- NÃO executa SVD."
echo "- NÃO abre as matrizes .npy."
echo "- NÃO recalcula curvas."
echo "- NÃO usa target."
echo "- NÃO escolhe K."
echo "- NÃO treina modelo."
echo "- NÃO altera RAW."
echo "- Apenas lê os artefatos existentes do C3.4-A."
echo


# ================================================================================================
# 1. ARTEFATOS
# ================================================================================================

echo "================================================================================================"
echo "1. VERIFICANDO ARTEFATOS"
echo "================================================================================================"

MISSING=0

for f in \
    "$FOLDS" \
    "$SUMMARY" \
    "$ONESE" \
    "$VALIDATION" \
    "$SUMMARY_JSON" \
    "$REPORT"
do
    if [[ -s "$f" ]]; then
        echo "[OK] $(basename "$f")"
    else
        echo "[AUSENTE] $f"
        MISSING=$((MISSING + 1))
    fi
done

echo
echo "Arquivos ausentes: $MISSING"

if [[ "$MISSING" -ne 0 ]]; then
    echo
    echo "[STOP] Existem artefatos ausentes."
    exit 2
fi


# ================================================================================================
# 2. SUMMARY JSON
# ================================================================================================

echo
echo "================================================================================================"
echo "2. ESTADO FORMAL DO C3.4-A"
echo "================================================================================================"

python3 -m json.tool "$SUMMARY_JSON"


# ================================================================================================
# 3. POLICY SUMMARY
# ================================================================================================

echo
echo "================================================================================================"
echo "3. POLÍTICAS PVE — K versus ERRO FUTURO"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04b_raw30_k_policy_summary.csv"
)

df = pd.read_csv(p)

cols = [
    "channel",
    "pve_policy",
    "temporal_tests",
    "mean_future_error",
    "median_future_error",
    "p95_future_error",
    "se_future_error",
    "mean_k",
    "median_k",
    "min_k",
    "max_k",
    "sd_k",
    "mean_future_variance_captured",
]

print(
    df[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)
PY


# ================================================================================================
# 4. ONE-SE
# ================================================================================================

echo
echo "================================================================================================"
echo "4. REGRA 1-SE — APENAS CANDIDATA"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04c_raw30_k_policy_one_se.csv"
)

df = pd.read_csv(p)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)
PY


# ================================================================================================
# 5. VALIDATION
# ================================================================================================

echo
echo "================================================================================================"
echo "5. VALIDAÇÃO DO C3.4-A"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04d_raw30_k_policy_validation.csv"
)

df = pd.read_csv(p)

print(df.to_string(index=False))

print()
print(
    "FAILURES =",
    int(df["status"].eq("FAIL").sum())
)
PY


# ================================================================================================
# 6. VISÃO COMPACTA DO q=.90/.95/.99
# ================================================================================================

echo
echo "================================================================================================"
echo "6. COMPARAÇÃO DIRETA — q=.90 vs .95 vs .99"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04b_raw30_k_policy_summary.csv"
)

df = pd.read_csv(p)

df = df[
    df["pve_policy"].round(2).isin(
        [0.90, 0.95, 0.99]
    )
].copy()

cols = [
    "channel",
    "pve_policy",
    "mean_k",
    "median_k",
    "min_k",
    "max_k",
    "mean_future_error",
    "p95_future_error",
    "mean_future_variance_captured",
]

print(
    df[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)
PY


# ================================================================================================
# 7. CUSTO DIMENSIONAL
# ================================================================================================

echo
echo "================================================================================================"
echo "7. QUANTO DA DIMENSÃO ORIGINAL ESTÁ SENDO MANTIDA"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04b_raw30_k_policy_summary.csv"
)

df = pd.read_csv(p)

df["mean_dimension_retained_pct"] = (
    100.0 * df["mean_k"] / 30.0
)

df["mean_dimension_removed_pct"] = (
    100.0 - df["mean_dimension_retained_pct"]
)

cols = [
    "channel",
    "pve_policy",
    "mean_k",
    "mean_dimension_retained_pct",
    "mean_dimension_removed_pct",
    "mean_future_error",
]

print(
    df[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


# ================================================================================================
# 8. q=.99 EM DESTAQUE
# ================================================================================================

echo
echo "================================================================================================"
echo "8. O QUE q=.99 SIGNIFICA NA PRÁTICA"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04b_raw30_k_policy_summary.csv"
)

df = pd.read_csv(p)

x = df[
    df["pve_policy"].round(2).eq(0.99)
].copy()

for _, r in x.iterrows():

    retained = (
        100.0
        * float(r["mean_k"])
        / 30.0
    )

    removed = (
        100.0
        -
        retained
    )

    print()
    print(r["channel"])
    print("-" * 80)

    print(
        "K médio                  :",
        f'{r["mean_k"]:.4f} / 30'
    )

    print(
        "K mediano                :",
        f'{r["median_k"]:.0f}'
    )

    print(
        "Intervalo K              :",
        f'{int(r["min_k"])} .. {int(r["max_k"])}'
    )

    print(
        "Dimensão média mantida   :",
        f"{retained:.2f}%"
    )

    print(
        "Dimensão média removida   :",
        f"{removed:.2f}%"
    )

    print(
        "Erro futuro médio        :",
        f'{r["mean_future_error"]:.10f}'
    )

    print(
        "Variância futura capturada:",
        f'{100*r["mean_future_variance_captured"]:.4f}%'
    )
PY


# ================================================================================================
# 9. FOLDS q=.99 MÊS A MÊS
# ================================================================================================

echo
echo "================================================================================================"
echo "9. q=.99 — COMPORTAMENTO MÊS A MÊS"
echo "================================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04a_raw30_k_policy_folds.csv"
)

df = pd.read_csv(p)

x = df[
    df["pve_policy"].round(2).eq(0.99)
].copy()

cols = [
    "channel",
    "current_month",
    "train_orders",
    "test_orders",
    "k_train",
    "train_cumulative_pve",
    "future_relative_error",
    "future_variance_captured",
]

print(
    x[cols].to_string(
        index=False,
        float_format=lambda z: f"{z:.8f}"
    )
)
PY


# ================================================================================================
# 10. RELATÓRIO FORMAL ORIGINAL
# ================================================================================================

echo
echo "================================================================================================"
echo "10. RELATÓRIO FORMAL DO C3.4-A"
echo "================================================================================================"

cat "$REPORT"


# ================================================================================================
# 11. LEITURA FINAL — SEM TOMAR NOVA DECISÃO
# ================================================================================================

echo
echo "================================================================================================"
echo "11. ESTADO APÓS C3.4-A"
echo "================================================================================================"

python3 - <<'PY'
import json
from pathlib import Path

p = Path(
    "reports/modeling/model_01_order_logistic/"
    "04e_raw30_k_policy_summary.json"
)

d = json.loads(
    p.read_text(encoding="utf-8")
)

print(
    "C3.4-A status              :",
    d.get("status")
)

print(
    "Representação              :",
    d.get("representation")
)

print(
    "Meses temporais            :",
    d.get("temporal_test_months")
)

print(
    "Primeiro mês               :",
    d.get("first_test_month")
)

print(
    "Último mês                 :",
    d.get("last_test_month")
)

print(
    "PVE candidate              :",
    d.get("joint_candidate_policy")
)

print(
    "Status candidate           :",
    d.get("joint_candidate_status")
)

print(
    "Target usado               :",
    d.get("target_used")
)

print(
    "PVE final escolhida        :",
    d.get("final_pve_policy_selected")
)

print(
    "K final escolhido          :",
    d.get("final_k_selected")
)

print(
    "Folds congelados           :",
    d.get("folds_frozen")
)

print(
    "Modelo treinado            :",
    d.get("classifier_trained")
)

print(
    "Silver criada              :",
    d.get("silver_created")
)

print(
    "RAW alterado               :",
    d.get("raw_modified")
)

print(
    "Validation failures        :",
    d.get("validation_failures")
)
PY


echo
echo "================================================================================================"
echo "FIM DOS RESULTADOS C3.4-A"
echo "================================================================================================"

echo
echo "Nenhum experimento foi reexecutado."
echo "PCA executada      : NÃO"
echo "SVD executada      : NÃO"
echo "Target usado       : NÃO"
echo "K escolhido        : NÃO"
echo "Modelo treinado    : NÃO"
echo "Silver criada      : NÃO"
echo "RAW alterado       : NÃO"

} 2>&1 | tee "$OUT"

echo
echo "================================================================================"
echo "RESULTADO DO LEITOR"
echo "================================================================================"

echo "[PASS] Apenas os resultados existentes do C3.4-A foram lidos."

echo
echo "Saída salva em:"
ls -lh "$OUT"

echo
echo "$OUT"

echo
echo "[OK] Shell continua vivo."
