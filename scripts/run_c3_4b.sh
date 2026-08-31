#!/usr/bin/env bash

PROJECT="$HOME/workspace/Delivery_Risk_Intelligence"
SCRIPT="$PROJECT/src/modeling/model_01_c3_4b_temporal_bcv_rank.py"
DIR="$PROJECT/reports/modeling/model_01_order_logistic"

cd "$PROJECT" || exit 1

echo
echo "================================================================================"
echo "C3.4-B — TEMPORAL BLOCK BI-CROSS-VALIDATION"
echo "================================================================================"

echo
echo "IMPORTANTE:"
echo "- trabalha com RAW-30D já produzido;"
echo "- usa apenas passado para TRAIN;"
echo "- usa mês futuro para TEST;"
echo "- esconde também blocos de lags;"
echo "- NÃO usa target;"
echo "- NÃO escolhe K final;"
echo "- NÃO treina classificador;"
echo "- NÃO altera RAW."

echo
echo "================================================================================"
echo "1. VALIDANDO SINTAXE"
echo "================================================================================"

if python3 -m py_compile "$SCRIPT"; then
    echo "[PASS] Sintaxe válida."
else
    echo "[FAIL] Erro de sintaxe."
    echo "[OK] Shell permanece livre."
    exit 2
fi


echo
echo "================================================================================"
echo "2. EXECUTANDO C3.4-B"
echo "================================================================================"

python3 "$SCRIPT"

STATUS=$?

echo
echo "Python exit code: $STATUS"

if [[ "$STATUS" -ne 0 ]]; then
    echo "[FAIL] C3.4-B precisa de revisão."
    echo "[OK] Shell permanece livre."
    exit "$STATUS"
fi


echo
echo "================================================================================"
echo "3. SUMMARY JSON"
echo "================================================================================"

python3 -m json.tool \
  "$DIR/04r_temporal_bcv_summary.json"


echo
echo "================================================================================"
echo "4. CANDIDATOS BCV"
echo "================================================================================"

python3 - <<'PY'
import pandas as pd

p = (
    "reports/modeling/model_01_order_logistic/"
    "04o_temporal_bcv_scheme_candidates.csv"
)

df = pd.read_csv(p)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)
PY


echo
echo "================================================================================"
echo "5. ARTEFATOS"
echo "================================================================================"

ls -lh \
  "$DIR/04m_temporal_bcv_fold_rank_errors.csv" \
  "$DIR/04n_temporal_bcv_rank_profiles.csv" \
  "$DIR/04o_temporal_bcv_scheme_candidates.csv" \
  "$DIR/04p_temporal_bcv_monthly_best_rank.csv" \
  "$DIR/04q_temporal_bcv_validation.csv" \
  "$DIR/04r_temporal_bcv_summary.json" \
  "$DIR/04s_temporal_bcv_report.txt"


echo
echo "================================================================================"
echo "PARAR AQUI"
echo "================================================================================"

echo "C3.4-A.1            : PASS"
echo "C3.4-B              : EXECUTADO"
echo "BCV temporal        : SIM"
echo "Blocos de lag       : 3 / 6 / 15"
echo "K candidato         : CALCULADO"
echo "K final             : NÃO"
echo "Target              : NÃO"
echo "Folds finais        : NÃO CONGELADOS"
echo "Modelo              : NÃO TREINADO"
echo "Silver              : NÃO CRIADA"
echo "RAW                 : INTACTO"

echo
echo "[OK] Shell continua vivo."
