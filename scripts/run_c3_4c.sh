#!/usr/bin/env bash

cd "$HOME/workspace/Delivery_Risk_Intelligence" || exit 1

SCRIPT="src/modeling/model_01_c3_4c_bcv_block_bootstrap_stability.py"

echo
echo "================================================================================"
echo "C3.4-C — TEMPORAL BLOCK-BOOTSTRAP BCV STABILITY"
echo "================================================================================"

echo
echo "1. VALIDANDO SINTAXE"
echo "--------------------------------------------------------------------------------"

python3 -m py_compile "$SCRIPT"

STATUS=$?

if [[ "$STATUS" -ne 0 ]]; then
    echo "[FAIL] Sintaxe inválida."
    exit "$STATUS"
fi

echo "[PASS] Sintaxe válida."

echo
echo "2. EXECUTANDO"
echo "--------------------------------------------------------------------------------"

python3 "$SCRIPT"

STATUS=$?

echo
echo "Python exit code: $STATUS"

if [[ "$STATUS" -ne 0 ]]; then
    echo "[FAIL] C3.4-C precisa de revisão."
    exit "$STATUS"
fi

echo
echo "================================================================================"
echo "ARTEFATOS"
echo "================================================================================"

ls -lh \
  reports/modeling/model_01_order_logistic/04t_bcv_monthly_rank_profile.csv \
  reports/modeling/model_01_order_logistic/04u_bcv_bootstrap_rank_frequency.csv \
  reports/modeling/model_01_order_logistic/04v_bcv_bootstrap_stability_summary.csv \
  reports/modeling/model_01_order_logistic/04w_bcv_bootstrap_validation.csv \
  reports/modeling/model_01_order_logistic/04x_bcv_bootstrap_summary.json \
  reports/modeling/model_01_order_logistic/04y_bcv_bootstrap_report.txt

echo
echo "================================================================================"
echo "PARAR AQUI"
echo "================================================================================"

echo "C3.4-B          : PASS"
echo "Monte Carlo     : EXECUTADO"
echo "Block bootstrap : 2 / 3 / 4 meses"
echo "K final         : NÃO"
echo "Adaptive        : AINDA NÃO"
echo "Target          : NÃO"
echo "Modelo          : NÃO TREINADO"
echo "Silver          : NÃO CRIADA"
echo "RAW             : INTACTO"

echo
echo "[OK] Shell continua vivo."
