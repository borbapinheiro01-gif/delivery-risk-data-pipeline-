#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/workspace/Delivery_Risk_Intelligence"
DIR="$PROJECT/reports/data_quality/gate_03b_conditional_task"

cd "$PROJECT" || exit 1

echo
echo "========================================================================================================"
echo "                    DQ GATE 03B — RESULTADOS PARA AUDITORIA"
echo "========================================================================================================"
echo
echo "Diretório:"
echo "$DIR"
echo


# ================================================================================================
# FUNÇÃO DE VERIFICAÇÃO
# ================================================================================================

verificar_arquivo() {

    local arquivo="$1"

    if [[ ! -f "$arquivo" ]]; then

        echo
        echo "[ERRO] Arquivo não encontrado:"
        echo "$arquivo"
        echo

        exit 2
    fi
}


# ================================================================================================
# ARQUIVOS
# ================================================================================================

READINESS="$DIR/05_task_feature_readiness_summary.csv"
REASONS="$DIR/07_missing_reason_registry.csv"
SOURCE_TASK="$DIR/08_source_vs_task_completeness.csv"
SCORECARD="$DIR/dq_gate_03b_scorecard.csv"
SUMMARY="$DIR/dq_gate_03b_summary.json"
REPORT="$DIR/DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt"


for f in \
    "$READINESS" \
    "$REASONS" \
    "$SOURCE_TASK" \
    "$SCORECARD" \
    "$SUMMARY" \
    "$REPORT"
do
    verificar_arquivo "$f"
done


# ================================================================================================
# 1. RESULTADO FORMAL DO GATE
# ================================================================================================

echo
echo "========================================================================================================"
echo "1. RESULTADO FORMAL — DQ GATE 03B"
echo "========================================================================================================"
echo

cat "$REPORT"


# ================================================================================================
# 2. TASK FEATURE READINESS
# ================================================================================================

echo
echo
echo "========================================================================================================"
echo "2. TASK ORDER FEATURE READINESS"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

path = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "05_task_feature_readiness_summary.csv"
)

df = pd.read_csv(path)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


# ================================================================================================
# 3. MISSING REASON TAXONOMY
# ================================================================================================

echo
echo
echo "========================================================================================================"
echo "3. MISSING REASON TAXONOMY"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

path = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "07_missing_reason_registry.csv"
)

df = pd.read_csv(path)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


# ================================================================================================
# 4. SOURCE QUALITY vs TASK QUALITY
# ================================================================================================

echo
echo
echo "========================================================================================================"
echo "4. SOURCE QUALITY vs TASK QUALITY"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

path = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "08_source_vs_task_completeness.csv"
)

df = pd.read_csv(path)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


# ================================================================================================
# 5. SCORECARD — SOMENTE WARN / FAIL
# ================================================================================================

echo
echo
echo "========================================================================================================"
echo "5. SCORECARD — WARN / FAIL"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

path = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "dq_gate_03b_scorecard.csv"
)

df = pd.read_csv(path)

bad = df[
    df["status"].isin(
        ["WARN", "FAIL"]
    )
].copy()

if bad.empty:

    print("[OK] Nenhum WARN ou FAIL.")

else:

    cols = [
        c for c in [
            "status",
            "check_id",
            "dimension",
            "scope",
            "entity",
            "applicability_condition",
            "affected",
            "denominator",
            "affected_pct",
            "severity",
            "details"
        ]
        if c in bad.columns
    ]

    print(
        bad[cols].to_string(
            index=False,
            max_rows=None,
            max_cols=None,
            float_format=lambda x: f"{x:.6f}"
        )
    )
PY


# ================================================================================================
# 6. SUMMARY JSON
# ================================================================================================

echo
echo
echo "========================================================================================================"
echo "6. SUMMARY JSON"
echo "========================================================================================================"
echo

python3 -m json.tool "$SUMMARY"


# ================================================================================================
# FIM
# ================================================================================================

echo
echo "========================================================================================================"
echo "[OK] IMPRESSÃO DOS RESULTADOS DO DQ GATE 03B CONCLUÍDA"
echo "========================================================================================================"

echo
echo "Nenhum CSV foi alterado."
echo "Nenhum dado RAW foi alterado."
echo
