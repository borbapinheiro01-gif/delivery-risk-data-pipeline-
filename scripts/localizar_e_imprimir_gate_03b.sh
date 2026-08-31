#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/workspace/Delivery_Risk_Intelligence"
cd "$PROJECT" || exit 1

echo
echo "========================================================================================================"
echo "DQ GATE 03B — LOCALIZAÇÃO E DIAGNÓSTICO"
echo "========================================================================================================"
echo

echo "[1] Procurando diretórios relacionados ao Gate 03B..."
echo "--------------------------------------------------------------------------------------------------------"

find "$PROJECT" \
  -type d \
  \( \
      -iname '*03b*' \
      -o -iname '*conditional*task*' \
      -o -iname '*conditional*completeness*' \
  \) \
  -print \
  2>/dev/null \
  | sort

echo
echo "[2] Procurando TODOS os artefatos esperados do Gate 03B..."
echo "--------------------------------------------------------------------------------------------------------"

TARGETS=(
    "05_task_feature_readiness_summary.csv"
    "07_missing_reason_registry.csv"
    "08_source_vs_task_completeness.csv"
    "dq_gate_03b_scorecard.csv"
    "dq_gate_03b_summary.json"
    "DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt"
)

FOUND=0

for nome in "${TARGETS[@]}"; do

    echo
    echo "### $nome"

    RESULTADO="$(
        find "$PROJECT" \
          -type f \
          -name "$nome" \
          -print \
          2>/dev/null \
          | sort
    )"

    if [[ -n "$RESULTADO" ]]; then
        echo "$RESULTADO"
        FOUND=$((FOUND + 1))
    else
        echo "[NÃO ENCONTRADO]"
    fi
done

echo
echo "[3] Procurando o EXECUTOR do Gate 03B..."
echo "--------------------------------------------------------------------------------------------------------"

EXECUTOR="$PROJECT/src/data/dq_gate_03b_conditional_task.py"

if [[ -f "$EXECUTOR" ]]; then
    echo "[OK] Executor encontrado:"
    echo "$EXECUTOR"

    echo
    echo "Sintaxe Python:"
    if python3 -m py_compile "$EXECUTOR"; then
        echo "[PASS] Sintaxe válida."
    else
        echo "[FAIL] O executor existe, mas tem erro de sintaxe."
        exit 2
    fi
else
    echo "[NÃO ENCONTRADO]"
    echo "$EXECUTOR"
fi


echo
echo "[4] INVENTÁRIO DE reports/data_quality"
echo "--------------------------------------------------------------------------------------------------------"

find "$PROJECT/reports/data_quality" \
  -maxdepth 2 \
  -type f \
  -printf '%p\n' \
  2>/dev/null \
  | sort


echo
echo "========================================================================================================"
echo "DIAGNÓSTICO"
echo "========================================================================================================"

if [[ "$FOUND" -eq "${#TARGETS[@]}" ]]; then

    echo
    echo "[OK] Os 6 artefatos existem."
    echo "Eles podem estar em outro diretório."
    echo
    echo "Não execute o Gate novamente ainda."

elif [[ "$FOUND" -gt 0 ]]; then

    echo
    echo "[ATENÇÃO] Apenas $FOUND de ${#TARGETS[@]} artefatos foram encontrados."
    echo "A execução pode ter sido parcial/interrompida."
    echo
    echo "Não apague nada."

else

    echo
    echo "[CONFIRMADO] Nenhum dos artefatos finais do Gate 03B existe."

    if [[ -f "$EXECUTOR" ]]; then

        echo
        echo "O executor existe e pode ser executado."
        echo
        echo "Executando agora o DQ Gate 03B..."
        echo "========================================================================================================"
        echo

        set +e
        python3 "$EXECUTOR"
        GATE_STATUS=$?
        set -e

        echo
        echo "========================================================================================================"
        echo "EXECUÇÃO DO GATE 03B TERMINOU"
        echo "========================================================================================================"
        echo "Exit code: $GATE_STATUS"

        if [[ "$GATE_STATUS" -ne 0 ]]; then
            echo
            echo "[FAIL] O Gate 03B não concluiu normalmente."
            echo "Cole a saída acima para auditoria."
            exit "$GATE_STATUS"
        fi

    else

        echo
        echo "[BLOQUEADO] O executor também não existe."
        echo "Precisaremos recriar somente o script do Gate 03B."
        exit 2
    fi
fi


echo
echo "[5] VERIFICAÇÃO PÓS-DIAGNÓSTICO"
echo "--------------------------------------------------------------------------------------------------------"

for nome in "${TARGETS[@]}"; do

    RESULTADO="$(
        find "$PROJECT" \
          -type f \
          -name "$nome" \
          -print \
          2>/dev/null \
          | head -n 1
    )"

    if [[ -n "$RESULTADO" ]]; then
        echo "[OK] $nome"
        echo "     $RESULTADO"
    else
        echo "[MISS] $nome"
    fi
done


echo
echo "========================================================================================================"
echo "6. IMPRIMIR RESULTADOS, SE AGORA EXISTIREM"
echo "========================================================================================================"
echo

DIR="$PROJECT/reports/data_quality/gate_03b_conditional_task"

if [[ ! -d "$DIR" ]]; then
    echo "[ERRO] Diretório final ainda não existe:"
    echo "$DIR"
    exit 2
fi


# --------------------------------------------------------------------------------
# REPORT
# --------------------------------------------------------------------------------

if [[ -f "$DIR/DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt" ]]; then

    echo
    echo "========================================================================================================"
    echo "A. RESULTADO FORMAL"
    echo "========================================================================================================"
    cat "$DIR/DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt"
fi


# --------------------------------------------------------------------------------
# TABELAS
# --------------------------------------------------------------------------------

python3 - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
)

tables = [
    (
        "B. TASK ORDER FEATURE READINESS",
        "05_task_feature_readiness_summary.csv",
    ),
    (
        "C. MISSING REASON TAXONOMY",
        "07_missing_reason_registry.csv",
    ),
    (
        "D. SOURCE QUALITY vs TASK QUALITY",
        "08_source_vs_task_completeness.csv",
    ),
]

for title, filename in tables:

    print()
    print("=" * 104)
    print(title)
    print("=" * 104)

    path = root / filename

    if not path.exists():
        print(f"[NÃO ENCONTRADO] {path}")
        continue

    df = pd.read_csv(path)

    print(
        df.to_string(
            index=False,
            max_rows=None,
            max_cols=None,
            float_format=lambda x: f"{x:.6f}",
        )
    )


print()
print("=" * 104)
print("E. SCORECARD — WARN / FAIL")
print("=" * 104)

score_path = root / "dq_gate_03b_scorecard.csv"

if score_path.exists():

    df = pd.read_csv(score_path)

    bad = df[
        df["status"].isin(["WARN", "FAIL"])
    ].copy()

    if bad.empty:
        print("[OK] Nenhum WARN ou FAIL.")
    else:
        print(
            bad.to_string(
                index=False,
                max_rows=None,
                max_cols=None,
                float_format=lambda x: f"{x:.6f}",
            )
        )
else:
    print("[NÃO ENCONTRADO]", score_path)


print()
print("=" * 104)
print("F. SUMMARY JSON")
print("=" * 104)

summary_path = root / "dq_gate_03b_summary.json"

if summary_path.exists():

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    print(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        )
    )
else:
    print("[NÃO ENCONTRADO]", summary_path)

PY


echo
echo "========================================================================================================"
echo "[OK] DIAGNÓSTICO/IMPRESSÃO CONCLUÍDOS"
echo "========================================================================================================"
echo
echo "Nenhum RAW foi alterado."
echo "Nenhum CSV existente foi apagado."
echo
