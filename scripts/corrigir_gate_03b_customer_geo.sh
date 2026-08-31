#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/workspace/Delivery_Risk_Intelligence"
ARQUIVO="$PROJECT/src/data/dq_gate_03b_conditional_task.py"
BACKUP="$PROJECT/src/data/dq_gate_03b_conditional_task_BEFORE_FIX_CUSTOMER_GEO.py"

cd "$PROJECT" || exit 1

echo
echo "========================================================================================================"
echo "DQ GATE 03B — CORREÇÃO DO BUG customer_geo_missing"
echo "========================================================================================================"
echo

if [[ ! -f "$ARQUIVO" ]]; then
    echo "[ERRO] Executor não encontrado:"
    echo "$ARQUIVO"
    exit 2
fi

# ================================================================================================
# 1. BACKUP
# ================================================================================================

cp -f "$ARQUIVO" "$BACKUP"

echo "[OK] Backup criado:"
echo "$BACKUP"
echo


# ================================================================================================
# 2. PATCH CONTROLADO
# ================================================================================================

python3 - <<'PY'
from pathlib import Path

project = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
path = project / "src" / "data" / "dq_gate_03b_conditional_task.py"

text = path.read_text(encoding="utf-8")

old = '''task_customer[
    "customer_resolved"
] = (
    task_customer[
        "__customer_merge"
    ]
    ==
    "both"
)


customer_resolved_rows = task_customer[
    task_customer[
        "customer_resolved"
    ]
].copy()


customer_resolved_rows[
    "customer_geo_missing"
] = (
    ~customer_resolved_rows[
        "customer_zip_code_prefix"
    ]
    .isin(
        geo_zip_set
    )
)
'''

new = '''task_customer[
    "customer_resolved"
] = (
    task_customer[
        "__customer_merge"
    ]
    ==
    "both"
)


# -------------------------------------------------------------------------
# IMPORTANTE:
#
# customer_geo_missing pertence ao DataFrame task_customer porque esse
# mesmo DataFrame será usado posteriormente para construir customer_flags.
#
# Para clientes não resolvidos, geolocation NÃO é simplesmente "missing":
# a relação upstream customer já está ausente. Por isso usamos pd.NA.
#
# Somente quando customer_id resolve para customers é que avaliamos se o
# ZIP possui referência na tabela geolocation.
# -------------------------------------------------------------------------

task_customer[
    "customer_geo_missing"
] = pd.Series(
    pd.NA,
    index=task_customer.index,
    dtype="boolean"
)


_customer_resolved_mask = (
    task_customer[
        "customer_resolved"
    ]
    .fillna(False)
)


task_customer.loc[
    _customer_resolved_mask,
    "customer_geo_missing"
] = (
    ~task_customer.loc[
        _customer_resolved_mask,
        "customer_zip_code_prefix"
    ]
    .isin(
        geo_zip_set
    )
)


customer_resolved_rows = task_customer.loc[
    _customer_resolved_mask
].copy()
'''

if old not in text:
    print("[ERRO] Bloco original não foi localizado exatamente.")
    print()
    print("Nenhuma alteração foi feita.")
    raise SystemExit(2)

if text.count(old) != 1:
    print(
        "[ERRO] O bloco apareceu uma quantidade inesperada de vezes:",
        text.count(old)
    )
    print("Nenhuma alteração foi feita.")
    raise SystemExit(2)

patched = text.replace(old, new, 1)

path.write_text(
    patched,
    encoding="utf-8"
)

print("[PASS] Patch aplicado exatamente uma vez.")
PY


# ================================================================================================
# 3. AUDITAR O TRECHO CORRIGIDO
# ================================================================================================

echo
echo "========================================================================================================"
echo "TRECHO CORRIGIDO"
echo "========================================================================================================"
echo

python3 - <<'PY'
from pathlib import Path

path = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
    / "src"
    / "data"
    / "dq_gate_03b_conditional_task.py"
)

lines = path.read_text(
    encoding="utf-8"
).splitlines()

targets = [
    'task_customer[',
    '"customer_geo_missing"',
    'customer_resolved_rows = task_customer.loc['
]

for i, line in enumerate(lines):
    if (
        '"customer_geo_missing"' in line
        and i > 0
    ):
        start = max(0, i - 35)
        end = min(len(lines), i + 40)

        for n in range(start, end):
            print(
                f"{n+1:5d} | {lines[n]}"
            )

        break
PY


# ================================================================================================
# 4. SINTAXE
# ================================================================================================

echo
echo "========================================================================================================"
echo "VALIDAÇÃO DE SINTAXE"
echo "========================================================================================================"
echo

python3 -m py_compile "$ARQUIVO"

echo "[PASS] Sintaxe Python válida."


# ================================================================================================
# 5. TESTE SEMÂNTICO LOCAL DO BLOCO
#
# Confirma que:
# - customer_geo_missing passa a existir em task_customer;
# - clientes não resolvidos não são confundidos com geo missing;
# - clientes resolvidos recebem True/False.
# ================================================================================================

echo
echo "========================================================================================================"
echo "TESTE SEMÂNTICO DO PATCH"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

task_customer = pd.DataFrame(
    {
        "order_id": ["A", "B", "C"],
        "customer_id": ["X", "Y", "Z"],
        "customer_zip_code_prefix": [1000, 2000, pd.NA],
        "__customer_merge": ["both", "both", "left_only"],
    }
)

geo_zip_set = {1000}

task_customer["customer_resolved"] = (
    task_customer["__customer_merge"] == "both"
)

task_customer["customer_geo_missing"] = pd.Series(
    pd.NA,
    index=task_customer.index,
    dtype="boolean",
)

mask = task_customer["customer_resolved"].fillna(False)

task_customer.loc[
    mask,
    "customer_geo_missing"
] = (
    ~task_customer.loc[
        mask,
        "customer_zip_code_prefix"
    ].isin(geo_zip_set)
)

print(
    task_customer[
        [
            "order_id",
            "customer_resolved",
            "customer_geo_missing",
        ]
    ].to_string(index=False)
)

assert task_customer.loc[0, "customer_geo_missing"] == False
assert task_customer.loc[1, "customer_geo_missing"] == True
assert pd.isna(
    task_customer.loc[2, "customer_geo_missing"]
)

print()
print("[PASS] RELATION_MISSING não foi confundido com GEO_MISSING.")
PY


# ================================================================================================
# 6. LIMPAR SOMENTE ARTEFATOS PARCIAIS DO GATE 03B
#
# Não toca em RAW.
# Não toca nos Gates 01, 02 ou 03.
# Não toca no Registry.
# ================================================================================================

OUT="$PROJECT/reports/data_quality/gate_03b_conditional_task"

echo
echo "========================================================================================================"
echo "REMOVENDO SOMENTE A EXECUÇÃO PARCIAL DO GATE 03B"
echo "========================================================================================================"
echo

if [[ -d "$OUT" ]]; then

    find "$OUT" \
      -maxdepth 1 \
      -type f \
      -print

    find "$OUT" \
      -maxdepth 1 \
      -type f \
      -delete

fi

echo
echo "[OK] Diretório do Gate 03B preparado para rerun limpo."


# ================================================================================================
# 7. EXECUTAR NOVAMENTE
# ================================================================================================

echo
echo "========================================================================================================"
echo "RERUN COMPLETO — DQ GATE 03B"
echo "========================================================================================================"
echo

set +e

python3 "$ARQUIVO"

STATUS=$?

set -e

echo
echo "========================================================================================================"
echo "DQ GATE 03B — EXIT CODE"
echo "========================================================================================================"
echo

echo "STATUS=$STATUS"

if [[ "$STATUS" -ne 0 ]]; then

    echo
    echo "[FAIL] O Gate ainda encontrou um erro de execução."
    echo "O backup do executor foi preservado."
    exit "$STATUS"

fi


# ================================================================================================
# 8. VERIFICAR TODOS OS ARTEFATOS
# ================================================================================================

echo
echo "========================================================================================================"
echo "VERIFICAÇÃO DOS ARTEFATOS FINAIS"
echo "========================================================================================================"
echo

EXPECTED=(
    "01_source_relationship_coverage.csv"
    "02_conditional_attribute_completeness.csv"
    "03_post_purchase_event_completeness.csv"
    "04_task_order_feature_readiness.csv"
    "05_task_feature_readiness_summary.csv"
    "06_task_completeness_by_month.csv"
    "07_missing_reason_registry.csv"
    "08_source_vs_task_completeness.csv"
    "09_multi_entity_structure.csv"
    "10_task_cohort_funnel.csv"
    "11_issue_evidence_gate_03b.csv"
    "dq_gate_03b_scorecard.csv"
    "dq_gate_03b_exceptions.csv"
    "dq_gate_03b_summary.json"
    "DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt"
)

MISSING=0

for nome in "${EXPECTED[@]}"; do

    if [[ -f "$OUT/$nome" ]]; then
        echo "[OK]   $nome"
    else
        echo "[MISS] $nome"
        MISSING=$((MISSING + 1))
    fi

done

echo

if [[ "$MISSING" -ne 0 ]]; then

    echo "[FAIL] Artefatos ausentes: $MISSING"
    exit 2

fi

echo "[PASS] Todos os ${#EXPECTED[@]} artefatos foram produzidos."


# ================================================================================================
# 9. IMPRIMIR EXATAMENTE O QUE PRECISAMOS AUDITAR
# ================================================================================================

echo
echo "========================================================================================================"
echo "A. RESULTADO FORMAL"
echo "========================================================================================================"
echo

cat "$OUT/DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt"


echo
echo "========================================================================================================"
echo "B. TASK ORDER FEATURE READINESS"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

p = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "05_task_feature_readiness_summary.csv"
)

df = pd.read_csv(p)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


echo
echo "========================================================================================================"
echo "C. MISSING REASON TAXONOMY"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

p = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "07_missing_reason_registry.csv"
)

df = pd.read_csv(p)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


echo
echo "========================================================================================================"
echo "D. SOURCE QUALITY vs TASK QUALITY"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

p = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "08_source_vs_task_completeness.csv"
)

df = pd.read_csv(p)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


echo
echo "========================================================================================================"
echo "E. MULTI-ENTITY STRUCTURE"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

p = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "09_multi_entity_structure.csv"
)

df = pd.read_csv(p)

print(
    df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda x: f"{x:.6f}"
    )
)
PY


echo
echo "========================================================================================================"
echo "F. SCORECARD — WARN / FAIL"
echo "========================================================================================================"
echo

python3 - <<'PY'
import pandas as pd

p = (
    "reports/data_quality/"
    "gate_03b_conditional_task/"
    "dq_gate_03b_scorecard.csv"
)

df = pd.read_csv(p)

bad = df[
    df["status"].isin(
        ["WARN", "FAIL"]
    )
]

if bad.empty:

    print("[OK] Nenhum WARN/FAIL.")

else:

    print(
        bad.to_string(
            index=False,
            max_rows=None,
            max_cols=None,
            float_format=lambda x: f"{x:.6f}"
        )
    )
PY


echo
echo "========================================================================================================"
echo "G. SUMMARY JSON"
echo "========================================================================================================"
echo

python3 -m json.tool \
  "$OUT/dq_gate_03b_summary.json"


echo
echo "========================================================================================================"
echo "[OK] CORREÇÃO + RERUN + AUDITORIA CONCLUÍDOS"
echo "========================================================================================================"
echo

echo "RAW alterado: NÃO"
echo "Gate 01 alterado: NÃO"
echo "Gate 02 alterado: NÃO"
echo "Gate 03 alterado: NÃO"
echo "Registry alterado: NÃO"
echo
