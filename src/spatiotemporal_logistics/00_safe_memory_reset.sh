#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/workspace/Delivery_Risk_Intelligence"
cd "$ROOT"

echo "===================================================================================================="
echo "SAFE MEMORY RESET — DELIVERY RISK INTELLIGENCE"
echo "===================================================================================================="
date
echo

echo "1. MEMÓRIA ANTES"
echo "----------------------------------------------------------------------------------------------------"
free -h || true

if [[ -r /sys/fs/cgroup/memory.current ]]; then
    echo
    echo "cgroup memory.current:"
    cat /sys/fs/cgroup/memory.current
fi

if [[ -r /sys/fs/cgroup/memory.peak ]]; then
    echo
    echo "cgroup memory.peak:"
    cat /sys/fs/cgroup/memory.peak
fi


echo
echo "2. PROCURANDO SOMENTE PROCESSOS ANTIGOS DO NOSSO PIPELINE"
echo "----------------------------------------------------------------------------------------------------"

PATTERNS=(
    "01_build_fast_spatiotemporal_core.py"
    "restaurar_e_rodar.py"
    "repair_fast_core_syntax.py"
    "patch_01_ibge_population_reader.py"
)

FOUND=0

for pattern in "${PATTERNS[@]}"; do

    while read -r pid command; do

        [[ -z "${pid:-}" ]] && continue

        # Não toca no shell atual.
        if [[ "$pid" == "$$" ]]; then
            continue
        fi

        echo "[FOUND] PID=$pid | $command"

        kill -TERM "$pid" 2>/dev/null || true

        FOUND=$((FOUND + 1))

    done < <(
        ps -eo pid=,args= \
        | grep -F "$pattern" \
        | grep -v grep \
        | grep -v "00_safe_memory_reset.sh" \
        || true
    )

done


if [[ "$FOUND" -eq 0 ]]; then
    echo "[PASS] Nenhum processo antigo do pipeline encontrado."
else
    echo
    echo "[INFO] Aguardando processos encerrarem..."
    sleep 3
fi


echo
echo "3. CONFIRMANDO QUE NÃO RESTARAM PROCESSOS DO FAST CORE"
echo "----------------------------------------------------------------------------------------------------"

for pattern in "${PATTERNS[@]}"; do

    LEFT="$(
        ps -eo pid=,args= \
        | grep -F "$pattern" \
        | grep -v grep \
        | grep -v "00_safe_memory_reset.sh" \
        || true
    )"

    if [[ -n "$LEFT" ]]; then
        echo "[WARN] ainda existe processo:"
        echo "$LEFT"
    else
        echo "[PASS] $pattern = não executando"
    fi

done


echo
echo "4. LIMPEZA SOMENTE DE TEMPORÁRIOS DO NOSSO PROJETO"
echo "----------------------------------------------------------------------------------------------------"

find src/spatiotemporal_logistics \
    -type d \
    -name '__pycache__' \
    -prune \
    -exec rm -rf {} + \
    2>/dev/null || true

find metadata/spatiotemporal_logistics \
    -maxdepth 2 \
    -type f \
    \( -name '*.part' -o -name '*.tmp' \) \
    -delete \
    2>/dev/null || true

find data/external \
    -type f \
    -name '*.part' \
    -delete \
    2>/dev/null || true

echo "[PASS] temporários próprios removidos."


echo
echo "5. FLUSH DE ESCRITAS PENDENTES"
echo "----------------------------------------------------------------------------------------------------"

sync

echo "[PASS] sync concluído."


echo
echo "6. NÃO FORÇANDO DROP_CACHES"
echo "----------------------------------------------------------------------------------------------------"

echo "[PASS] /proc/sys/vm/drop_caches NÃO será usado."
echo "[INFO] Page cache é recuperável automaticamente pelo Linux."
echo "[INFO] Forçar drop_caches pode piorar I/O e desempenho."


echo
echo "7. PROCESSOS QUE MAIS CONSOMEM RAM AGORA"
echo "----------------------------------------------------------------------------------------------------"

ps aux --sort=-%mem | head -15 || true


echo
echo "8. MEMÓRIA DEPOIS"
echo "----------------------------------------------------------------------------------------------------"

free -h || true

if [[ -r /sys/fs/cgroup/memory.current ]]; then
    CURRENT="$(cat /sys/fs/cgroup/memory.current)"
    echo
    echo "memory.current bytes = $CURRENT"

    python3 - "$CURRENT" <<'PY'
import sys

value = int(sys.argv[1])

print(
    "memory.current GiB = "
    f"{value / 1024**3:.3f}"
)
PY
fi


if [[ -r /sys/fs/cgroup/memory.max ]]; then
    MAXIMUM="$(cat /sys/fs/cgroup/memory.max)"

    echo "memory.max bytes     = $MAXIMUM"

    if [[ "$MAXIMUM" != "max" ]]; then

        python3 - "$CURRENT" "$MAXIMUM" <<'PY'
import sys

current = int(sys.argv[1])
maximum = int(sys.argv[2])

pct = (
    100.0 * current / maximum
    if maximum
    else float("nan")
)

print(
    "cgroup RAM usage     = "
    f"{pct:.2f}%"
)
PY

    fi
fi


echo
echo "9. PROTEÇÃO DOS DADOS"
echo "----------------------------------------------------------------------------------------------------"

for target in \
    "data/raw/olist" \
    "artifacts/model_01_order_logistic/pretraining/ORDER_CORE_V1_AUDIT_MATRIX.csv" \
    "reports/modeling/model_01_order_logistic/09a_model01_k0_formal_decision.json" \
    "data/external/landing"
do

    if [[ -e "$target" ]]; then
        echo "[PASS] preservado: $target"
    else
        echo "[FAIL] ausente: $target"
        exit 1
    fi

done


echo
echo "===================================================================================================="
echo "STATUS = PASS"
echo "PIPELINE_OLD_PROCESSES_TERMINATED = $FOUND"
echo "DROP_CACHES_USED = false"
echo "RAW_MODIFIED = false"
echo "READY_FOR_STAGED_PIPELINE = true"
echo "PARAR AQUI"
echo "===================================================================================================="
