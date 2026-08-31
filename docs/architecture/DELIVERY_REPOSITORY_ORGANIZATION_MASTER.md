# Delivery Risk & Shipping Friction Intelligence Platform
## Master Repository Organization Runner

Este arquivo é o **runner mestre** para organizar a arquitetura final do repositório sem mover nem apagar o Scientific Core existente.

### Estratégia

O arquivo foi desenhado para a “jogada” de carregar um `.md` na infraestrutura, extrair o único bloco `bash`, validar e executar.

Ele é idempotente e protege o estado anterior do Git. Se ainda existirem arquivos LFS staged, a arquitetura é criada no working tree, mas não é adicionada ao índice; depois do commit LFS, basta executar o mesmo arquivo novamente.

### Arquitetura congelada

```text
DATA FOUNDATION
      ↓
LOGISTICS INTELLIGENCE
      ↓
PREDICTIVE AI
      ↓
DECISION ENGINE
      ↓
MLOps / CLOUD
```

### O que preserva

- 120 arquivos Python científicos;
- 5 fontes legacy;
- RAW e external fora do Git;
- paths científicos existentes;
- nenhuma execução de modelo;
- nenhuma recomputação de feature;
- nenhum commit;
- nenhum push.

### Script executável

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/workspace/Delivery_Risk_GitHub_Staging"
EXPECTED_REMOTE="https://github.com/borbapinheiro01-gif/delivery-risk-data-pipeline-.git"

cd "$ROOT" || {
    echo "[FAIL] Staging não encontrado: $ROOT"
    exit 1
}

echo
echo "===================================================================================================="
echo "DELIVERY RISK & SHIPPING FRICTION INTELLIGENCE PLATFORM"
echo "MASTER REPOSITORY ORGANIZATION"
echo "===================================================================================================="

# 1. GIT PRECONDITIONS
[[ -d .git ]] || {
    echo "[FAIL] .git não encontrado."
    exit 1
}

BRANCH="$(git branch --show-current)"
REMOTE="$(git remote get-url origin)"

echo "BRANCH = $BRANCH"
echo "ORIGIN = $REMOTE"
echo "HEAD   = $(git rev-parse --short HEAD)"

[[ "$BRANCH" == "main" ]] || {
    echo "[FAIL] Branch esperada: main"
    exit 1
}

[[ "$REMOTE" == "$EXPECTED_REMOTE" ]] || {
    echo "[FAIL] Remote diferente do esperado."
    exit 1
}

echo "[PASS] Repositório correto."

# 2. CURRENT GIT STATE
STAGED_BEFORE="$(git diff --cached --name-only | wc -l)"
UNSTAGED_BEFORE="$(git diff --name-only | wc -l)"
UNTRACKED_BEFORE="$(git ls-files --others --exclude-standard | wc -l)"

echo
echo "STAGED_BEFORE    = $STAGED_BEFORE"
echo "UNSTAGED_BEFORE  = $UNSTAGED_BEFORE"
echo "UNTRACKED_BEFORE = $UNTRACKED_BEFORE"

if [[ "$STAGED_BEFORE" -gt 0 ]]; then
    echo "[INFO] Há alterações staged de uma etapa anterior."
    echo "[INFO] A arquitetura será criada no working tree, mas NÃO será adicionada ao index nesta execução."
fi

# 3. SCIENTIFIC CORE INVENTORY
PY_BEFORE="$(find src -type f -name '*.py' | wc -l)"
LEGACY_BEFORE="$(find legacy/source_history -type f -name '*.py.txt' 2>/dev/null | wc -l || true)"

echo
echo "PYTHON_SOURCE_BEFORE = $PY_BEFORE"
echo "LEGACY_BEFORE        = $LEGACY_BEFORE"

if [[ "$PY_BEFORE" -ne 120 ]]; then
    echo "[FAIL] Esperados 120 arquivos Python científicos existentes."
    exit 1
fi

if [[ "$LEGACY_BEFORE" -ne 5 ]]; then
    echo "[FAIL] Esperados 5 arquivos legacy."
    exit 1
fi

echo "[PASS] Scientific Core reconhecido."

# 4. CREATE TARGET DIRECTORY STRUCTURE
mkdir -p \
    src/production/features \
    src/production/late_risk \
    src/production/eta \
    src/production/explainability \
    src/production/decision_engine \
    src/production/monitoring \
    src/production/nlp \
    sql/bronze \
    sql/silver \
    sql/gold \
    sql/quality_checks \
    api \
    mlops/mlflow \
    mlops/registry \
    mlops/monitoring \
    infra/docker \
    infra/aws \
    tests/data \
    tests/features \
    tests/models \
    tests/api \
    docs/architecture \
    docs/methodology \
    docs/results \
    docs/data_governance \
    docs/model_cards \
    docs/portfolio \
    data/bronze \
    data/silver \
    data/gold

echo "[PASS] Estrutura física criada."

# 5. ARCHITECTURE CONTRACT
cat > configs/product_architecture_v1.json <<'JSON'
{
  "project": {
    "name": "Delivery Risk & Shipping Friction Intelligence Platform",
    "subtitle": "End-to-End AI for Predictive Logistics, Cost Intelligence and MLOps",
    "architecture_version": "1.0",
    "status": "FROZEN_TARGET_ARCHITECTURE"
  },
  "architecture": [
    "DATA_FOUNDATION",
    "LOGISTICS_INTELLIGENCE",
    "PREDICTIVE_AI",
    "DECISION_ENGINE",
    "MLOPS_CLOUD"
  ],
  "scientific_core": {
    "preserve_existing_paths": true,
    "physical_reorganization_of_existing_sources": false
  },
  "prediction_contract": {
    "prediction_time": "order_purchase_timestamp",
    "official_target": "late_delivery_calendar_day",
    "point_in_time_required": true,
    "future_information_forbidden": true
  },
  "shipping_friction": {
    "role": "DIAGNOSTIC_AND_FEATURE_INTELLIGENCE",
    "critical_shipping_friction_is_ex_post": true,
    "critical_shipping_friction_allowed_as_same_order_t0_predictor": false,
    "expected_freight_may_be_predictive_if_point_in_time_safe": true
  },
  "late_risk": {
    "scientific_experimental_core": "EXISTS",
    "production_benchmark": "PLANNED",
    "primary_metric": "PR_AUC",
    "operational_metrics": [
      "Recall@TopK",
      "Precision@TopK",
      "Calibration",
      "Brier"
    ]
  },
  "eta": {
    "status": "PLANNED",
    "probabilistic_outputs": [
      "P50",
      "P80",
      "P95"
    ]
  },
  "production": {
    "sql": "PLANNED",
    "mlflow": "PLANNED",
    "fastapi": "PLANNED",
    "docker": "PLANNED",
    "aws": "PLANNED",
    "github_actions": "PLANNED",
    "monitoring": "PLANNED"
  }
}
JSON

python3 -m json.tool configs/product_architecture_v1.json >/dev/null
echo "[PASS] Architecture contract válido."

# 6. MODULE README FILES
python3 - <<'PY'
from pathlib import Path

root = Path.home() / "workspace" / "Delivery_Risk_GitHub_Staging"

docs = {
    "src/production/README.md": "# Production Layer\n\nData Foundation -> Logistics Intelligence -> Predictive AI -> Decision Engine -> MLOps / Cloud.\n",
    "src/production/features/README.md": "# Production Features\n\nPoint-in-Time-safe production feature generation.\n",
    "src/production/late_risk/README.md": "# Late Risk\n\nM0: Core PIT features.\n\nM1: Core + PIT-safe Shipping Intelligence.\n\nPrimary metric: PR-AUC.\nOperational metric: Recall@Top-K.\n",
    "src/production/eta/README.md": "# ETA\n\nDeterministic ETA plus P50 / P80 / P95.\n",
    "src/production/explainability/README.md": "# Explainability\n\nGlobal and local predictive explanations. Predictive explanation is not causality.\n",
    "src/production/decision_engine/README.md": "# Decision Engine\n\nPrediction -> Ranking -> Capacity -> Expected Value -> Action.\n",
    "src/production/monitoring/README.md": "# Monitoring\n\nData, model, calibration, ranking and system monitoring.\n",
    "src/production/nlp/README.md": "# NLP\n\nPost-delivery review intelligence. Reviews are not same-order purchase-time features.\n",
    "sql/README.md": "# SQL Layer\n\nBronze -> Silver -> Gold -> Quality Checks.\n",
    "sql/bronze/README.md": "# Bronze SQL\n\nSource-aligned ingestion views.\n",
    "sql/silver/README.md": "# Silver SQL\n\nTyped and validated transformations.\n",
    "sql/gold/README.md": "# Gold SQL\n\nModel-facing entities. Planned grain: one row per order.\n",
    "sql/quality_checks/README.md": "# SQL Quality Checks\n\nSchema, uniqueness, nullability and referential integrity.\n",
    "api/README.md": "# FastAPI Serving Layer\n\nServing must reproduce training-time PIT definitions.\n",
    "mlops/README.md": "# MLOps\n\nMLflow -> Registry -> Serving -> Monitoring -> Challenger/Champion.\n",
    "mlops/mlflow/README.md": "# MLflow\n\nExperiment tracking and reproducible metadata.\n",
    "mlops/registry/README.md": "# Model Registry\n\nCandidate, challenger and champion governance.\n",
    "mlops/monitoring/README.md": "# Monitoring\n\nData drift, performance, calibration and operational metrics.\n",
    "infra/README.md": "# Infrastructure\n\nDeployment and cloud infrastructure.\n",
    "infra/docker/README.md": "# Docker\n\nReproducible training and serving containers.\n",
    "infra/aws/README.md": "# AWS\n\nTarget cloud architecture.\n",
    "tests/README.md": "# Tests\n\nData, feature, model and API tests.\n",
    "tests/data/README.md": "# Data Tests\n\nData contract and quality tests.\n",
    "tests/features/README.md": "# Feature Tests\n\nPIT and feature identity tests.\n",
    "tests/models/README.md": "# Model Tests\n\nPredictive and serialization tests.\n",
    "tests/api/README.md": "# API Tests\n\nServing contract and response tests.\n",
    "data/bronze/README.md": "# Bronze Data\n\nLocal/runtime Bronze materializations.\n",
    "data/silver/README.md": "# Silver Data\n\nLocal/runtime validated materializations.\n",
    "data/gold/README.md": "# Gold Data\n\nLocal/runtime model-ready datasets.\n"
}

for rel, text in docs.items():
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

print("MODULE_READMES =", len(docs))
print("[PASS] Module README files created.")
PY

# 7. CODE CATALOG
python3 - <<'PY'
from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path.home() / "workspace" / "Delivery_Risk_GitHub_Staging"
SRC = ROOT / "src"

CSV_OUT = ROOT / "metadata" / "code_catalog.csv"
MD_OUT = ROOT / "docs" / "architecture" / "CODE_CATALOG.md"

CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)

def domain(rel):
    p = rel.as_posix().lower()
    if p.startswith("src/modeling/"):
        return "MODEL_01_SCIENTIFIC_CORE"
    if p.startswith("src/spatiotemporal_logistics/"):
        return "SPATIOTEMPORAL_LOGISTICS"
    if p.startswith("src/shipping_friction/"):
        return "SHIPPING_FRICTION"
    if p.startswith("src/production/"):
        return "PRODUCT_LAYER"
    if "quality" in p or "audit" in p or "gate" in p:
        return "DATA_QUALITY_GOVERNANCE"
    return "OTHER_SCIENTIFIC_SUPPORT"

def lifecycle(rel):
    p = rel.as_posix().lower()
    historical = (
        "backup",
        "before_",
        "_before",
        "broken",
        "repair",
        "restore",
        "restaur",
        "legacy",
        "snapshot"
    )
    if any(t in p for t in historical):
        return "SUPPORT_OR_HISTORICAL"
    if p.startswith("src/production/"):
        return "PRODUCT_LAYER"
    return "SCIENTIFIC_CORE"

files = sorted(SRC.rglob("*.py"))

rows = []
for path in files:
    rel = path.relative_to(ROOT)
    rows.append({
        "path": rel.as_posix(),
        "domain": domain(rel),
        "lifecycle": lifecycle(rel),
        "exists": "true"
    })

with CSV_OUT.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["path", "domain", "lifecycle", "exists"]
    )
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[row["domain"]].append(row)

lines = [
    "# Code Catalog",
    "",
    "Machine-generated navigation catalog.",
    "",
    "Existing scientific source paths are intentionally preserved.",
    "",
    f"Total Python files catalogued: **{len(rows)}**",
    ""
]

for key in sorted(groups):
    lines.append(f"## {key}")
    lines.append("")
    for row in groups[key]:
        lines.append(f"- `{row['path']}` — `{row['lifecycle']}`")
    lines.append("")

MD_OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

print("PYTHON_CATALOG_ROWS =", len(rows))

if len(rows) != 120:
    raise SystemExit(f"Expected 120 Python files, observed {len(rows)}")

print("[PASS] 120/120 source files catalogued.")
PY

# 8. ARCHITECTURE DOCUMENTATION
cat > docs/architecture/FINAL_PRODUCT_ARCHITECTURE.md <<'MD'
# Delivery Risk & Shipping Friction Intelligence Platform

## End-to-End Architecture

DATA FOUNDATION
→ LOGISTICS INTELLIGENCE
→ PREDICTIVE AI
→ DECISION ENGINE
→ MLOps / CLOUD

## Scientific Core

Olist
→ Order x Seller
→ Order Level
→ Municipal / External Context
→ Spatiotemporal Core
→ Expected Freight OOT
→ Shipping Friction

## Predictive AI

Late Risk production benchmark:

M0 = Core Point-in-Time features

M1 = Core Point-in-Time features + Point-in-Time-safe Shipping Intelligence

Primary metric: PR-AUC.

Operational metric: Recall@Top-K.

## ETA

Deterministic ETA followed by probabilistic P50 / P80 / P95.

## Decision Engine

Prediction
→ Risk Ranking
→ Operational Capacity
→ Expected Intervention Value
→ Recommended Action

## Serving

MLflow
→ Model Registry
→ FastAPI
→ Docker
→ AWS

## Scientific Guardrails

prediction != causality

anomaly != proven inefficiency

high freight != logistics failure

post-outcome data != purchase-time predictor

diagnostic intelligence != predictive feature
MD

cat > docs/architecture/REPOSITORY_STRUCTURE.md <<'MD'
# Repository Structure

Scientific Core remains in the current source paths.

New Product Layer:

- src/production/features
- src/production/late_risk
- src/production/eta
- src/production/explainability
- src/production/decision_engine
- src/production/monitoring
- src/production/nlp
- sql/bronze
- sql/silver
- sql/gold
- sql/quality_checks
- api
- mlops
- infra
- tests

Existing scientific sources are intentionally not moved during this organization pass.
MD

cat > docs/architecture/SCIENTIFIC_TO_PRODUCT_MAP.md <<'MD'
# Scientific Core → Product Layer

Data Quality
→ Point-in-Time Governance
→ Order Core
→ Temporal Diagnostics
→ Spatiotemporal Logistics
→ Expected Freight
→ Shipping Friction
→ Production Feature Contract
→ Late Risk M0 vs M1
→ Algorithm Benchmark
→ Recall@Top-K
→ Explainability
→ ETA
→ Probabilistic ETA
→ Decision Engine
→ API
→ MLOps
→ Cloud
→ Monitoring
MD

cat > docs/architecture/NEXT_IMPLEMENTATION_ORDER.md <<'MD'
# Next Implementation Order

00. Repository / GitHub governance
01. SQL / Medallion
02. Production feature contract
03. Late Risk M0 vs M1
04. Algorithm benchmark
05. Recall@Top-K
06. Explainability
07. ETA
08. Probabilistic ETA
09. Decision Engine
10. MLflow
11. FastAPI
12. Docker
13. AWS
14. GitHub Actions / CI-CD
15. Monitoring / drift
16. Final portfolio documentation
MD

# 9. MODULE STATUS MATRIX
cat > metadata/product_module_status.csv <<'CSV'
layer,module,status,role
DATA_FOUNDATION,Raw Governance,COMPLETE,Scientific Core
DATA_FOUNDATION,Data Quality Gates,COMPLETE,Scientific Core
DATA_FOUNDATION,Point-in-Time Governance,COMPLETE,Scientific Core
LOGISTICS_INTELLIGENCE,Route Engine,COMPLETE,Scientific Core
LOGISTICS_INTELLIGENCE,Spatiotemporal Core,COMPLETE,Scientific Core
LOGISTICS_INTELLIGENCE,Expected Freight OOT,COMPLETE,Scientific Core
LOGISTICS_INTELLIGENCE,Shipping Friction,COMPLETE,Scientific Core
LOGISTICS_INTELLIGENCE,Critical Shipping Friction,COMPLETE,Ex-post Diagnostic
PREDICTIVE_AI,Late Risk Scientific Core,COMPLETE,Scientific Core
PREDICTIVE_AI,Late Risk Production Benchmark,PLANNED,Product Layer
PREDICTIVE_AI,Shipping Intelligence Feature Ablation,PLANNED,Product Layer
PREDICTIVE_AI,ETA,PLANNED,Product Layer
PREDICTIVE_AI,Probabilistic ETA,PLANNED,Product Layer
PREDICTIVE_AI,Explainability,PLANNED,Product Layer
DECISION_ENGINE,Recall Top-K Operational Ranking,PLANNED,Product Layer
DECISION_ENGINE,Expected Value Decisioning,PLANNED,Product Layer
MLOPS_CLOUD,SQL Medallion,PLANNED,Production Infrastructure
MLOPS_CLOUD,MLflow,PLANNED,Production Infrastructure
MLOPS_CLOUD,Model Registry,PLANNED,Production Infrastructure
MLOPS_CLOUD,FastAPI,PLANNED,Production Infrastructure
MLOPS_CLOUD,Docker,PLANNED,Production Infrastructure
MLOPS_CLOUD,AWS,PLANNED,Production Infrastructure
MLOPS_CLOUD,GitHub Actions,PLANNED,Production Infrastructure
MLOPS_CLOUD,Monitoring and Drift,PLANNED,Production Infrastructure
CSV

# 10. PYTHON VALIDATION
python3 - <<'PY'
from pathlib import Path

root = Path.home() / "workspace" / "Delivery_Risk_GitHub_Staging"
files = sorted((root / "src").rglob("*.py"))
failures = []

for p in files:
    try:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
    except Exception as exc:
        failures.append((p.relative_to(root).as_posix(), repr(exc)))

print("PYTHON_TOTAL =", len(files))
print("PYTHON_PASS  =", len(files) - len(failures))
print("PYTHON_FAIL  =", len(failures))

for p, exc in failures:
    print("[FAIL]", p, exc)

if len(files) != 120:
    raise SystemExit(f"Expected 120 Python files, observed {len(files)}")

if failures:
    raise SystemExit(1)

print("[PASS] 120/120 Python source files compile in-memory.")
PY

# 11. SCIENTIFIC CORE PRESERVATION
PY_AFTER="$(find src -type f -name '*.py' | wc -l)"
LEGACY_AFTER="$(find legacy/source_history -type f -name '*.py.txt' 2>/dev/null | wc -l || true)"

[[ "$PY_BEFORE" -eq "$PY_AFTER" ]]
[[ "$LEGACY_BEFORE" -eq "$LEGACY_AFTER" ]]

echo
echo "PYTHON_SOURCE_AFTER = $PY_AFTER"
echo "LEGACY_AFTER        = $LEGACY_AFTER"
echo "[PASS] Scientific Core preserved."

# 12. RAW / EXTERNAL GOVERNANCE
RAW_TRACKED="$(git ls-files | grep '^data/raw/' | grep -v '/README.md$' | grep -v '/\.gitkeep$' | wc -l || true)"
EXTERNAL_TRACKED="$(git ls-files | grep '^data/external/' | grep -v '/README.md$' | grep -v '/\.gitkeep$' | wc -l || true)"

echo
echo "TRACKED_RAW_REAL_FILES      = $RAW_TRACKED"
echo "TRACKED_EXTERNAL_REAL_FILES = $EXTERNAL_TRACKED"

[[ "$RAW_TRACKED" -eq 0 ]]
[[ "$EXTERNAL_TRACKED" -eq 0 ]]

# 13. CONTROLLED GIT ADD
if [[ "$STAGED_BEFORE" -eq 0 ]]; then

    git add \
        configs/product_architecture_v1.json \
        metadata/code_catalog.csv \
        metadata/product_module_status.csv \
        docs/architecture \
        src/production \
        sql \
        api \
        mlops \
        infra \
        tests

    git add -f \
        data/bronze/README.md \
        data/silver/README.md \
        data/gold/README.md

    BAD_RAW="$(git diff --cached --name-only | grep -E '^data/(raw|external)/' | wc -l || true)"
    echo "INDEXED_RAW_OR_EXTERNAL = $BAD_RAW"
    [[ "$BAD_RAW" -eq 0 ]]

    FINAL_STATUS="REPOSITORY_ARCHITECTURE_STAGED_FOR_REVIEW"

else

    echo "[INFO] Alterações staged anteriores preservadas."
    echo "[INFO] Arquitetura NÃO adicionada ao index nesta execução."
    FINAL_STATUS="ARCHITECTURE_CREATED_WAITING_FOR_PREVIOUS_COMMIT"

fi

# 14. FINAL
echo
echo "===================================================================================================="
echo "FINAL"
echo "===================================================================================================="
echo
echo "SCIENTIFIC_PYTHON = $PY_AFTER"
echo "LEGACY_PRESERVED  = $LEGACY_AFTER"
echo "PYTHON_FAIL       = 0"
echo "RAW_TRACKED       = $RAW_TRACKED"
echo "EXTERNAL_TRACKED  = $EXTERNAL_TRACKED"
echo "COMMIT_PERFORMED  = false"
echo "PUSH_PERFORMED    = false"
echo
echo "STATUS = $FINAL_STATUS"
echo
echo "GIT STATUS:"
git status --short
echo
echo "[OK] Shell alive."

```

### Como executar depois do upload

Depois que este arquivo estiver dentro de `~/workspace/Delivery_Risk_GitHub_Staging`, rode:

```text
awk '/^```bash$/{inside=1; next} /^```$/ && inside{exit} inside' \
  DELIVERY_REPOSITORY_ORGANIZATION_MASTER.md \
  > /tmp/delivery_repository_master.sh

bash -n /tmp/delivery_repository_master.sh || exit 1
bash /tmp/delivery_repository_master.sh
```

### Status possíveis

`ARCHITECTURE_CREATED_WAITING_FOR_PREVIOUS_COMMIT`

Significa que ainda há alterações staged do passo anterior; a arquitetura foi criada, mas não foi adicionada ao index.

`REPOSITORY_ARCHITECTURE_STAGED_FOR_REVIEW`

Significa que o Git estava limpo e a nova organização foi preparada para revisão.
