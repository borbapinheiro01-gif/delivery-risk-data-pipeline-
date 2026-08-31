#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

import pandas as pd


PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

DOC = (
    PROJECT
    / "docs"
    / "PROJECT_MATHEMATICAL_SPEC_AND_RESULTS.md"
)

BACKUP = (
    PROJECT
    / "docs"
    / "backups"
    / "PROJECT_MATHEMATICAL_SPEC_AND_RESULTS_BEFORE_GATE_04B.md"
)

REGISTRY = (
    PROJECT
    / "metadata"
    / "registry_v1_2"
    / "canonical_issue_registry.csv"
)

CONTRACT = (
    PROJECT
    / "contracts"
    / "DELIVERY_RISK_LABEL_CONTRACT_V1.json"
)


if not DOC.exists():
    raise SystemExit(
        f"[ERRO] Documento mestre ausente: {DOC}"
    )


if not REGISTRY.exists():
    raise SystemExit(
        f"[ERRO] Registry 1.2 ausente: {REGISTRY}"
    )


if not CONTRACT.exists():
    raise SystemExit(
        f"[ERRO] Label Contract ausente: {CONTRACT}"
    )


# =============================================================================
# BACKUP
# =============================================================================

BACKUP.parent.mkdir(
    parents=True,
    exist_ok=True
)


shutil.copy2(
    DOC,
    BACKUP
)


text = DOC.read_text(
    encoding="utf-8"
)


with CONTRACT.open(
    "r",
    encoding="utf-8"
) as f:

    contract = json.load(f)


issues = pd.read_csv(
    REGISTRY,
    low_memory=False
)


# =============================================================================
# 1. CORRIGIR A FÓRMULA PROBABILÍSTICA
# =============================================================================

text = text.replace(
    r"\hat p_i=f_\theta\left(X_i(t_0)\right)\approxP\left(Y_i=1\midX_i(t_0)\right)",
    r"\hat p_i=f_\theta\left(X_i(t_0)\right)\approx P\!\left(Y_i=1\mid X_i(t_0)\right)"
)


# =============================================================================
# 2. ATUALIZAR REGISTRY NO STATUS PRINCIPAL
# =============================================================================

text = text.replace(
    "| Knowledge & Truth Registry 1.1 | PASS |",
    "| Knowledge & Truth Registry 1.2 | PASS |"
)


# =============================================================================
# 3. ATUALIZAR STATUS DO LABEL NA SEÇÃO 14
# =============================================================================

text = text.replace(
    "- Construct recommendation: **CALENDAR_DAY_RECOMMENDED_PENDING_BUSINESS_CONFIRMATION**",
    (
        "- Construct recommendation do Gate 04: "
        "**CALENDAR_DAY_RECOMMENDED_PENDING_BUSINESS_CONFIRMATION**\n"
        "- Modeling Label Contract após Gate 04B: "
        "**CALENDAR_DAY — FROZEN_FOR_MODELING**\n"
        "- External Olist historical SLA: "
        "**NOT_VERIFIED**"
    )
)


text = text.replace(
    "- Label finalizado automaticamente: **False**",
    (
        "- Label finalizado automaticamente pelo Gate 04: **False**\n"
        "- Label posteriormente congelado pelo Gate 04B: "
        "**True — para este case de modelagem**"
    )
)


# =============================================================================
# 4. CRIAR SEÇÃO 14B
# =============================================================================

section_14b = r"""
## 14B. Label Contract Freeze — Gate 04B

O contrato do target foi congelado para a modelagem deste projeto.

\[
\boxed{
Y_i =
\mathbf{1}
\left[
\operatorname{date}
(\texttt{order\_delivered\_customer\_date}_i)
>
\operatorname{date}
(\texttt{order\_estimated\_delivery\_date}_i)
\right]
}
\]

### Estado do contrato

- **Contract ID:** `DELIVERY_RISK_LABEL_V1`
- **Versão:** `1.0.0`
- **Status:** `FROZEN_FOR_MODELING`
- **Unidade estatística:** `order_id`
- **Instante de previsão:** `order_purchase_timestamp`
- **População supervisionada:** 96.470 pedidos
- **Pedidos atrasados:** 6.534
- **Pedidos não atrasados:** 89.936
- **Prevalência positiva:** 6,773090%
- **No-skill PR-AUC:** 0,0677309

### Limite de interpretação

O contrato acima é a definição formal adotada **neste case de modelagem**.

Ele não constitui afirmação de que o SLA operacional privado e histórico
da Olist tenha sido independentemente verificado.

\[
\boxed{
\text{modeling contract frozen}
\neq
\text{external Olist SLA verified}
}
\]

As definições `TIMESTAMP_STRICT` e `CALENDAR_DAY_PLUS_1_GRACE`
permanecem apenas como análises de sensibilidade.

"""


marker_15 = (
    "## 15. Campos proibidos no baseline em t0"
)


if "## 14B. Label Contract Freeze — Gate 04B" in text:

    start = text.index(
        "## 14B. Label Contract Freeze — Gate 04B"
    )

    end = text.index(
        marker_15,
        start
    )

    text = (
        text[:start]
        +
        section_14b
        +
        text[end:]
    )

else:

    text = text.replace(
        marker_15,
        section_14b
        +
        marker_15
    )


# =============================================================================
# 5. RECONSTRUIR SEÇÃO 16 COM STATUS SINCRONIZADO
# =============================================================================

def safe(v):

    if pd.isna(v):
        return ""

    return (
        str(v)
        .replace(
            "|",
            "\\|"
        )
        .replace(
            "\n",
            " "
        )
    )


cols = [
    "canonical_issue_id",
    "canonical_topic",
    "priority",
    "resolution_status",
    "resolved_by",
    "resolution_decision",
    "external_truth_status"
]


table = []

table.append(
    "| "
    +
    " | ".join(
        cols
    )
    +
    " |"
)

table.append(
    "| "
    +
    " | ".join(
        "---"
        for _ in cols
    )
    +
    " |"
)


for _, row in issues.iterrows():

    table.append(
        "| "
        +
        " | ".join(
            safe(
                row.get(
                    col,
                    ""
                )
            )
            for col in cols
        )
        +
        " |"
    )


section16 = (
    "## 16. Questões canônicas — estado sincronizado\n\n"
    +
    "O Registry 1.2 preserva as questões históricas, "
    "mas distingue explicitamente questões abertas de avaliações já resolvidas.\n\n"
    +
    "\n".join(
        table
    )
    +
    "\n\n"
)


start_marker = (
    "## 16. Questões canônicas abertas"
)

end_marker = (
    "## 17. Decisões metodológicas atualmente estabelecidas"
)


if start_marker in text:

    start = text.index(
        start_marker
    )

    end = text.index(
        end_marker,
        start
    )

    text = (
        text[:start]
        +
        section16
        +
        text[end:]
    )


# =============================================================================
# 6. ATUALIZAR DECISÃO METODOLÓGICA
# =============================================================================

text = text.replace(
    "8. O label final requer contrato explícito de construct validity.",
    (
        "8. O label passou por construct validity no Gate 04.\n"
        "9. O Gate 04B congelou `CALENDAR_DAY` como contrato de modelagem.\n"
        "10. O SLA histórico privado da Olist permanece externamente não verificado."
    )
)


# =============================================================================
# 7. ADICIONAR ARTEFATOS 04B
# =============================================================================

artifact_marker = (
    "- `docs/PROJECT_MATHEMATICAL_SPEC_AND_RESULTS.md`"
)


artifact_replacement = """- `docs/PROJECT_MATHEMATICAL_SPEC_AND_RESULTS.md`
- `contracts/DELIVERY_RISK_LABEL_CONTRACT_V1.json`
- `metadata/registry_v1_2/registry_v1_2_manifest.json`
- `metadata/registry_v1_2/issue_resolution_registry.csv`
- `reports/data_quality/gate_04b_label_contract/DQ_GATE_04B_LABEL_CONTRACT_REPORT.txt`"""


text = text.replace(
    artifact_marker,
    artifact_replacement
)


# =============================================================================
# 8. ATUALIZAR PIPELINE
# =============================================================================

text = text.replace(
    "Label Contract final\n        ↓",
    "Label Contract v1.0 — FROZEN\n        ↓"
)


# =============================================================================
# 9. ATUALIZAR DATA DE SINCRONIZAÇÃO
# =============================================================================

sync_stamp = (
    datetime.now(
        timezone.utc
    )
    .isoformat()
)


text = text.replace(
    "> Este documento é gerado automaticamente a partir dos artefatos do projeto. Ele deve ser tratado como o mapa central da formulação matemática, das decisões metodológicas e dos resultados de auditoria.",
    (
        "> Este documento é gerado automaticamente a partir dos artefatos do projeto. "
        "Ele deve ser tratado como o mapa central da formulação matemática, "
        "das decisões metodológicas e dos resultados de auditoria.\n>\n"
        f"> **Última sincronização de contrato/registry:** {sync_stamp}"
    )
)


# =============================================================================
# WRITE
# =============================================================================

DOC.write_text(
    text,
    encoding="utf-8"
)


print("=" * 100)
print("DOCUMENTO-MESTRE — SYNC GATE 04B")
print("=" * 100)
print()

print(
    f"[OK] Backup: {BACKUP}"
)

print(
    f"[OK] Atualizado: {DOC}"
)

print()
print("[CORRIGIDO]")
print("- fórmula probabilística")
print("- Label Contract")
print("- Registry 1.2")
print("- status das três issues resolvidas")
print("- pipeline do projeto")
print("- artefatos 04B")

