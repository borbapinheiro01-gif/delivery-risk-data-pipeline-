#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
PROJECT MATHEMATICAL SPEC & RESULTS
Delivery Risk Intelligence Platform
===============================================================================

Gera um único arquivo Markdown contendo:

- formulação matemática;
- definição temporal;
- candidatos a label;
- restrições de informação;
- métricas planejadas;
- resultados dos Gates 01, 02, 03, 03B e 04;
- Registry 1.1;
- cardinalidades;
- readiness;
- warnings;
- questões ainda abertas;
- localização dos artefatos.

O documento é REGERADO a partir dos resultados do projeto.
Não altera RAW.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


PROJECT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

DOCS = (
    PROJECT
    / "docs"
)

DOCS.mkdir(
    parents=True,
    exist_ok=True
)

OUT = (
    DOCS
    / "PROJECT_MATHEMATICAL_SPEC_AND_RESULTS.md"
)


# =============================================================================
# HELPERS
# =============================================================================

def load_json(
    path
):

    if not path.exists():

        return {}

    try:

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def load_csv(
    path
):

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path,
            low_memory=False
        )

    except Exception:

        return pd.DataFrame()


def fmt(
    value,
    digits=6
):

    if value is None:

        return "N/A"

    if isinstance(
        value,
        (
            float,
            np.floating
        )
    ):

        if np.isnan(
            value
        ):

            return "N/A"

        return (
            f"{value:.{digits}f}"
        )

    if isinstance(
        value,
        (
            int,
            np.integer
        )
    ):

        return (
            f"{int(value):,}"
        )

    return str(
        value
    )


def dataframe_to_markdown(
    df,
    max_rows=None
):

    if df.empty:

        return "_Não disponível._"

    x = df.copy()

    if (
        max_rows is not None
        and
        len(
            x
        )
        >
        max_rows
    ):

        x = x.head(
            max_rows
        )

    cols = list(
        x.columns
    )

    lines = []

    lines.append(
        "| "
        +
        " | ".join(
            str(c)
            for c in cols
        )
        +
        " |"
    )

    lines.append(
        "| "
        +
        " | ".join(
            "---"
            for _ in cols
        )
        +
        " |"
    )


    for _, row in x.iterrows():

        values = []

        for col in cols:

            value = row[
                col
            ]

            if pd.isna(
                value
            ):

                text = ""

            elif isinstance(
                value,
                float
            ):

                text = (
                    f"{value:.6f}"
                )

            else:

                text = str(
                    value
                )

            text = (
                text
                .replace(
                    "|",
                    "\\|"
                )
                .replace(
                    "\n",
                    " "
                )
            )

            values.append(
                text
            )

        lines.append(
            "| "
            +
            " | ".join(
                values
            )
            +
            " |"
        )

    return "\n".join(
        lines
    )


# =============================================================================
# LOAD RESULTS
# =============================================================================

gate1 = load_json(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_01_structural"
    / "dq_gate_01_summary.json"
)

gate2 = load_json(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_02_semantic"
    / "dq_gate_02_summary.json"
)

gate3 = load_json(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03_statistical"
    / "dq_gate_03_summary.json"
)

gate3b = load_json(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
    / "dq_gate_03b_summary.json"
)

gate4 = load_json(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "dq_gate_04_summary.json"
)

dataset_manifest = load_json(
    PROJECT
    / "metadata"
    / "dataset_manifest.json"
)

registry = load_json(
    PROJECT
    / "metadata"
    / "registry_v1_1"
    / "registry_v1_1_manifest.json"
)


gate2_exceptions = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_02_semantic"
    / "dq_gate_02_exceptions.csv"
)


numeric_profile = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03_statistical"
    / "03_numeric_statistical_profile.csv"
)


readiness = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
    / "05_task_feature_readiness_summary.csv"
)


missing_reasons = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
    / "07_missing_reason_registry.csv"
)


source_vs_task = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
    / "08_source_vs_task_completeness.csv"
)


multi = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
    / "09_multi_entity_structure.csv"
)


canonical_issues = load_csv(
    PROJECT
    / "metadata"
    / "registry_v1_1"
    / "canonical_issue_registry.csv"
)


relationships = load_csv(
    PROJECT
    / "metadata"
    / "registry_v1_1"
    / "relationship_registry_v1_1.csv"
)


label_comparison = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "02_label_definition_comparison.csv"
)


label_confusion = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "03_label_confusion_matrix.csv"
)


margin_profile = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "05_lateness_margin_profile.csv"
)


margin_bins = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "06_lateness_margin_bins.csv"
)


observation_window = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "08_observation_window_audit.csv"
)


label_contract = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "11_label_contract_candidates.csv"
)


gate4_exceptions = load_csv(
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
    / "dq_gate_04_exceptions.csv"
)


# =============================================================================
# STATUS TABLE
# =============================================================================

status_df = pd.DataFrame(
    [
        {
            "Componente":
                "DQ Gate 01 — Structural",

            "Status":
                gate1.get(
                    "status",
                    "N/A"
                )
        },

        {
            "Componente":
                "DQ Gate 02 — Semantic / Validity",

            "Status":
                gate2.get(
                    "status",
                    "N/A"
                )
        },

        {
            "Componente":
                "DQ Gate 03 — Statistical / Completeness",

            "Status":
                gate3.get(
                    "status",
                    "N/A"
                )
        },

        {
            "Componente":
                "Knowledge & Truth Registry 1.1",

            "Status":
                registry.get(
                    "internal_registry_status",
                    "N/A"
                )
        },

        {
            "Componente":
                "DQ Gate 03B — Conditional / Task",

            "Status":
                gate3b.get(
                    "status",
                    "N/A"
                )
        },

        {
            "Componente":
                "DQ Gate 04 — Label / Construct",

            "Status":
                gate4.get(
                    "status",
                    "N/A"
                )
        }
    ]
)


# =============================================================================
# DOCUMENT
# =============================================================================

md = []


md.append(
    "# Delivery Risk Intelligence Platform"
)

md.append(
    ""
)

md.append(
    "## Mathematical Specification, Data Contract & Experimental State"
)

md.append(
    ""
)

md.append(
    f"**Última geração:** "
    f"{datetime.now(timezone.utc).isoformat()}"
)

md.append(
    ""
)

md.append(
    "> Este documento é gerado automaticamente a partir dos artefatos "
    "do projeto. Ele deve ser tratado como o mapa central da formulação "
    "matemática, das decisões metodológicas e dos resultados de auditoria."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 1
# -----------------------------------------------------------------------------

md.append(
    "## 1. Objetivo do sistema"
)

md.append(
    ""
)

md.append(
    "No instante da compra, estimar a probabilidade de que um pedido "
    "seja entregue após a promessa de entrega informada ao consumidor."
)

md.append(
    ""
)

md.append(
    "A saída futura do sistema será um **score de risco de atraso**, "
    "utilizado para priorizar ações preventivas."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 2
# -----------------------------------------------------------------------------

md.append(
    "## 2. Unidade estatística e instante de decisão"
)

md.append(
    ""
)

md.append(
    r"A unidade de análise é:"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\boxed{\text{1 observação} = \text{1 order\_id}}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    r"O instante de previsão é:"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\boxed{t_0 = \text{order\_purchase\_timestamp}}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "Portanto, uma feature só é elegível quando sua disponibilidade "
    "em produção pode ser demonstrada em ou antes de "
    r"\(t_0\)."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 3
# -----------------------------------------------------------------------------

md.append(
    "## 3. Notação matemática"
)

md.append(
    ""
)

md.append(
    r"Para cada pedido \(i\):"
)

md.append(
    ""
)

md.append(
    r"- \(X_i(t_0)\): vetor de informações disponíveis no instante da compra;"
)

md.append(
    r"- \(T_i^{p}\): timestamp/data prometida de entrega;"
)

md.append(
    r"- \(T_i^{d}\): timestamp observado de entrega ao cliente;"
)

md.append(
    r"- \(M_i=T_i^{d}-T_i^{p}\): margem temporal da entrega em relação à promessa;"
)

md.append(
    r"- \(Y_i\): indicador de violação da promessa."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 4
# -----------------------------------------------------------------------------

md.append(
    "## 4. Formulações candidatas do target"
)

md.append(
    ""
)

md.append(
    "### 4.1 Definição por timestamp estrito"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"Y_i^{(\mathrm{ts})}"
    r"="
    r"\mathbf{1}"
    r"\left("
    r"T_i^{d}>T_i^{p}"
    r"\right)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "Nesta definição, qualquer entrega posterior ao instante exato "
    "armazenado em `order_estimated_delivery_date` é classificada como atraso."
)

md.append(
    ""
)

md.append(
    "### 4.2 Definição por dia-calendário"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"Y_i^{(\mathrm{day})}"
    r"="
    r"\mathbf{1}"
    r"\left("
    r"\operatorname{date}(T_i^{d})"
    r">"
    r"\operatorname{date}(T_i^{p})"
    r"\right)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "Nesta definição, uma entrega realizada durante o próprio dia "
    "prometido permanece classificada como on-time."
)

md.append(
    ""
)

md.append(
    "### 4.3 Sensibilidade com um dia de tolerância"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"Y_i^{(+1)}"
    r"="
    r"\mathbf{1}"
    r"\left("
    r"\operatorname{date}(T_i^{d})"
    r">"
    r"\operatorname{date}(T_i^{p})+1"
    r"\right)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "Esta terceira formulação é apenas análise de sensibilidade. "
    "Ela não possui, neste momento, justificativa empresarial para uso primário."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 5
# -----------------------------------------------------------------------------

md.append(
    "## 5. Modelo probabilístico"
)

md.append(
    ""
)

md.append(
    r"O modelo futuro estima:"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\hat p_i"
    r"="
    r"f_\theta"
    r"\left("
    r"X_i(t_0)"
    r"\right)"
    r"\approx"
    r"P"
    r"\left("
    r"Y_i=1"
    r"\mid"
    r"X_i(t_0)"
    r"\right)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    r"onde \(\hat p_i\in[0,1]\) representa o risco estimado de atraso."
)

md.append(
    ""
)

md.append(
    "A decisão operacional poderá usar um threshold "
    r"\(\tau\):"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\hat Y_i"
    r"="
    r"\mathbf{1}"
    r"\left("
    r"\hat p_i\ge\tau"
    r"\right)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "O valor de "
    r"\(\tau\)"
    " não será escolhido por conveniência estatística; deverá refletir "
    "capacidade operacional e custo dos erros."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 6
# -----------------------------------------------------------------------------

md.append(
    "## 6. Função de custo operacional"
)

md.append(
    ""
)

md.append(
    r"Uma formulação futura possível é:"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"C(\tau)"
    r"="
    r"c_{FN}FN(\tau)"
    r"+"
    r"c_{FP}FP(\tau)"
    r"+"
    r"c_A A(\tau)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    r"onde \(c_{FN}\) representa o custo de não detectar um atraso, "
    r"\(c_{FP}\) o custo de agir sobre um pedido que não atrasaria, "
    r"e \(c_A\) o custo da própria intervenção."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 7 metrics
# -----------------------------------------------------------------------------

md.append(
    "## 7. Métricas previstas para os modelos"
)

md.append(
    ""
)

md.append(
    r"### Precision"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\mathrm{Precision}"
    r"="
    r"\frac{TP}{TP+FP}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    r"### Recall"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\mathrm{Recall}"
    r"="
    r"\frac{TP}{TP+FN}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    r"### \(F_\beta\)"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"F_\beta"
    r"="
    r"(1+\beta^2)"
    r"\frac{PR}{\beta^2P+R}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "Se a prioridade operacional for capturar atrasos, "
    r"\(\beta>1\)"
    " poderá ser utilizado."
)

md.append(
    ""
)

md.append(
    "### Brier Score"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\mathrm{Brier}"
    r"="
    r"\frac{1}{n}"
    r"\sum_{i=1}^{n}"
    r"(\hat p_i-Y_i)^2"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "### Log Loss"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\mathrm{LogLoss}"
    r"="
    r"-\frac1n"
    r"\sum_i"
    r"\left["
    r"Y_i\log\hat p_i"
    r"+"
    r"(1-Y_i)\log(1-\hat p_i)"
    r"\right]"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "### Capture@K"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\mathrm{Capture@K}"
    r"="
    r"\frac"
    r"{\text{atrasos presentes nos K pedidos de maior risco}}"
    r"{\text{total de atrasos}}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "A métrica técnica primária prevista para comparação de classificadores "
    "é **PR-AUC**, acompanhada de recall, precision e calibração."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 8 DQ math
# -----------------------------------------------------------------------------

md.append(
    "## 8. Formulação matemática da qualidade dos dados"
)

md.append(
    ""
)

md.append(
    "### 8.1 Cobertura relacional"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"C_R"
    r"="
    r"1-"
    r"\frac"
    r"{N_{\mathrm{relation\ missing}}}"
    r"{N_{\mathrm{applicable}}}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "### 8.2 Completude condicional de atributo"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"C_{A\mid R}"
    r"="
    r"1-"
    r"P"
    r"\left("
    r"A\text{ ausente}"
    r"\mid"
    r"R\text{ existe}"
    r"\right)"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    "Portanto:"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\boxed{"
    r"\mathrm{RELATION\_MISSING}"
    r"\neq"
    r"\mathrm{ATTRIBUTE\_MISSING}"
    r"}"
)

md.append(
    r"\]"
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\boxed{"
    r"\mathrm{SOURCE\ QUALITY}"
    r"\neq"
    r"\mathrm{TASK\ QUALITY}"
    r"}"
)

md.append(
    r"\]"
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 9 status
# -----------------------------------------------------------------------------

md.append(
    "## 9. Estado dos controles de Data Quality"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        status_df
    )
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 10 dataset/registry
# -----------------------------------------------------------------------------

md.append(
    "## 10. Dataset Knowledge Registry"
)

md.append(
    ""
)

md.append(
    f"- Tabelas mapeadas: **{fmt(registry.get('tables'))}**"
)

md.append(
    f"- Colunas mapeadas: **{fmt(registry.get('columns'))}**"
)

md.append(
    f"- Relacionamentos: **{fmt(registry.get('relationships'))}**"
)

md.append(
    f"- Questões brutas: **{fmt(registry.get('raw_open_issues'))}**"
)

md.append(
    f"- Questões canônicas: **{fmt(registry.get('canonical_open_issues'))}**"
)

md.append(
    f"- Search records: **{fmt(registry.get('search_records'))}**"
)

md.append(
    ""
)

md.append(
    "### Cardinalidades semânticas e observadas"
)

md.append(
    ""
)


relationship_cols = [
    c
    for c in [
        "parent_table",
        "child_table",
        "semantic_cardinality",
        "observed_cardinality",
        "child_rows_per_parent_mean",
        "child_rows_per_parent_max",
        "child_orphan_rows_recomputed"
    ]
    if c in relationships.columns
]


md.append(
    dataframe_to_markdown(
        relationships[
            relationship_cols
        ]
        if relationship_cols
        else relationships
    )
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# Gate 02
# -----------------------------------------------------------------------------

md.append(
    "## 11. DQ Gate 02 — warnings semânticos conhecidos"
)

md.append(
    ""
)


gate2_cols = [
    c
    for c in [
        "status",
        "check_id",
        "table",
        "variable",
        "affected_rows",
        "affected_pct",
        "observed"
    ]
    if c in gate2_exceptions.columns
]


md.append(
    dataframe_to_markdown(
        gate2_exceptions[
            gate2_cols
        ]
        if gate2_cols
        else gate2_exceptions
    )
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# Gate03
# -----------------------------------------------------------------------------

md.append(
    "## 12. DQ Gate 03 — perfil estatístico numérico"
)

md.append(
    ""
)


numeric_cols = [
    c
    for c in [
        "table",
        "variable",
        "missing_pct",
        "min",
        "median",
        "mean",
        "p99",
        "max",
        "skewness",
        "iqr_candidate_pct",
        "mad_candidate_pct"
    ]
    if c in numeric_profile.columns
]


md.append(
    dataframe_to_markdown(
        numeric_profile[
            numeric_cols
        ]
        if numeric_cols
        else numeric_profile
    )
)

md.append(
    ""
)

md.append(
    "Extremos detectados por IQR/MAD permanecem **diagnósticos**, "
    "não regras automáticas de exclusão."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# Gate03B
# -----------------------------------------------------------------------------

md.append(
    "## 13. DQ Gate 03B — Conditional & Task Completeness"
)

md.append(
    ""
)

md.append(
    f"- Source orders: **{fmt(gate3b.get('source_orders'))}**"
)

md.append(
    f"- Task candidate orders: **{fmt(gate3b.get('task_candidate_orders'))}**"
)

md.append(
    f"- Task/source: **{fmt(gate3b.get('task_candidate_pct'), 4)}%**"
)

md.append(
    f"- Critical failures: **{fmt(gate3b.get('critical_failures'))}**"
)

md.append(
    f"- Warnings: **{fmt(gate3b.get('warnings'))}**"
)

md.append(
    ""
)

md.append(
    "### Feature readiness da coorte"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        readiness
    )
)

md.append(
    ""
)

md.append(
    "### Taxonomia de ausência"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        missing_reasons
    )
)

md.append(
    ""
)

md.append(
    "### Source quality vs task quality"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        source_vs_task
    )
)

md.append(
    ""
)

md.append(
    "### Estrutura multi-entidade"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        multi
    )
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# Gate04
# -----------------------------------------------------------------------------

md.append(
    "## 14. DQ Gate 04 — Label / Construct Validity"
)

md.append(
    ""
)

md.append(
    f"- Status: **{gate4.get('status', 'N/A')}**"
)

md.append(
    f"- Observation end: **{gate4.get('observation_end', 'N/A')}**"
)

md.append(
    f"- Estimated timestamps at midnight: "
    f"**{fmt(gate4.get('estimated_midnight_pct'))}%**"
)

md.append(
    f"- Label disagreements: "
    f"**{fmt(gate4.get('label_disagreements'))}** "
    f"(**{fmt(gate4.get('label_disagreement_pct'))}%**)"
)

md.append(
    f"- Same-day disagreements: "
    f"**{fmt(gate4.get('same_day_disagreements'))}** "
    f"(**{fmt(gate4.get('same_day_disagreement_pct'))}% "
    f"dos desacordos**)"
)

md.append(
    f"- Construct recommendation: "
    f"**{gate4.get('construct_recommendation', 'N/A')}**"
)

md.append(
    f"- Label finalizado automaticamente: "
    f"**{gate4.get('label_finalized', False)}**"
)

md.append(
    ""
)

md.append(
    "### Comparação das definições"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        label_comparison
    )
)

md.append(
    ""
)

md.append(
    "### Matriz de concordância"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        label_confusion
    )
)

md.append(
    ""
)

md.append(
    "### Distribuição da margem"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        margin_profile
    )
)

md.append(
    ""
)

md.append(
    "### Faixas de adiantamento/atraso"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        margin_bins
    )
)

md.append(
    ""
)

md.append(
    "### Janela de observação"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        observation_window
    )
)

md.append(
    ""
)

md.append(
    "### Label Contract — candidatos"
)

md.append(
    ""
)

md.append(
    dataframe_to_markdown(
        label_contract
    )
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 15 forbidden features
# -----------------------------------------------------------------------------

md.append(
    "## 15. Campos proibidos no baseline em t0"
)

md.append(
    ""
)

md.append(
    "`order_status` final, `order_approved_at`, "
    "`order_delivered_carrier_date`, "
    "`order_delivered_customer_date` e reviews "
    "não podem ser utilizados como features do modelo preventivo em "
    "`order_purchase_timestamp`."
)

md.append(
    ""
)

md.append(
    "`shipping_limit_date` permanece em HOLD até confirmação de sua "
    "disponibilidade operacional em t0."
)

md.append(
    ""
)

md.append(
    "Os campos de pagamento permanecem candidatos sujeitos à confirmação "
    "de point-in-time availability."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 16 open issues
# -----------------------------------------------------------------------------

md.append(
    "## 16. Questões canônicas abertas"
)

md.append(
    ""
)


issue_cols = [
    c
    for c in [
        "canonical_issue_id",
        "canonical_topic",
        "priority",
        "source_issue_count",
        "source_issue_ids",
        "question_summary",
        "required_actions"
    ]
    if c in canonical_issues.columns
]


md.append(
    dataframe_to_markdown(
        canonical_issues[
            issue_cols
        ]
        if issue_cols
        else canonical_issues
    )
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 17 current decisions
# -----------------------------------------------------------------------------

md.append(
    "## 17. Decisões metodológicas atualmente estabelecidas"
)

md.append(
    ""
)

md.append(
    "1. A unidade estatística do modelo é `order_id`."
)

md.append(
    "2. O instante de previsão base é `order_purchase_timestamp`."
)

md.append(
    "3. Child tables (`order_items`, `payments`) precisam ser agregadas "
    "antes da modelagem em nível de pedido."
)

md.append(
    "4. Missing de relação e missing de atributo são fenômenos diferentes."
)

md.append(
    "5. Source Quality e Task Quality são medidas separadamente."
)

md.append(
    "6. Informação futura não entra no baseline mesmo que seja 100% completa."
)

md.append(
    "7. Resultado não observado não é convertido automaticamente em "
    "`on-time`."
)

md.append(
    "8. O label final requer contrato explícito de construct validity."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 18 provenance/ground truth
# -----------------------------------------------------------------------------

md.append(
    "## 18. Proveniência e limites da verdade observável"
)

md.append(
    ""
)

md.append(
    "Os valores existentes nos CSVs constituem a **source truth da "
    "distribuição pública**, mas não temos acesso aos sistemas transacionais "
    "privados originais para confirmação independente."
)

md.append(
    ""
)

md.append(
    r"\["
)

md.append(
    r"\boxed{"
    r"\text{internal consistency}"
    r"\neq"
    r"\text{externally verified ground truth}"
    r"}"
)

md.append(
    r"\]"
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 19 methodological evidence
# -----------------------------------------------------------------------------

md.append(
    "## 19. Referências metodológicas verificadas"
)

md.append(
    ""
)

md.append(
    "- **Olist — Brazilian E-Commerce Public Dataset:** dataset comercial "
    "anonimizado; a data estimada é informada ao cliente no momento da "
    "compra; pedidos podem possuir múltiplos itens e sellers."
)

md.append(
    "- **NIST AI Risk Management Framework / Playbook, Measure 2.5:** "
    "proxies e indicadores devem demonstrar construct validity, isto é, "
    "medir o conceito que afirmam representar."
)

md.append(
    "- **Survival Analysis / time-to-event methodology:** quando o evento "
    "não pode ser observado até o final da janela de acompanhamento, a "
    "observação não deve ser simplesmente transformada em evento negativo."
)

md.append(
    "- **Croissant / MLCommons:** metadata de datasets deve tornar estrutura, "
    "recursos e semântica legíveis por máquina e reutilizáveis."
)

md.append(
    "- **Conditional data validation:** regras de qualidade devem ser "
    "avaliadas sobre a população em que são semanticamente aplicáveis."
)

md.append(
    ""
)


# -----------------------------------------------------------------------------
# 20 artifacts
# -----------------------------------------------------------------------------

md.append(
    "## 20. Artefatos centrais do projeto"
)

md.append(
    ""
)

artifacts = [
    "metadata/column_catalog.csv",
    "metadata/truth_provenance_registry.csv",
    "metadata/temporal_availability_registry.csv",
    "metadata/registry_v1_1/canonical_issue_registry.csv",
    "metadata/registry_v1_1/ml_grain_policy_registry.csv",
    "metadata/registry_v1_1/croissant_1_1_FIXED.json",
    "metadata/registry_v1_1/croissant_1_1_MLCROISSANT_COMPAT.json",
    "reports/data_quality/gate_03b_conditional_task/DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt",
    "reports/data_quality/gate_04_label_construct/DQ_GATE_04_LABEL_CONSTRUCT_REPORT.txt",
    "reports/data_quality/gate_04_label_construct/dq_gate_04_summary.json",
    "docs/PROJECT_MATHEMATICAL_SPEC_AND_RESULTS.md",
]


for path in artifacts:

    md.append(
        f"- `{path}`"
    )


md.append(
    ""
)


# -----------------------------------------------------------------------------
# 21 next
# -----------------------------------------------------------------------------

md.append(
    "## 21. Próximas decisões"
)

md.append(
    ""
)

md.append(
    "A sequência metodológica após o Gate 04 é:"
)

md.append(
    ""
)

md.append(
    "```text"
)

md.append(
    "Label Contract final"
)

md.append(
    "        ↓"
)

md.append(
    "Point-in-Time / Leakage Gate"
)

md.append(
    "        ↓"
)

md.append(
    "Representativeness / Shift"
)

md.append(
    "        ↓"
)

md.append(
    "Data Treatment Plan"
)

md.append(
    "        ↓"
)

md.append(
    "Silver order-level"
)

md.append(
    "        ↓"
)

md.append(
    "Feature Engineering"
)

md.append(
    "        ↓"
)

md.append(
    "Temporal Train / Validation / Test"
)

md.append(
    "        ↓"
)

md.append(
    "Baseline + ML Models"
)

md.append(
    "        ↓"
)

md.append(
    "Calibration + Business Threshold"
)

md.append(
    "        ↓"
)

md.append(
    "Deploy / Monitoring / MLOps"
)

md.append(
    "```"
)

md.append(
    ""
)


# =============================================================================
# WRITE
# =============================================================================

OUT.write_text(
    "\n".join(
        md
    ),
    encoding="utf-8"
)


print("=" * 100)

print(
    "PROJECT MATHEMATICAL SPEC & RESULTS"
)

print("=" * 100)

print()

print(
    f"[OK] Documento gerado: {OUT}"
)

print()

print(
    f"Tamanho: {OUT.stat().st_size:,} bytes"
)

print()

print(
    "[OK] O documento foi construído a partir dos artefatos atuais."
)

print(
    "[OK] Nenhum RAW foi modificado."
)

