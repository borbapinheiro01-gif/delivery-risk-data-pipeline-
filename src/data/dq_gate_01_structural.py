#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DQ GATE 01 — STRUCTURAL
Delivery Risk Intelligence Platform

Valida:
    1. existência dos arquivos
    2. quantidade de CSVs
    3. legibilidade
    4. dataset não vazio
    5. schema esperado
    6. ordem das colunas
    7. famílias de tipos
    8. NULL em campos estruturais obrigatórios
    9. PK / chave composta
   10. duplicação integral
   11. integridade referencial

NÃO valida ainda:
    - ranges de negócio
    - outliers
    - plausibilidade estatística
    - drift
    - target
    - imputação
    - qualidade semântica

FAIL crítico => bloqueia construção da Silver.
"""

from pathlib import Path
import json
import sys
import traceback

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


# ======================================================================
# PATHS
# ======================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"

CONFIG = PROJECT / "configs" / "dq_gate_01_structural.json"

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_01_structural"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# CONFIG
# ======================================================================

with CONFIG.open(
    "r",
    encoding="utf-8"
) as f:

    CONTRACT = json.load(f)


# ======================================================================
# RESULTADOS
# ======================================================================

results = []

loaded = {}


def add_result(
    check_id,
    table,
    dimension,
    severity,
    status,
    observed,
    expected,
    details=""
):

    results.append(
        {
            "gate":
                "DQ_GATE_01_STRUCTURAL",

            "check_id":
                check_id,

            "table":
                table,

            "dimension":
                dimension,

            "severity":
                severity,

            "status":
                status,

            "observed":
                str(observed),

            "expected":
                str(expected),

            "details":
                str(details),
        }
    )


def pass_check(
    check_id,
    table,
    dimension,
    observed,
    expected,
    details=""
):

    add_result(
        check_id,
        table,
        dimension,
        "CRITICAL",
        "PASS",
        observed,
        expected,
        details
    )


def fail_check(
    check_id,
    table,
    dimension,
    observed,
    expected,
    details=""
):

    add_result(
        check_id,
        table,
        dimension,
        "CRITICAL",
        "FAIL",
        observed,
        expected,
        details
    )


def warn_check(
    check_id,
    table,
    dimension,
    observed,
    expected,
    details=""
):

    add_result(
        check_id,
        table,
        dimension,
        "WARNING",
        "WARN",
        observed,
        expected,
        details
    )


# ======================================================================
# TIPO
# ======================================================================

def type_matches(series, expected):

    dtype = series.dtype

    if expected == "integer":

        return bool(
            is_integer_dtype(dtype)
        )

    if expected == "numeric":

        return bool(
            is_numeric_dtype(dtype)
        )

    if expected == "string":

        return bool(
            is_object_dtype(dtype)
            or
            is_string_dtype(dtype)
        )

    return False


# ======================================================================
# HEADER
# ======================================================================

print("=" * 90)
print("DQ GATE 01 — STRUCTURAL")
print("Delivery Risk Intelligence Platform")
print("=" * 90)

print()
print(f"RAW      : {RAW}")
print(f"CONTRACT : {CONFIG}")
print(f"OUTPUT   : {OUT}")


# ======================================================================
# CHECK 001 — RAW EXISTE
# ======================================================================

if RAW.exists():

    pass_check(
        "STR-001",
        "__dataset__",
        "directory",
        RAW,
        "RAW directory exists"
    )

else:

    fail_check(
        "STR-001",
        "__dataset__",
        "directory",
        RAW,
        "RAW directory exists"
    )


# ======================================================================
# CHECK 002 — QUANTIDADE / NOMES DE CSV
# ======================================================================

actual_csvs = sorted(
    p.name
    for p in RAW.glob("*.csv")
) if RAW.exists() else []


expected_csvs = sorted(
    table_cfg["file"]
    for table_cfg
    in CONTRACT["tables"].values()
)


missing_csvs = sorted(
    set(expected_csvs)
    -
    set(actual_csvs)
)


extra_csvs = sorted(
    set(actual_csvs)
    -
    set(expected_csvs)
)


if not missing_csvs:

    pass_check(
        "STR-002",
        "__dataset__",
        "files_presence",
        len(actual_csvs),
        f"all {len(expected_csvs)} expected files present"
    )

else:

    fail_check(
        "STR-002",
        "__dataset__",
        "files_presence",
        missing_csvs,
        "no missing expected CSVs"
    )


if len(actual_csvs) == CONTRACT["expected_csv_count"]:

    pass_check(
        "STR-003",
        "__dataset__",
        "file_count",
        len(actual_csvs),
        CONTRACT["expected_csv_count"]
    )

else:

    fail_check(
        "STR-003",
        "__dataset__",
        "file_count",
        len(actual_csvs),
        CONTRACT["expected_csv_count"]
    )


if extra_csvs:

    warn_check(
        "STR-004",
        "__dataset__",
        "unexpected_files",
        extra_csvs,
        "no unexpected CSV files"
    )

else:

    pass_check(
        "STR-004",
        "__dataset__",
        "unexpected_files",
        0,
        0
    )


# ======================================================================
# CHECKS POR TABELA
# ======================================================================

for table_name, cfg in CONTRACT["tables"].items():

    print()
    print("-" * 90)
    print(f"[TABLE] {table_name}")
    print("-" * 90)

    path = RAW / cfg["file"]

    # ------------------------------------------------------------------
    # EXISTÊNCIA
    # ------------------------------------------------------------------

    if not path.exists():

        fail_check(
            f"{table_name}-FILE",
            table_name,
            "file_presence",
            "missing",
            cfg["file"]
        )

        print("  [FAIL] arquivo ausente")

        continue

    # ------------------------------------------------------------------
    # LEITURA
    # ------------------------------------------------------------------

    try:

        df = pd.read_csv(
            path,
            low_memory=False
        )

        loaded[table_name] = df

        pass_check(
            f"{table_name}-READ",
            table_name,
            "readability",
            "read successfully",
            "read successfully"
        )

    except Exception as exc:

        fail_check(
            f"{table_name}-READ",
            table_name,
            "readability",
            repr(exc),
            "read successfully"
        )

        print(
            f"  [FAIL] leitura: {exc}"
        )

        continue

    rows, cols = df.shape

    print(
        f"  rows={rows:,} | cols={cols}"
    )

    # ------------------------------------------------------------------
    # NÃO VAZIO
    # ------------------------------------------------------------------

    if rows > 0:

        pass_check(
            f"{table_name}-ROWS",
            table_name,
            "non_empty",
            rows,
            "> 0"
        )

    else:

        fail_check(
            f"{table_name}-ROWS",
            table_name,
            "non_empty",
            rows,
            "> 0"
        )

    # ------------------------------------------------------------------
    # SCHEMA — SET DE COLUNAS
    # ------------------------------------------------------------------

    expected_columns = cfg["columns"]

    actual_columns = list(df.columns)

    missing_columns = sorted(
        set(expected_columns)
        -
        set(actual_columns)
    )

    extra_columns = sorted(
        set(actual_columns)
        -
        set(expected_columns)
    )

    if (
        not missing_columns
        and
        not extra_columns
    ):

        pass_check(
            f"{table_name}-SCHEMA",
            table_name,
            "schema_columns",
            actual_columns,
            expected_columns
        )

    else:

        fail_check(
            f"{table_name}-SCHEMA",
            table_name,
            "schema_columns",
            {
                "missing":
                    missing_columns,

                "extra":
                    extra_columns
            },
            expected_columns
        )

    # ------------------------------------------------------------------
    # ORDEM DAS COLUNAS
    #
    # Ordem diferente não destrói semanticamente o dataset.
    # Por isso é WARNING, não critical.
    # ------------------------------------------------------------------

    if actual_columns == expected_columns:

        pass_check(
            f"{table_name}-ORDER",
            table_name,
            "column_order",
            "expected order",
            "expected order"
        )

    else:

        warn_check(
            f"{table_name}-ORDER",
            table_name,
            "column_order",
            actual_columns,
            expected_columns
        )

    # ------------------------------------------------------------------
    # FAMÍLIA DE DTYPE
    # ------------------------------------------------------------------

    for column, expected_type in cfg["types"].items():

        if column not in df.columns:

            continue

        ok = type_matches(
            df[column],
            expected_type
        )

        if ok:

            pass_check(
                f"{table_name}-TYPE-{column}",
                table_name,
                "dtype",
                str(df[column].dtype),
                expected_type
            )

        else:

            fail_check(
                f"{table_name}-TYPE-{column}",
                table_name,
                "dtype",
                str(df[column].dtype),
                expected_type
            )

    # ------------------------------------------------------------------
    # REQUIRED NOT NULL
    # Somente campos ESTRUTURAIS.
    #
    # Missing de atributos analíticos será outro gate.
    # ------------------------------------------------------------------

    for column in cfg.get(
        "required_not_null",
        []
    ):

        if column not in df.columns:

            continue

        null_count = int(
            df[column]
            .isna()
            .sum()
        )

        if null_count == 0:

            pass_check(
                f"{table_name}-NULL-{column}",
                table_name,
                "required_not_null",
                0,
                0
            )

        else:

            fail_check(
                f"{table_name}-NULL-{column}",
                table_name,
                "required_not_null",
                null_count,
                0
            )

    # ------------------------------------------------------------------
    # PK / COMPOSITE KEY
    # ------------------------------------------------------------------

    pk = cfg.get(
        "primary_key"
    )

    if pk:

        if all(
            col in df.columns
            for col in pk
        ):

            null_pk = int(
                df[pk]
                .isna()
                .any(axis=1)
                .sum()
            )

            duplicated_pk = int(
                df
                .duplicated(
                    subset=pk,
                    keep=False
                )
                .sum()
            )

            if null_pk == 0:

                pass_check(
                    f"{table_name}-PKNULL",
                    table_name,
                    "primary_key_null",
                    0,
                    0
                )

            else:

                fail_check(
                    f"{table_name}-PKNULL",
                    table_name,
                    "primary_key_null",
                    null_pk,
                    0
                )

            if duplicated_pk == 0:

                pass_check(
                    f"{table_name}-PKUNIQUE",
                    table_name,
                    "primary_key_uniqueness",
                    0,
                    0,
                    "+".join(pk)
                )

            else:

                fail_check(
                    f"{table_name}-PKUNIQUE",
                    table_name,
                    "primary_key_uniqueness",
                    duplicated_pk,
                    0,
                    "+".join(pk)
                )

    # ------------------------------------------------------------------
    # FULL ROW DUPLICATES
    # ------------------------------------------------------------------

    duplicates = int(
        df
        .duplicated()
        .sum()
    )

    max_dup = cfg.get(
        "max_full_row_duplicates"
    )

    if max_dup is None:

        # geolocation, por exemplo:
        # duplicação será tratada posteriormente,
        # mas não é violação de PK porque não há PK declarada.

        warn_check(
            f"{table_name}-ROWDUP",
            table_name,
            "full_row_duplicates",
            duplicates,
            "informational / allowed at structural gate"
        )

    elif duplicates <= max_dup:

        pass_check(
            f"{table_name}-ROWDUP",
            table_name,
            "full_row_duplicates",
            duplicates,
            f"<= {max_dup}"
        )

    else:

        fail_check(
            f"{table_name}-ROWDUP",
            table_name,
            "full_row_duplicates",
            duplicates,
            f"<= {max_dup}"
        )


# ======================================================================
# INTEGRIDADE REFERENCIAL
# ======================================================================

print()
print("=" * 90)
print("REFERENTIAL INTEGRITY")
print("=" * 90)


relationship_rows = []


for i, rel in enumerate(
    CONTRACT["relationships"],
    start=1
):

    child_name = rel[
        "child_table"
    ]

    parent_name = rel[
        "parent_table"
    ]

    child_col = rel[
        "child_column"
    ]

    parent_col = rel[
        "parent_column"
    ]

    check_id = (
        f"FK-{i:02d}-"
        f"{child_name}-{child_col}"
    )

    if (
        child_name not in loaded
        or
        parent_name not in loaded
    ):

        fail_check(
            check_id,
            child_name,
            "referential_integrity",
            "table unavailable",
            "0 orphan values",
            (
                f"{child_name}.{child_col} -> "
                f"{parent_name}.{parent_col}"
            )
        )

        continue

    child = loaded[
        child_name
    ]

    parent = loaded[
        parent_name
    ]

    if (
        child_col not in child.columns
        or
        parent_col not in parent.columns
    ):

        fail_check(
            check_id,
            child_name,
            "referential_integrity",
            "column unavailable",
            "0 orphan values"
        )

        continue

    parent_values = set(
        parent[
            parent_col
        ]
        .dropna()
        .astype(str)
    )

    child_values = (
        child[
            child_col
        ]
        .dropna()
        .astype(str)
    )

    orphan_mask = (
        ~child_values.isin(
            parent_values
        )
    )

    orphan_rows = int(
        orphan_mask.sum()
    )

    orphan_values = int(
        child_values[
            orphan_mask
        ]
        .nunique()
    )

    relationship_rows.append(
        {
            "child_table":
                child_name,

            "child_column":
                child_col,

            "parent_table":
                parent_name,

            "parent_column":
                parent_col,

            "child_non_null_rows":
                len(child_values),

            "orphan_rows":
                orphan_rows,

            "orphan_unique_values":
                orphan_values,

            "orphan_pct":
                (
                    100
                    *
                    orphan_rows
                    /
                    len(child_values)
                )
                if len(child_values)
                else np.nan,
        }
    )

    if orphan_rows == 0:

        pass_check(
            check_id,
            child_name,
            "referential_integrity",
            0,
            0,
            (
                f"{child_name}.{child_col} -> "
                f"{parent_name}.{parent_col}"
            )
        )

        print(
            f"[PASS] "
            f"{child_name}.{child_col} -> "
            f"{parent_name}.{parent_col}"
        )

    else:

        fail_check(
            check_id,
            child_name,
            "referential_integrity",
            orphan_rows,
            0,
            (
                f"{orphan_values} unique orphan keys | "
                f"{child_name}.{child_col} -> "
                f"{parent_name}.{parent_col}"
            )
        )

        print(
            f"[FAIL] "
            f"{child_name}.{child_col} -> "
            f"{parent_name}.{parent_col} | "
            f"orphan_rows={orphan_rows:,}"
        )


# ======================================================================
# SCORECARD
# ======================================================================

scorecard = pd.DataFrame(
    results
)


scorecard.to_csv(
    OUT / "dq_gate_01_scorecard.csv",
    index=False
)


exceptions = scorecard[
    scorecard[
        "status"
    ].isin(
        [
            "FAIL",
            "WARN"
        ]
    )
].copy()


exceptions.to_csv(
    OUT / "dq_gate_01_exceptions.csv",
    index=False
)


pd.DataFrame(
    relationship_rows
).to_csv(
    OUT / "dq_gate_01_relationships.csv",
    index=False
)


# ======================================================================
# RESUMO
# ======================================================================

critical_failures = int(
    (
        (scorecard["severity"] == "CRITICAL")
        &
        (scorecard["status"] == "FAIL")
    ).sum()
)


warnings = int(
    (
        scorecard["status"]
        ==
        "WARN"
    ).sum()
)


passes = int(
    (
        scorecard["status"]
        ==
        "PASS"
    ).sum()
)


total_checks = len(
    scorecard
)


if critical_failures == 0:

    gate_status = "PASS"

else:

    gate_status = "FAIL"


summary = {
    "gate":
        "DQ_GATE_01_STRUCTURAL",

    "status":
        gate_status,

    "total_checks":
        total_checks,

    "passes":
        passes,

    "warnings":
        warnings,

    "critical_failures":
        critical_failures,

    "raw_directory":
        str(RAW),

    "contract":
        str(CONFIG),
}


with (
    OUT
    /
    "dq_gate_01_summary.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ======================================================================
# RELATÓRIO TXT
# ======================================================================

report = []

report.append(
    "=" * 90
)

report.append(
    "DQ GATE 01 — STRUCTURAL"
)

report.append(
    "=" * 90
)

report.append("")

report.append(
    f"STATUS            : {gate_status}"
)

report.append(
    f"TOTAL CHECKS      : {total_checks}"
)

report.append(
    f"PASS              : {passes}"
)

report.append(
    f"WARNINGS          : {warnings}"
)

report.append(
    f"CRITICAL FAILURES : {critical_failures}"
)

report.append("")

report.append(
    "CHECKS NÃO-PASS:"
)

report.append(
    "-" * 90
)


if exceptions.empty:

    report.append(
        "Nenhum."
    )

else:

    for _, row in exceptions.iterrows():

        report.append(
            f"[{row['status']}] "
            f"{row['check_id']} | "
            f"{row['table']} | "
            f"{row['dimension']} | "
            f"observado={row['observed']} | "
            f"esperado={row['expected']}"
        )


report.append("")

report.append(
    "-" * 90
)

report.append(
    "REGRA DO GATE:"
)

report.append(
    "FAIL crítico -> NÃO construir Silver."
)

report.append(
    "PASS com warnings -> pipeline pode avançar, "
    "mas warnings devem permanecer documentados."
)

report.append(
    "-" * 90
)


report_path = (
    OUT
    /
    "DQ_GATE_01_STRUCTURAL_REPORT.txt"
)


report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ======================================================================
# TERMINAL
# ======================================================================

print()
print("=" * 90)
print("DQ GATE 01 — RESULTADO")
print("=" * 90)

print(
    f"STATUS            : {gate_status}"
)

print(
    f"TOTAL CHECKS      : {total_checks}"
)

print(
    f"PASS              : {passes}"
)

print(
    f"WARNINGS          : {warnings}"
)

print(
    f"CRITICAL FAILURES : {critical_failures}"
)

print()

if not exceptions.empty:

    print(
        "CHECKS NÃO-PASS:"
    )

    print(
        exceptions[
            [
                "status",
                "check_id",
                "table",
                "dimension",
                "observed",
            ]
        ].to_string(
            index=False
        )
    )


print()

print(
    "Arquivos gerados:"
)

for path in sorted(
    OUT.glob("*")
):

    print(
        f"  - {path}"
    )


print()

if gate_status == "PASS":

    print(
        "[PASS] DQ GATE 01 — STRUCTURAL APROVADO."
    )

    print(
        "A estrutura dos dados está apta a seguir "
        "para o próximo gate."
    )

    sys.exit(0)

else:

    print(
        "[FAIL] DQ GATE 01 — STRUCTURAL REPROVADO."
    )

    print(
        "Construção da Silver BLOQUEADA "
        "até resolver as falhas críticas."
    )

    sys.exit(2)
