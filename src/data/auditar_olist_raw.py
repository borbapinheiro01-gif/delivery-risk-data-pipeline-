#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORIA 01 — OLIST RAW DATA
Delivery Risk Intelligence Platform

Objetivo
--------
Auditar os 9 CSVs originais da Olist SEM alterar os dados.

Verificações:
1. Arquivos e dimensões
2. Colunas e tipos
3. Missing values
4. Duplicatas
5. Cardinalidades
6. Chaves
7. Integridade referencial
8. Período temporal
9. Status dos pedidos
10. Target preliminar de atraso
11. Consistência temporal
12. Cardinalidade das relações

Saídas:
reports/audit/01_raw_tables_summary.csv
reports/audit/02_missing_values.csv
reports/audit/03_key_audit.csv
reports/audit/04_relationship_audit.csv
reports/audit/05_order_status.csv
reports/audit/06_delivery_target_summary.csv
reports/audit/07_temporal_consistency.csv
reports/audit/olist_raw_audit.txt
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"
REPORT = PROJECT / "reports" / "audit"

REPORT.mkdir(parents=True, exist_ok=True)

FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

EXPECTED_PRIMARY_KEYS = {
    "orders": ["order_id"],
    "customers": ["customer_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "category_translation": ["product_category_name"],
}

EXPECTED_COMPOSITE_KEYS = {
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
}

DATE_COLUMNS_ORDERS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


# ============================================================
# UTILITÁRIOS
# ============================================================

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def section(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


def pct(n, d):
    if d == 0:
        return np.nan
    return 100.0 * n / d


# ============================================================
# 1. CONFERÊNCIA DOS ARQUIVOS
# ============================================================

section("AUDITORIA 01 — OLIST RAW DATA")

log(f"Projeto : {PROJECT}")
log(f"RAW     : {RAW}")
log(f"Relatório: {REPORT}")

if not RAW.exists():
    log(f"[ERRO] Diretório RAW não existe: {RAW}")
    sys.exit(1)

missing_files = []

for name, filename in FILES.items():
    path = RAW / filename
    if not path.exists():
        missing_files.append(filename)

if missing_files:
    log()
    log("[ERRO] Arquivos ausentes:")
    for f in missing_files:
        log(f"  - {f}")
    sys.exit(1)

log()
log("[OK] Todos os 9 arquivos foram encontrados.")


# ============================================================
# 2. CARREGAMENTO
# ============================================================

section("2. CARREGAMENTO DOS CSVs")

dfs = {}

for name, filename in FILES.items():

    path = RAW / filename

    log(f"Lendo {filename} ...")

    df = pd.read_csv(path, low_memory=False)

    dfs[name] = df

    log(
        f"  [OK] {name:<22} "
        f"{len(df):>10,} linhas x {df.shape[1]:>3} colunas"
    )


# ============================================================
# 3. RESUMO DAS TABELAS
# ============================================================

section("3. DIMENSÕES E DUPLICATAS")

table_summary = []

for name, df in dfs.items():

    duplicated_rows = int(df.duplicated().sum())

    row = {
        "table": name,
        "rows": len(df),
        "columns": df.shape[1],
        "duplicated_rows": duplicated_rows,
        "duplicated_rows_pct": pct(duplicated_rows, len(df)),
        "memory_mb": df.memory_usage(deep=True).sum() / (1024**2),
    }

    table_summary.append(row)

    log(
        f"{name:<22} "
        f"rows={len(df):>9,} | "
        f"cols={df.shape[1]:>2} | "
        f"duplicadas={duplicated_rows:>7,}"
    )

table_summary_df = pd.DataFrame(table_summary)

table_summary_df.to_csv(
    REPORT / "01_raw_tables_summary.csv",
    index=False
)


# ============================================================
# 4. SCHEMA
# ============================================================

section("4. SCHEMA — COLUNAS E TIPOS")

for name, df in dfs.items():

    log()
    log(f"[{name}]")

    for col in df.columns:
        log(
            f"  {col:<40} "
            f"dtype={str(df[col].dtype):<10} "
            f"unique={df[col].nunique(dropna=True):>8,}"
        )


# ============================================================
# 5. MISSING VALUES
# ============================================================

section("5. MISSING VALUES")

missing_records = []

for name, df in dfs.items():

    for col in df.columns:

        n_missing = int(df[col].isna().sum())

        missing_records.append({
            "table": name,
            "column": col,
            "missing": n_missing,
            "missing_pct": pct(n_missing, len(df)),
        })

missing_df = pd.DataFrame(missing_records)

missing_df.to_csv(
    REPORT / "02_missing_values.csv",
    index=False
)

important_missing = (
    missing_df[missing_df["missing"] > 0]
    .sort_values(["missing_pct"], ascending=False)
)

if important_missing.empty:
    log("Nenhum missing encontrado.")
else:
    for _, r in important_missing.iterrows():
        log(
            f"{r['table']:<22} "
            f"{r['column']:<40} "
            f"{int(r['missing']):>9,} "
            f"({r['missing_pct']:.2f}%)"
        )


# ============================================================
# 6. AUDITORIA DE CHAVES
# ============================================================

section("6. CHAVES PRIMÁRIAS / COMPOSTAS")

key_results = []

for table, cols in EXPECTED_PRIMARY_KEYS.items():

    df = dfs[table]

    missing_key = int(df[cols].isna().any(axis=1).sum())

    duplicated_key = int(
        df.duplicated(subset=cols, keep=False).sum()
    )

    unique_ok = duplicated_key == 0
    null_ok = missing_key == 0

    result = {
        "table": table,
        "key": "+".join(cols),
        "type": "candidate_primary_key",
        "rows_with_null_key": missing_key,
        "rows_with_duplicated_key": duplicated_key,
        "unique": unique_ok,
        "no_nulls": null_ok,
    }

    key_results.append(result)

    status = "PASS" if unique_ok and null_ok else "CHECK"

    log(
        f"[{status}] {table:<22} "
        f"key={'+'.join(cols):<30} "
        f"null={missing_key:<6} "
        f"duplicated={duplicated_key}"
    )


for table, cols in EXPECTED_COMPOSITE_KEYS.items():

    df = dfs[table]

    missing_key = int(df[cols].isna().any(axis=1).sum())

    duplicated_key = int(
        df.duplicated(subset=cols, keep=False).sum()
    )

    result = {
        "table": table,
        "key": "+".join(cols),
        "type": "candidate_composite_key",
        "rows_with_null_key": missing_key,
        "rows_with_duplicated_key": duplicated_key,
        "unique": duplicated_key == 0,
        "no_nulls": missing_key == 0,
    }

    key_results.append(result)

    status = (
        "PASS"
        if duplicated_key == 0 and missing_key == 0
        else "CHECK"
    )

    log(
        f"[{status}] {table:<22} "
        f"key={'+'.join(cols):<30} "
        f"null={missing_key:<6} "
        f"duplicated={duplicated_key}"
    )


pd.DataFrame(key_results).to_csv(
    REPORT / "03_key_audit.csv",
    index=False
)


# ============================================================
# 7. INTEGRIDADE REFERENCIAL
# ============================================================

section("7. INTEGRIDADE REFERENCIAL")

relationships = [
    (
        "orders",
        "customer_id",
        "customers",
        "customer_id"
    ),
    (
        "order_items",
        "order_id",
        "orders",
        "order_id"
    ),
    (
        "order_items",
        "product_id",
        "products",
        "product_id"
    ),
    (
        "order_items",
        "seller_id",
        "sellers",
        "seller_id"
    ),
    (
        "payments",
        "order_id",
        "orders",
        "order_id"
    ),
    (
        "reviews",
        "order_id",
        "orders",
        "order_id"
    ),
]

relationship_results = []

for child_table, child_key, parent_table, parent_key in relationships:

    child = dfs[child_table]
    parent = dfs[parent_table]

    parent_values = set(
        parent[parent_key].dropna().astype(str)
    )

    child_values = child[child_key].dropna().astype(str)

    orphan_mask = ~child_values.isin(parent_values)

    orphan_rows = int(orphan_mask.sum())

    relationship_results.append({
        "child_table": child_table,
        "child_key": child_key,
        "parent_table": parent_table,
        "parent_key": parent_key,
        "child_rows": len(child),
        "orphan_rows": orphan_rows,
        "orphan_pct": pct(orphan_rows, len(child)),
    })

    status = "PASS" if orphan_rows == 0 else "CHECK"

    log(
        f"[{status}] "
        f"{child_table}.{child_key} -> "
        f"{parent_table}.{parent_key} | "
        f"órfãos={orphan_rows:,}"
    )


pd.DataFrame(relationship_results).to_csv(
    REPORT / "04_relationship_audit.csv",
    index=False
)


# ============================================================
# 8. ORDERS — DATAS
# ============================================================

section("8. ORDERS — PERÍODO TEMPORAL")

orders = dfs["orders"].copy()

for col in DATE_COLUMNS_ORDERS:
    if col in orders.columns:
        orders[col] = pd.to_datetime(
            orders[col],
            errors="coerce"
        )

purchase = orders["order_purchase_timestamp"]

log(
    f"Primeira compra : {purchase.min()}"
)

log(
    f"Última compra   : {purchase.max()}"
)

log(
    f"Pedidos         : {len(orders):,}"
)


# ============================================================
# 9. STATUS
# ============================================================

section("9. DISTRIBUIÇÃO DOS STATUS")

status_df = (
    orders["order_status"]
    .value_counts(dropna=False)
    .rename_axis("order_status")
    .reset_index(name="count")
)

status_df["pct"] = (
    status_df["count"] / len(orders) * 100
)

status_df.to_csv(
    REPORT / "05_order_status.csv",
    index=False
)

for _, r in status_df.iterrows():

    log(
        f"{str(r['order_status']):<20} "
        f"{int(r['count']):>8,} "
        f"{r['pct']:>7.3f}%"
    )


# ============================================================
# 10. TARGET DE ATRASO
# ============================================================

section("10. TARGET PRELIMINAR — ATRASO")

eligible = orders[
    orders["order_delivered_customer_date"].notna()
    &
    orders["order_estimated_delivery_date"].notna()
].copy()

eligible["delivery_delta_days"] = (
    eligible["order_delivered_customer_date"]
    -
    eligible["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

eligible["late"] = (
    eligible["order_delivered_customer_date"]
    >
    eligible["order_estimated_delivery_date"]
).astype(int)

n_eligible = len(eligible)
n_late = int(eligible["late"].sum())
n_ontime = n_eligible - n_late

late_rate = pct(n_late, n_eligible)

log(f"Pedidos totais               : {len(orders):,}")
log(f"Pedidos com target observável : {n_eligible:,}")
log(f"Não atrasados                 : {n_ontime:,}")
log(f"Atrasados                     : {n_late:,}")
log(f"Taxa de atraso                : {late_rate:.4f}%")

delivery_summary = pd.DataFrame([{
    "orders_total": len(orders),
    "eligible_target": n_eligible,
    "on_time": n_ontime,
    "late": n_late,
    "late_rate_pct": late_rate,
    "delivery_delta_mean_days":
        eligible["delivery_delta_days"].mean(),
    "delivery_delta_median_days":
        eligible["delivery_delta_days"].median(),
    "delivery_delta_p90_days":
        eligible["delivery_delta_days"].quantile(0.90),
    "delivery_delta_p95_days":
        eligible["delivery_delta_days"].quantile(0.95),
    "delivery_delta_max_days":
        eligible["delivery_delta_days"].max(),
}])

delivery_summary.to_csv(
    REPORT / "06_delivery_target_summary.csv",
    index=False
)


# ============================================================
# 11. CONSISTÊNCIA TEMPORAL
# ============================================================

section("11. CONSISTÊNCIA TEMPORAL")

temporal_checks = []

checks = [
    (
        "purchase_after_approval",
        "order_purchase_timestamp",
        "order_approved_at",
        lambda a, b: a > b
    ),
    (
        "approval_after_carrier",
        "order_approved_at",
        "order_delivered_carrier_date",
        lambda a, b: a > b
    ),
    (
        "carrier_after_customer_delivery",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        lambda a, b: a > b
    ),
]

for name, c1, c2, condition in checks:

    valid = orders[c1].notna() & orders[c2].notna()

    violations = int(
        condition(
            orders.loc[valid, c1],
            orders.loc[valid, c2]
        ).sum()
    )

    temporal_checks.append({
        "check": name,
        "comparable_rows": int(valid.sum()),
        "violations": violations,
        "violation_pct": pct(
            violations,
            int(valid.sum())
        ),
    })

    status = "PASS" if violations == 0 else "CHECK"

    log(
        f"[{status}] {name:<35} "
        f"comparáveis={int(valid.sum()):>8,} "
        f"violações={violations:>6,}"
    )


pd.DataFrame(temporal_checks).to_csv(
    REPORT / "07_temporal_consistency.csv",
    index=False
)


# ============================================================
# 12. CARDINALIDADE ORDER_ID
# ============================================================

section("12. CARDINALIDADE DAS TABELAS TRANSACIONAIS")

for name in [
    "orders",
    "order_items",
    "payments",
    "reviews",
]:

    df = dfs[name]

    if "order_id" not in df.columns:
        continue

    unique_orders = df["order_id"].nunique()

    rows_per_order = len(df) / unique_orders

    max_rows = (
        df.groupby("order_id")
        .size()
        .max()
    )

    log(
        f"{name:<18} "
        f"rows={len(df):>9,} | "
        f"orders={unique_orders:>8,} | "
        f"rows/order={rows_per_order:>7.3f} | "
        f"máximo/order={max_rows}"
    )


# ============================================================
# 13. SELLERS POR PEDIDO
# ============================================================

section("13. SELLERS E ITENS POR PEDIDO")

items = dfs["order_items"]

items_per_order = (
    items.groupby("order_id")
    .size()
)

sellers_per_order = (
    items.groupby("order_id")["seller_id"]
    .nunique()
)

products_per_order = (
    items.groupby("order_id")["product_id"]
    .nunique()
)

multi_item = int((items_per_order > 1).sum())
multi_seller = int((sellers_per_order > 1).sum())
multi_product = int((products_per_order > 1).sum())

log(
    f"Pedidos em order_items       : "
    f"{items_per_order.size:,}"
)

log(
    f"Pedidos com >1 item          : "
    f"{multi_item:,} "
    f"({pct(multi_item, items_per_order.size):.3f}%)"
)

log(
    f"Pedidos com >1 seller        : "
    f"{multi_seller:,} "
    f"({pct(multi_seller, sellers_per_order.size):.3f}%)"
)

log(
    f"Pedidos com >1 produto       : "
    f"{multi_product:,} "
    f"({pct(multi_product, products_per_order.size):.3f}%)"
)


# ============================================================
# 14. GEOLOCALIZAÇÃO
# ============================================================

section("14. GEOLOCALIZAÇÃO")

geo = dfs["geolocation"]

zip_col = "geolocation_zip_code_prefix"

if zip_col in geo.columns:

    duplicated_zip = int(
        geo.duplicated(subset=[zip_col]).sum()
    )

    unique_zip = geo[zip_col].nunique()

    log(
        f"Linhas geolocation          : {len(geo):,}"
    )

    log(
        f"CEPs-prefixo únicos         : {unique_zip:,}"
    )

    log(
        f"Linhas além da 1ª por CEP   : {duplicated_zip:,}"
    )

    log()
    log(
        "[INFO] geolocation NÃO deve ser ligada diretamente "
        "a customers/sellers antes de agregarmos o CEP."
    )


# ============================================================
# 15. RESULTADO
# ============================================================

section("15. RESULTADO DA AUDITORIA")

log("[OK] RAW não foi alterado.")

log()
log("Relatórios gerados:")

for path in sorted(REPORT.glob("*")):
    log(f"  - {path.relative_to(PROJECT)}")


# Texto integral da auditoria
txt_path = REPORT / "olist_raw_audit.txt"

txt_path.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

log()
log("==============================================================")
log(" AUDITORIA 01 CONCLUÍDA")
log("==============================================================")
log()
log(f"Relatório principal:")
log(txt_path)
