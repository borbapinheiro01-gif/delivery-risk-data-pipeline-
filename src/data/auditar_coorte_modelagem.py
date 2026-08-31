#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
RAW = PROJECT / "data" / "raw" / "olist"
OUT = PROJECT / "reports" / "audit"

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CARREGAMENTO
# ============================================================

orders = pd.read_csv(
    RAW / "olist_orders_dataset.csv",
    low_memory=False
)

items = pd.read_csv(
    RAW / "olist_order_items_dataset.csv",
    low_memory=False
)

payments = pd.read_csv(
    RAW / "olist_order_payments_dataset.csv",
    low_memory=False
)

customers = pd.read_csv(
    RAW / "olist_customers_dataset.csv",
    low_memory=False
)

products = pd.read_csv(
    RAW / "olist_products_dataset.csv",
    low_memory=False
)

sellers = pd.read_csv(
    RAW / "olist_sellers_dataset.csv",
    low_memory=False
)

translation = pd.read_csv(
    RAW / "product_category_name_translation.csv",
    low_memory=False
)

date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")


def pct(a, b):
    return 100 * a / b if b else np.nan


print("=" * 78)
print("AUDITORIA 02 — COORTE TEMPORAL DE MODELAGEM")
print("=" * 78)

# ============================================================
# 1. TARGET
# ============================================================

orders["target_observable"] = (
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
)

orders["late"] = np.where(
    orders["target_observable"],
    (
        orders["order_delivered_customer_date"]
        >
        orders["order_estimated_delivery_date"]
    ).astype(int),
    np.nan,
)

orders["purchase_month"] = (
    orders["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

# ============================================================
# 2. LIMITES TEMPORAIS
# ============================================================

print()
print("[1] LIMITES TEMPORAIS")
print("-" * 78)

for col in date_cols:
    print(
        f"{col:<40}"
        f" min={orders[col].min()} |"
        f" max={orders[col].max()}"
    )

# ============================================================
# 3. COBERTURA DO TARGET POR MÊS
# ============================================================

monthly = (
    orders
    .groupby("purchase_month", dropna=False)
    .agg(
        orders_total=("order_id", "size"),
        delivered_status=(
            "order_status",
            lambda x: (x == "delivered").sum()
        ),
        target_observable=(
            "target_observable",
            "sum"
        ),
        approvals_present=(
            "order_approved_at",
            lambda x: x.notna().sum()
        ),
    )
    .reset_index()
)

late_month = (
    orders[orders["target_observable"]]
    .groupby("purchase_month")
    .agg(
        late_orders=("late", "sum"),
        late_rate_pct=("late", lambda x: 100 * x.mean())
    )
    .reset_index()
)

monthly = monthly.merge(
    late_month,
    on="purchase_month",
    how="left"
)

monthly["target_coverage_pct"] = (
    100
    * monthly["target_observable"]
    / monthly["orders_total"]
)

monthly["delivered_pct"] = (
    100
    * monthly["delivered_status"]
    / monthly["orders_total"]
)

monthly.to_csv(
    OUT / "08_monthly_target_coverage.csv",
    index=False
)

print()
print("[2] COBERTURA E ATRASO POR MÊS")
print("-" * 78)

print(
    monthly.to_string(
        index=False,
        formatters={
            "target_coverage_pct": "{:.2f}".format,
            "delivered_pct": "{:.2f}".format,
            "late_rate_pct": lambda x:
                "" if pd.isna(x) else f"{x:.2f}",
        }
    )
)

# ============================================================
# 4. COMPARAÇÃO DOS DOIS INSTANTES DE SCORING
# ============================================================

cohort = orders[orders["target_observable"]].copy()

approval_missing = cohort["order_approved_at"].isna()

approval_after_carrier = (
    cohort["order_approved_at"].notna()
    & cohort["order_delivered_carrier_date"].notna()
    & (
        cohort["order_approved_at"]
        >
        cohort["order_delivered_carrier_date"]
    )
)

purchase_missing = cohort[
    "order_purchase_timestamp"
].isna()

print()
print("[3] CANDIDATOS A t0")
print("-" * 78)

print(f"Coorte com target       : {len(cohort):,}")

print()
print("t0 = PURCHASE")
print(
    f"  purchase ausente      : "
    f"{purchase_missing.sum():,}"
)

print()
print("t0 = APPROVAL")
print(
    f"  approval ausente      : "
    f"{approval_missing.sum():,}"
)
print(
    f"  approval > carrier    : "
    f"{approval_after_carrier.sum():,}"
)

# ============================================================
# 5. CONSISTÊNCIA COMPLETA DA COORTE
# ============================================================

carrier_after_customer = (
    cohort["order_delivered_carrier_date"].notna()
    & (
        cohort["order_delivered_carrier_date"]
        >
        cohort["order_delivered_customer_date"]
    )
)

print()
print("[4] ANOMALIAS TEMPORAIS NA COORTE")
print("-" * 78)

print(
    f"approval > carrier      : "
    f"{approval_after_carrier.sum():,}"
)

print(
    f"carrier > customer      : "
    f"{carrier_after_customer.sum():,}"
)

# ============================================================
# 6. COBERTURA DOS JOINS NA COORTE
# ============================================================

cohort_ids = set(cohort["order_id"])

item_ids = set(items["order_id"])
payment_ids = set(payments["order_id"])
customer_ids = set(customers["customer_id"])

print()
print("[5] COBERTURA DAS FONTES NA COORTE")
print("-" * 78)

missing_items = cohort_ids - item_ids
missing_payments = cohort_ids - payment_ids

print(
    f"Pedidos sem order_items : {len(missing_items):,}"
)

print(
    f"Pedidos sem payments    : {len(missing_payments):,}"
)

missing_customers = (
    set(cohort["customer_id"])
    - customer_ids
)

print(
    f"Pedidos sem customer    : {len(missing_customers):,}"
)

# ============================================================
# 7. PRODUTOS / SELLERS DA COORTE
# ============================================================

cohort_items = items[
    items["order_id"].isin(cohort_ids)
].copy()

missing_product = (
    ~cohort_items["product_id"]
    .isin(set(products["product_id"]))
).sum()

missing_seller = (
    ~cohort_items["seller_id"]
    .isin(set(sellers["seller_id"]))
).sum()

print()
print("[6] ITEM-LEVEL COVERAGE")
print("-" * 78)

print(
    f"Itens da coorte         : "
    f"{len(cohort_items):,}"
)

print(
    f"product_id sem cadastro : "
    f"{missing_product:,}"
)

print(
    f"seller_id sem cadastro  : "
    f"{missing_seller:,}"
)

# ============================================================
# 8. CATEGORIAS SEM TRADUÇÃO
# ============================================================

categories = (
    products["product_category_name"]
    .dropna()
    .drop_duplicates()
)

translated = set(
    translation[
        "product_category_name"
    ].dropna()
)

untranslated = sorted(
    set(categories) - translated
)

print()
print("[7] CATEGORIAS SEM TRADUÇÃO")
print("-" * 78)

print(f"Quantidade: {len(untranslated)}")

for cat in untranslated:
    print(f"  - {cat}")

pd.DataFrame({
    "product_category_name": untranslated
}).to_csv(
    OUT / "09_untranslated_categories.csv",
    index=False
)

# ============================================================
# 9. RECOMENDAÇÃO AUTOMÁTICA DE CUTOFF
# ============================================================

print()
print("[8] DIAGNÓSTICO DE MATURIDADE DO LABEL")
print("-" * 78)

recent = monthly.tail(6).copy()

print(
    recent[
        [
            "purchase_month",
            "orders_total",
            "target_observable",
            "target_coverage_pct",
            "late_rate_pct",
        ]
    ].to_string(
        index=False,
        formatters={
            "target_coverage_pct": "{:.2f}".format,
            "late_rate_pct": lambda x:
                "" if pd.isna(x) else f"{x:.2f}",
        }
    )
)

print()
print("=" * 78)
print("[OK] AUDITORIA 02 CONCLUÍDA")
print("=" * 78)

print()
print("Arquivo principal:")
print(OUT / "08_monthly_target_coverage.csv")
