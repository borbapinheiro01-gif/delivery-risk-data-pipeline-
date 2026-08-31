#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DATASET KNOWLEDGE & TRUTH REGISTRY
Delivery Risk Intelligence Platform
===============================================================================

OBJETIVO
--------
Construir uma representação pesquisável e reutilizável de todo o conhecimento
já disponível sobre a base Olist.

ENTRADAS
--------
data/raw/olist/*.csv

reports/data_quality/gate_01_structural/
reports/data_quality/gate_02_semantic/
reports/data_quality/gate_03_statistical/

SAÍDAS
------
metadata/

    00_registry_build_manifest.json
    dataset_manifest.json
    table_catalog.csv
    column_catalog.csv
    relationship_registry.csv
    truth_provenance_registry.csv
    temporal_availability_registry.csv
    value_domain_registry.csv
    quality_observation_registry.csv
    unresolved_questions.csv
    search_knowledge.jsonl
    DATA_CARD_OLIST.md
    DATASET_KNOWLEDGE_SUMMARY.txt

PRINCÍPIOS
----------
1. RAW nunca é alterado.
2. Informação observada é separada de interpretação.
3. Informação documentada é separada de inferência.
4. Disponibilidade temporal é separada da existência do dado no CSV.
5. Missing de entidade é separado de missing de atributo.
6. Feature permitida é separada de fonte de target.
7. Verdade interna não é confundida com ground truth externo.
8. Decisões ainda não comprovadas permanecem explicitamente OPEN.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"

META = PROJECT / "metadata"

META.mkdir(
    parents=True,
    exist_ok=True
)


GATE_PATHS = {
    "gate_01_structural":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_01_structural"
        / "dq_gate_01_scorecard.csv",

    "gate_02_semantic":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_02_semantic"
        / "dq_gate_02_scorecard.csv",

    "gate_03_statistical":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_03_statistical"
        / "dq_gate_03_scorecard.csv",
}


GATE_SUMMARY_PATHS = {
    "gate_01_structural":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_01_structural"
        / "dq_gate_01_summary.json",

    "gate_02_semantic":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_02_semantic"
        / "dq_gate_02_summary.json",

    "gate_03_statistical":
        PROJECT
        / "reports"
        / "data_quality"
        / "gate_03_statistical"
        / "dq_gate_03_summary.json",
}


PENDING_AUDIT_03 = (
    PROJECT
    / "reports"
    / "audit"
    / "03_data_diagnosis"
    / "27_pending_data_decisions.json"
)


# =============================================================================
# DATASET
# =============================================================================

DATASET_ID = "olistbr/brazilian-ecommerce"

DATASET_NAME = "Brazilian E-Commerce Public Dataset by Olist"

PROJECT_NAME = "Delivery Risk Intelligence Platform"

PREDICTION_T0 = "order_purchase_timestamp"


FILES = {
    "customers":
        "olist_customers_dataset.csv",

    "geolocation":
        "olist_geolocation_dataset.csv",

    "order_items":
        "olist_order_items_dataset.csv",

    "payments":
        "olist_order_payments_dataset.csv",

    "reviews":
        "olist_order_reviews_dataset.csv",

    "orders":
        "olist_orders_dataset.csv",

    "products":
        "olist_products_dataset.csv",

    "sellers":
        "olist_sellers_dataset.csv",

    "translation":
        "product_category_name_translation.csv",
}


# =============================================================================
# DESCRIÇÃO DAS TABELAS
# =============================================================================

TABLE_META = {

    "customers": {
        "entity": "customer_order_identity",
        "grain": "1 row per order-specific customer_id",
        "description_pt":
            "Identificação do cliente associada ao pedido e localização "
            "de destino.",
        "lifecycle_role": "master/reference + order destination"
    },

    "geolocation": {
        "entity": "zip_geolocation_observation",
        "grain": "1 row per geolocation observation for ZIP prefix",
        "description_pt":
            "Observações geográficas associadas a prefixos de CEP brasileiros.",
        "lifecycle_role": "reference"
    },

    "order_items": {
        "entity": "order_item",
        "grain": "1 row per item position inside an order",
        "description_pt":
            "Itens de pedidos, produtos, sellers, preço, frete e limite "
            "de postagem.",
        "lifecycle_role": "transaction"
    },

    "payments": {
        "entity": "order_payment",
        "grain": "1 row per payment sequence",
        "description_pt":
            "Registros de pagamentos associados aos pedidos.",
        "lifecycle_role": "transaction"
    },

    "reviews": {
        "entity": "order_review",
        "grain": "1 row per review record",
        "description_pt":
            "Avaliações e comentários fornecidos após a experiência de compra.",
        "lifecycle_role": "post_delivery / post_purchase"
    },

    "orders": {
        "entity": "order",
        "grain": "1 row per order",
        "description_pt":
            "Tabela central do ciclo de vida do pedido.",
        "lifecycle_role": "transaction / central fact"
    },

    "products": {
        "entity": "product",
        "grain": "1 row per product_id",
        "description_pt":
            "Metadados, categoria e características físicas do produto.",
        "lifecycle_role": "master/reference"
    },

    "sellers": {
        "entity": "seller",
        "grain": "1 row per seller_id",
        "description_pt":
            "Cadastro e localização dos vendedores.",
        "lifecycle_role": "master/reference"
    },

    "translation": {
        "entity": "product_category_translation",
        "grain": "1 row per Portuguese product category",
        "description_pt":
            "Tradução das categorias de produto do português para inglês.",
        "lifecycle_role": "reference / presentation"
    },
}


# =============================================================================
# CHAVES
# =============================================================================

PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "geolocation": [],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "reviews": [],
    "orders": ["order_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "translation": ["product_category_name"],
}


FOREIGN_KEYS = {

    ("orders", "customer_id"):
        "customers.customer_id",

    ("order_items", "order_id"):
        "orders.order_id",

    ("order_items", "product_id"):
        "products.product_id",

    ("order_items", "seller_id"):
        "sellers.seller_id",

    ("payments", "order_id"):
        "orders.order_id",

    ("reviews", "order_id"):
        "orders.order_id",

    ("products", "product_category_name"):
        "translation.product_category_name",
}


# =============================================================================
# RELACIONAMENTOS CONHECIDOS
# =============================================================================

RELATIONSHIPS = [

    {
        "parent_table": "customers",
        "parent_column": "customer_id",
        "child_table": "orders",
        "child_column": "customer_id",
        "expected_relationship": "1:1 order-specific customer identity",
        "join_type_recommended": "many_to_one from orders perspective",
        "purpose": "order destination/customer metadata"
    },

    {
        "parent_table": "orders",
        "parent_column": "order_id",
        "child_table": "order_items",
        "child_column": "order_id",
        "expected_relationship": "1:N",
        "join_type_recommended": "aggregate child before order-level ML join",
        "purpose": "items/products/sellers/freight"
    },

    {
        "parent_table": "orders",
        "parent_column": "order_id",
        "child_table": "payments",
        "child_column": "order_id",
        "expected_relationship": "1:N",
        "join_type_recommended": "aggregate child before order-level ML join",
        "purpose": "payments"
    },

    {
        "parent_table": "orders",
        "parent_column": "order_id",
        "child_table": "reviews",
        "child_column": "order_id",
        "expected_relationship": "1:N possible",
        "join_type_recommended": "keep outside baseline risk features",
        "purpose": "post-delivery NLP/customer impact"
    },

    {
        "parent_table": "products",
        "parent_column": "product_id",
        "child_table": "order_items",
        "child_column": "product_id",
        "expected_relationship": "1:N",
        "join_type_recommended": "many_to_one item -> product",
        "purpose": "product attributes"
    },

    {
        "parent_table": "sellers",
        "parent_column": "seller_id",
        "child_table": "order_items",
        "child_column": "seller_id",
        "expected_relationship": "1:N",
        "join_type_recommended": "many_to_one item -> seller",
        "purpose": "seller attributes"
    },

    {
        "parent_table": "translation",
        "parent_column": "product_category_name",
        "child_table": "products",
        "child_column": "product_category_name",
        "expected_relationship": "1:N with known translation gaps",
        "join_type_recommended": "LEFT JOIN only",
        "purpose": "presentation/English category label"
    },
]


# =============================================================================
# DESCRIÇÃO DE COLUNAS
# =============================================================================

DESC = {

    # customers ---------------------------------------------------------------

    ("customers", "customer_id"):
        "Identificador de cliente específico do pedido.",

    ("customers", "customer_unique_id"):
        "Identificador utilizado para reconhecer o mesmo cliente em "
        "compras diferentes.",

    ("customers", "customer_zip_code_prefix"):
        "Prefixo do CEP do endereço do cliente.",

    ("customers", "customer_city"):
        "Cidade associada ao endereço do cliente.",

    ("customers", "customer_state"):
        "UF associada ao endereço do cliente.",


    # geolocation -------------------------------------------------------------

    ("geolocation", "geolocation_zip_code_prefix"):
        "Prefixo de CEP associado à observação geográfica.",

    ("geolocation", "geolocation_lat"):
        "Latitude associada à observação geográfica.",

    ("geolocation", "geolocation_lng"):
        "Longitude associada à observação geográfica.",

    ("geolocation", "geolocation_city"):
        "Cidade associada à observação geográfica.",

    ("geolocation", "geolocation_state"):
        "UF associada à observação geográfica.",


    # order_items -------------------------------------------------------------

    ("order_items", "order_id"):
        "Identificador do pedido ao qual o item pertence.",

    ("order_items", "order_item_id"):
        "Posição/sequência do item dentro do pedido.",

    ("order_items", "product_id"):
        "Identificador do produto comprado.",

    ("order_items", "seller_id"):
        "Identificador do vendedor responsável pelo item.",

    ("order_items", "shipping_limit_date"):
        "Data limite relacionada ao envio do item pelo seller.",

    ("order_items", "price"):
        "Preço do item.",

    ("order_items", "freight_value"):
        "Valor de frete associado ao item.",


    # payments ----------------------------------------------------------------

    ("payments", "order_id"):
        "Identificador do pedido associado ao pagamento.",

    ("payments", "payment_sequential"):
        "Número sequencial do registro de pagamento dentro do pedido.",

    ("payments", "payment_type"):
        "Tipo/meio de pagamento.",

    ("payments", "payment_installments"):
        "Número de parcelas registrado para o pagamento.",

    ("payments", "payment_value"):
        "Valor registrado no pagamento.",


    # reviews -----------------------------------------------------------------

    ("reviews", "review_id"):
        "Identificador do registro de avaliação.",

    ("reviews", "order_id"):
        "Identificador do pedido avaliado.",

    ("reviews", "review_score"):
        "Nota atribuída pelo cliente à experiência da compra.",

    ("reviews", "review_comment_title"):
        "Título opcional do comentário da avaliação.",

    ("reviews", "review_comment_message"):
        "Texto opcional do comentário da avaliação.",

    ("reviews", "review_creation_date"):
        "Data de criação/solicitação associada à avaliação.",

    ("reviews", "review_answer_timestamp"):
        "Timestamp da resposta à avaliação.",


    # orders ------------------------------------------------------------------

    ("orders", "order_id"):
        "Identificador único do pedido.",

    ("orders", "customer_id"):
        "Identificador do cliente específico associado ao pedido.",

    ("orders", "order_status"):
        "Estado registrado do ciclo de vida do pedido.",

    ("orders", "order_purchase_timestamp"):
        "Timestamp da realização da compra.",

    ("orders", "order_approved_at"):
        "Timestamp registrado de aprovação do pedido/pagamento.",

    ("orders", "order_delivered_carrier_date"):
        "Timestamp registrado da entrega do pedido ao parceiro logístico.",

    ("orders", "order_delivered_customer_date"):
        "Timestamp registrado da entrega efetiva ao cliente.",

    ("orders", "order_estimated_delivery_date"):
        "Data estimada/prometida de entrega associada ao pedido.",


    # products ----------------------------------------------------------------

    ("products", "product_id"):
        "Identificador do produto.",

    ("products", "product_category_name"):
        "Categoria do produto em português.",

    ("products", "product_name_lenght"):
        "Comprimento registrado do nome do produto.",

    ("products", "product_description_lenght"):
        "Comprimento registrado da descrição do produto.",

    ("products", "product_photos_qty"):
        "Quantidade de fotos associadas ao produto.",

    ("products", "product_weight_g"):
        "Peso do produto em gramas.",

    ("products", "product_length_cm"):
        "Comprimento físico do produto em centímetros.",

    ("products", "product_height_cm"):
        "Altura física do produto em centímetros.",

    ("products", "product_width_cm"):
        "Largura física do produto em centímetros.",


    # sellers -----------------------------------------------------------------

    ("sellers", "seller_id"):
        "Identificador do seller.",

    ("sellers", "seller_zip_code_prefix"):
        "Prefixo do CEP cadastrado do seller.",

    ("sellers", "seller_city"):
        "Cidade cadastrada do seller.",

    ("sellers", "seller_state"):
        "UF cadastrada do seller.",


    # translation -------------------------------------------------------------

    ("translation", "product_category_name"):
        "Categoria do produto em português.",

    ("translation", "product_category_name_english"):
        "Tradução da categoria do produto para inglês.",
}


# =============================================================================
# ALIASES IMPORTANTES PARA BUSCA
# =============================================================================

ALIASES = {

    ("orders", "order_purchase_timestamp"): {
        "pt":
            "data da compra; hora da compra; checkout; instante da compra; "
            "momento da compra",
        "en":
            "purchase timestamp; checkout time; order time; purchase time",
        "research":
            "prediction time; decision point; event time; temporal leakage"
    },

    ("orders", "order_estimated_delivery_date"): {
        "pt":
            "data prometida; prazo prometido; previsão de entrega; "
            "data estimada; SLA de entrega",
        "en":
            "promised delivery date; estimated delivery date; EDD; "
            "delivery promise; delivery SLA",
        "research":
            "delivery promise; promise accuracy; ETA; EDD; "
            "promise optimization; late delivery"
    },

    ("orders", "order_delivered_customer_date"): {
        "pt":
            "data real de entrega; entrega efetiva; chegada ao cliente",
        "en":
            "actual delivery date; customer delivery timestamp; "
            "actual arrival",
        "research":
            "delivery ground truth; fulfillment outcome; delivery label"
    },

    ("order_items", "shipping_limit_date"): {
        "pt":
            "prazo de postagem; limite de envio; prazo do seller",
        "en":
            "shipping limit; seller shipping deadline; dispatch deadline",
        "research":
            "seller handling time; dispatch SLA; point-in-time availability"
    },

    ("order_items", "freight_value"): {
        "pt":
            "frete; custo do frete; valor de envio",
        "en":
            "freight; shipping cost; delivery fee",
        "research":
            "shipping cost; logistics cost; fulfillment cost"
    },

    ("order_items", "seller_id"): {
        "pt":
            "vendedor; seller; lojista",
        "en":
            "seller; merchant; vendor",
        "research":
            "seller performance; merchant risk; historical seller features"
    },

    ("products", "product_weight_g"): {
        "pt":
            "peso; peso do produto",
        "en":
            "product weight; item weight",
        "research":
            "package characteristics; logistics features"
    },

    ("geolocation", "geolocation_lat"): {
        "pt":
            "latitude; localização",
        "en":
            "latitude; geolocation",
        "research":
            "geospatial features; delivery distance; logistics distance"
    },

    ("geolocation", "geolocation_lng"): {
        "pt":
            "longitude; localização",
        "en":
            "longitude; geolocation",
        "research":
            "geospatial features; delivery distance; logistics distance"
    },

    ("reviews", "review_comment_message"): {
        "pt":
            "comentário; avaliação textual; reclamação; feedback",
        "en":
            "review text; feedback; complaint; customer comment",
        "research":
            "NLP; sentiment analysis; complaint classification; "
            "customer experience"
    },
}


# =============================================================================
# UTILIDADES
# =============================================================================

def sha256_file(path, block_size=1024 * 1024):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(block_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_json_dump(obj, path):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )


def pct(a, b):

    if not b:
        return np.nan

    return 100.0 * a / b


def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).replace(
        "\n",
        " "
    )

    return text[:250]


def examples(series, n=5):

    vals = (
        series
        .dropna()
        .drop_duplicates()
        .head(n)
        .tolist()
    )

    return " | ".join(
        clean_text(x)
        for x in vals
    )


# =============================================================================
# SEMANTIC TYPE
# =============================================================================

def semantic_type(table, column, series):

    name = column.lower()

    if column in {
        "order_id",
        "customer_id",
        "customer_unique_id",
        "product_id",
        "seller_id",
        "review_id"
    }:
        return "identifier"

    if "timestamp" in name or name.endswith("_date") or name.endswith("_at"):
        return "datetime"

    if name.endswith("_state"):
        return "geographic_state"

    if "zip_code_prefix" in name:
        return "postal_prefix"

    if name.endswith("_city"):
        return "geographic_city"

    if name in {
        "price",
        "freight_value",
        "payment_value"
    }:
        return "monetary"

    if "weight" in name:
        return "physical_weight"

    if any(
        token in name
        for token in [
            "length_cm",
            "height_cm",
            "width_cm"
        ]
    ):
        return "physical_dimension"

    if name in {
        "geolocation_lat",
        "geolocation_lng"
    }:
        return "geographic_coordinate"

    if name in {
        "review_comment_title",
        "review_comment_message"
    }:
        return "free_text"

    if name in {
        "order_status",
        "payment_type",
        "product_category_name",
        "product_category_name_english"
    }:
        return "categorical"

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return "numeric"

    return "categorical_or_text"


# =============================================================================
# UNIDADE
# =============================================================================

def unit_for(column):

    if column in {
        "price",
        "freight_value",
        "payment_value"
    }:
        return "BRL"

    if column == "product_weight_g":
        return "g"

    if column in {
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    }:
        return "cm"

    if column == "geolocation_lat":
        return "degrees_latitude"

    if column == "geolocation_lng":
        return "degrees_longitude"

    if column == "review_score":
        return "score_1_5"

    if column == "payment_installments":
        return "count"

    if column == "product_photos_qty":
        return "count"

    return ""


# =============================================================================
# DISPONIBILIDADE TEMPORAL / LEAKAGE
# =============================================================================

def temporal_policy(table, column):

    # default
    result = {
        "event_stage": "reference_or_unknown",
        "available_at_t0": "UNKNOWN",
        "availability_confidence": "LOW",
        "allowed_as_baseline_feature": "REVIEW",
        "allowed_as_target_source": "NO",
        "leakage_risk": "REVIEW"
    }

    # -------------------------------------------------------------------------
    # reference/master data
    # -------------------------------------------------------------------------

    if table in {
        "customers",
        "products",
        "sellers",
        "geolocation",
        "translation"
    }:

        result.update(
            {
                "event_stage": "reference/master data",
                "available_at_t0": "LIKELY_YES",
                "availability_confidence": "MEDIUM",
                "allowed_as_baseline_feature": "YES_CANDIDATE",
                "leakage_risk": "LOW_TO_REVIEW"
            }
        )

    # IDs não entram crus
    if column.endswith("_id") or column == "customer_unique_id":

        result[
            "allowed_as_baseline_feature"
        ] = "NO_RAW_IDENTIFIER"

    # -------------------------------------------------------------------------
    # order item
    # -------------------------------------------------------------------------

    if table == "order_items":

        result.update(
            {
                "event_stage": "purchase/order composition",
                "available_at_t0": "LIKELY_YES",
                "availability_confidence": "MEDIUM",
                "allowed_as_baseline_feature": "YES_CANDIDATE",
                "leakage_risk": "LOW_TO_REVIEW"
            }
        )

        if column in {
            "order_id",
            "product_id",
            "seller_id"
        }:
            result[
                "allowed_as_baseline_feature"
            ] = "NO_RAW_IDENTIFIER"

        if column == "shipping_limit_date":

            result.update(
                {
                    "available_at_t0": "UNVERIFIED",
                    "availability_confidence": "LOW",
                    "allowed_as_baseline_feature": "HOLD",
                    "leakage_risk": "POTENTIAL"
                }
            )

    # -------------------------------------------------------------------------
    # payment
    # -------------------------------------------------------------------------

    if table == "payments":

        result.update(
            {
                "event_stage": "checkout/payment",
                "available_at_t0": "ASSUMED_AT_CHECKOUT_NEEDS_VERIFICATION",
                "availability_confidence": "MEDIUM_LOW",
                "allowed_as_baseline_feature": "YES_CANDIDATE_AFTER_PROVENANCE",
                "leakage_risk": "REVIEW"
            }
        )

        if column == "order_id":
            result[
                "allowed_as_baseline_feature"
            ] = "NO_RAW_IDENTIFIER"

    # -------------------------------------------------------------------------
    # reviews
    # -------------------------------------------------------------------------

    if table == "reviews":

        result.update(
            {
                "event_stage": "post_purchase/post_delivery",
                "available_at_t0": "NO",
                "availability_confidence": "HIGH",
                "allowed_as_baseline_feature": "NO",
                "leakage_risk": "CRITICAL_IF_USED"
            }
        )

    # -------------------------------------------------------------------------
    # orders
    # -------------------------------------------------------------------------

    if table == "orders":

        if column == "order_id":

            result.update(
                {
                    "event_stage": "purchase",
                    "available_at_t0": "YES",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "NO_RAW_IDENTIFIER",
                    "leakage_risk": "LOW"
                }
            )

        elif column == "customer_id":

            result.update(
                {
                    "event_stage": "purchase",
                    "available_at_t0": "YES",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "NO_RAW_IDENTIFIER",
                    "leakage_risk": "LOW"
                }
            )

        elif column == "order_purchase_timestamp":

            result.update(
                {
                    "event_stage": "purchase",
                    "available_at_t0": "YES",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "DERIVE_CALENDAR_FEATURES",
                    "leakage_risk": "LOW"
                }
            )

        elif column == "order_estimated_delivery_date":

            result.update(
                {
                    "event_stage": "purchase/promise",
                    "available_at_t0": "YES_CANDIDATE",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature":
                        "DERIVE_PROMISED_LEAD_TIME",
                    "allowed_as_target_source": "YES",
                    "leakage_risk": "LOW"
                }
            )

        elif column == "order_status":

            result.update(
                {
                    "event_stage": "lifecycle/final recorded state",
                    "available_at_t0": "NO_AS_FINAL_RAW_VALUE",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "NO",
                    "leakage_risk": "CRITICAL_IF_FINAL_STATUS_USED"
                }
            )

        elif column == "order_approved_at":

            result.update(
                {
                    "event_stage": "post_purchase approval",
                    "available_at_t0": "NO",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "NO",
                    "leakage_risk": "HIGH"
                }
            )

        elif column == "order_delivered_carrier_date":

            result.update(
                {
                    "event_stage": "shipping/handoff",
                    "available_at_t0": "NO",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "NO",
                    "leakage_risk": "CRITICAL_IF_USED"
                }
            )

        elif column == "order_delivered_customer_date":

            result.update(
                {
                    "event_stage": "delivery/outcome",
                    "available_at_t0": "NO",
                    "availability_confidence": "HIGH",
                    "allowed_as_baseline_feature": "NO",
                    "allowed_as_target_source": "YES",
                    "leakage_risk": "CRITICAL_IF_USED"
                }
            )

    return result


# =============================================================================
# APLICABILIDADE / MISSINGNESS
# =============================================================================

def applicability_policy(table, column):

    if table == "reviews" and column in {
        "review_comment_title",
        "review_comment_message"
    }:
        return (
            "review record exists AND customer supplied optional text"
        )

    if table == "products" and column != "product_id":
        return (
            "product_id resolves to products; attribute evaluated only "
            "when product entity exists"
        )

    if table == "sellers" and column != "seller_id":
        return (
            "seller_id resolves to sellers"
        )

    if table == "customers" and column != "customer_id":
        return (
            "customer_id resolves to customers"
        )

    if table == "geolocation":
        return (
            "ZIP prefix has geolocation observation"
        )

    if column == "order_delivered_customer_date":
        return (
            "delivery outcome is observable; do not equate not-observed "
            "with on-time"
        )

    if column == "order_delivered_carrier_date":
        return (
            "carrier handoff event applicable/observed"
        )

    return "row/entity exists"


def missing_reason_policy(table, column):

    if table == "reviews":
        return (
            "distinguish OPTIONAL_TEXT / POST_EVENT_NOT_AVAILABLE / "
            "ATTRIBUTE_MISSING"
        )

    if table == "products" and column != "product_id":
        return (
            "distinguish RELATION_MISSING from ATTRIBUTE_MISSING"
        )

    if table in {
        "customers",
        "sellers"
    }:
        return (
            "distinguish RELATION_MISSING from ATTRIBUTE_MISSING"
        )

    if column in {
        "order_delivered_carrier_date",
        "order_delivered_customer_date"
    }:
        return (
            "distinguish NOT_YET_OBSERVED / NOT_APPLICABLE / CENSORED / "
            "ATTRIBUTE_MISSING"
        )

    return "OBSERVED or ATTRIBUTE_MISSING"


# =============================================================================
# EVIDÊNCIA
# =============================================================================

def evidence_policy(table, column):

    # Tudo é observado na cópia RAW.
    # Significado e verdade externa são outra questão.

    if (
        table == "customers"
        and column in {
            "customer_id",
            "customer_unique_id"
        }
    ):

        return {
            "source_evidence": "E0_RAW_OBSERVED",
            "meaning_evidence": "E4_OFFICIAL_DATASET_DESCRIPTION",
            "external_ground_truth": "NOT_AVAILABLE"
        }

    return {
        "source_evidence": "E0_RAW_OBSERVED",
        "meaning_evidence":
            "E3_SCHEMA_AND_PROJECT_INTERPRETATION_REQUIRES_TRACEABILITY",
        "external_ground_truth": "NOT_AVAILABLE"
    }


# =============================================================================
# COLUMN PROFILE
# =============================================================================

def profile_column(
    table,
    column,
    series
):

    total = len(series)

    missing = int(
        series.isna().sum()
    )

    non_null = (
        series.dropna()
    )

    unique = int(
        non_null.nunique()
    )

    unique_ratio = (
        pct(
            unique,
            len(non_null)
        )
        if len(non_null)
        else np.nan
    )

    top_value = ""

    top_count = 0

    top_pct = np.nan

    entropy_bits = np.nan


    if len(non_null):

        counts = non_null.value_counts(
            dropna=False
        )

        top_value = clean_text(
            counts.index[0]
        )

        top_count = int(
            counts.iloc[0]
        )

        top_pct = pct(
            top_count,
            len(non_null)
        )

        # Entropia apenas em domínios razoáveis.
        if unique <= 500:

            probs = (
                counts
                /
                counts.sum()
            ).to_numpy(
                dtype=float
            )

            entropy_bits = float(
                -np.sum(
                    probs
                    *
                    np.log2(
                        probs
                    )
                )
            )


    result = {
        "table":
            table,

        "column":
            column,

        "raw_dtype":
            str(
                series.dtype
            ),

        "rows":
            total,

        "non_null":
            len(non_null),

        "missing":
            missing,

        "missing_pct":
            pct(
                missing,
                total
            ),

        "completeness_pct":
            100.0
            -
            pct(
                missing,
                total
            ),

        "cardinality":
            unique,

        "unique_ratio_pct":
            unique_ratio,

        "top_value":
            top_value,

        "top_value_count":
            top_count,

        "top_value_pct":
            top_pct,

        "entropy_bits":
            entropy_bits,

        "examples":
            examples(
                series
            ),

        "numeric_min":
            np.nan,

        "numeric_p01":
            np.nan,

        "numeric_median":
            np.nan,

        "numeric_mean":
            np.nan,

        "numeric_p99":
            np.nan,

        "numeric_max":
            np.nan,

        "string_length_min":
            np.nan,

        "string_length_median":
            np.nan,

        "string_length_max":
            np.nan,

        "datetime_parse_failures":
            np.nan,

        "datetime_min":
            "",

        "datetime_max":
            ""
    }


    # -------------------------------------------------------------------------
    # numeric
    # -------------------------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        series
    ):

        x = pd.to_numeric(
            series,
            errors="coerce"
        )

        x = x.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        ).dropna()

        if len(x):

            result.update(
                {
                    "numeric_min":
                        float(
                            x.min()
                        ),

                    "numeric_p01":
                        float(
                            x.quantile(
                                0.01
                            )
                        ),

                    "numeric_median":
                        float(
                            x.median()
                        ),

                    "numeric_mean":
                        float(
                            x.mean()
                        ),

                    "numeric_p99":
                        float(
                            x.quantile(
                                0.99
                            )
                        ),

                    "numeric_max":
                        float(
                            x.max()
                        ),
                }
            )


    # -------------------------------------------------------------------------
    # string lengths
    # -------------------------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(
            series
        )
        or
        pd.api.types.is_string_dtype(
            series
        )
    ):

        if len(non_null):

            lengths = (
                non_null
                .astype(str)
                .str.len()
            )

            result.update(
                {
                    "string_length_min":
                        int(
                            lengths.min()
                        ),

                    "string_length_median":
                        float(
                            lengths.median()
                        ),

                    "string_length_max":
                        int(
                            lengths.max()
                        ),
                }
            )


    # -------------------------------------------------------------------------
    # datetime candidate
    # -------------------------------------------------------------------------

    lower = column.lower()

    if (
        "timestamp" in lower
        or
        lower.endswith("_date")
        or
        lower.endswith("_at")
    ):

        parsed = pd.to_datetime(
            series,
            errors="coerce"
        )

        failures = int(
            (
                series.notna()
                &
                parsed.isna()
            ).sum()
        )

        result[
            "datetime_parse_failures"
        ] = failures

        if parsed.notna().any():

            result[
                "datetime_min"
            ] = str(
                parsed.min()
            )

            result[
                "datetime_max"
            ] = str(
                parsed.max()
            )


    return result


# =============================================================================
# BUILD
# =============================================================================

print(
    "=" * 100
)

print(
    "DATASET KNOWLEDGE & TRUTH REGISTRY"
)

print(
    "=" * 100
)

print()

print(
    f"Projeto : {PROJECT}"
)

print(
    f"RAW     : {RAW}"
)

print(
    f"Saída   : {META}"
)

print()


# =============================================================================
# VALIDAR RAW
# =============================================================================

missing_files = [

    filename
    for filename
    in FILES.values()

    if not (
        RAW / filename
    ).exists()
]


if missing_files:

    raise SystemExit(
        f"Arquivos RAW ausentes: {missing_files}"
    )


# =============================================================================
# GATE SUMMARIES
# =============================================================================

gate_summaries = {}


for gate, path in GATE_SUMMARY_PATHS.items():

    if path.exists():

        try:

            with path.open(
                "r",
                encoding="utf-8"
            ) as f:

                gate_summaries[
                    gate
                ] = json.load(f)

        except Exception as exc:

            gate_summaries[
                gate
            ] = {
                "status":
                    "READ_ERROR",

                "error":
                    str(exc)
            }

    else:

        gate_summaries[
            gate
        ] = {
            "status":
                "NOT_AVAILABLE"
        }


# =============================================================================
# TABLE + COLUMN CATALOGS
# =============================================================================

table_rows = []

column_rows = []

value_domain_rows = []

file_fingerprints = []


DOMAIN_COLUMNS = {

    ("orders", "order_status"),

    ("payments", "payment_type"),

    ("customers", "customer_state"),

    ("sellers", "seller_state"),

    ("geolocation", "geolocation_state"),

    ("products", "product_category_name"),

    ("reviews", "review_score")
}


print(
    "[1/9] Profiling dos 9 arquivos RAW..."
)


for table, filename in FILES.items():

    path = (
        RAW
        /
        filename
    )

    print(
        f"  - {table:<12} {filename}"
    )

    file_size = (
        path.stat().st_size
    )

    digest = sha256_file(
        path
    )

    df = pd.read_csv(
        path,
        low_memory=False
    )


    # -------------------------------------------------------------------------
    # table
    # -------------------------------------------------------------------------

    total_cells = (
        df.shape[0]
        *
        df.shape[1]
    )

    missing_cells = int(
        df.isna()
        .sum()
        .sum()
    )

    duplicated_rows = int(
        df.duplicated()
        .sum()
    )

    pk = PRIMARY_KEYS[
        table
    ]

    pk_null_rows = np.nan

    pk_duplicate_rows = np.nan

    if pk:

        pk_null_rows = int(
            df[
                pk
            ]
            .isna()
            .any(
                axis=1
            )
            .sum()
        )

        pk_duplicate_rows = int(
            df
            .duplicated(
                subset=pk,
                keep=False
            )
            .sum()
        )


    table_rows.append(
        {
            "table":
                table,

            "file":
                filename,

            "entity":
                TABLE_META[
                    table
                ][
                    "entity"
                ],

            "grain":
                TABLE_META[
                    table
                ][
                    "grain"
                ],

            "description_pt":
                TABLE_META[
                    table
                ][
                    "description_pt"
                ],

            "lifecycle_role":
                TABLE_META[
                    table
                ][
                    "lifecycle_role"
                ],

            "rows":
                df.shape[0],

            "columns":
                df.shape[1],

            "file_size_mb":
                file_size
                /
                (
                    1024 ** 2
                ),

            "sha256":
                digest,

            "full_row_duplicates":
                duplicated_rows,

            "missing_cells":
                missing_cells,

            "missing_cells_pct":
                pct(
                    missing_cells,
                    total_cells
                ),

            "primary_key":
                "+".join(
                    pk
                )
                if pk
                else "",

            "pk_null_rows":
                pk_null_rows,

            "pk_duplicate_rows":
                pk_duplicate_rows
        }
    )


    file_fingerprints.append(
        {
            "table":
                table,

            "file":
                filename,

            "size_bytes":
                file_size,

            "sha256":
                digest
        }
    )


    # -------------------------------------------------------------------------
    # columns
    # -------------------------------------------------------------------------

    for position, column in enumerate(
        df.columns,
        start=1
    ):

        s = df[
            column
        ]

        profile = profile_column(
            table,
            column,
            s
        )

        semantic = semantic_type(
            table,
            column,
            s
        )

        temporal = temporal_policy(
            table,
            column
        )

        evidence = evidence_policy(
            table,
            column
        )

        aliases = ALIASES.get(
            (
                table,
                column
            ),
            {}
        )


        key_role = "NONE"

        if column in PRIMARY_KEYS[
            table
        ]:

            key_role = (
                "PRIMARY_KEY_COMPONENT"
            )

        if (
            table,
            column
        ) in FOREIGN_KEYS:

            if key_role == "NONE":
                key_role = "FOREIGN_KEY"
            else:
                key_role += "+FOREIGN_KEY"


        profile.update(
            {
                "position":
                    position,

                "entity":
                    TABLE_META[
                        table
                    ][
                        "entity"
                    ],

                "semantic_type":
                    semantic,

                "unit":
                    unit_for(
                        column
                    ),

                "description_pt":
                    DESC.get(
                        (
                            table,
                            column
                        ),
                        "Descrição de domínio ainda não revisada."
                    ),

                "description_status":
                    (
                        "CURATED_PROJECT_REGISTRY"
                        if (
                            table,
                            column
                        )
                        in DESC
                        else
                        "NEEDS_DOMAIN_REVIEW"
                    ),

                "key_role":
                    key_role,

                "foreign_key_target":
                    FOREIGN_KEYS.get(
                        (
                            table,
                            column
                        ),
                        ""
                    ),

                "applicability_condition":
                    applicability_policy(
                        table,
                        column
                    ),

                "missing_reason_policy":
                    missing_reason_policy(
                        table,
                        column
                    ),

                "aliases_pt":
                    aliases.get(
                        "pt",
                        column.replace(
                            "_",
                            " "
                        )
                    ),

                "aliases_en":
                    aliases.get(
                        "en",
                        column.replace(
                            "_",
                            " "
                        )
                    ),

                "research_terms":
                    aliases.get(
                        "research",
                        ""
                    ),

                **temporal,

                **evidence
            }
        )


        column_rows.append(
            profile
        )


        # ---------------------------------------------------------------------
        # value domain
        # ---------------------------------------------------------------------

        if (
            table,
            column
        ) in DOMAIN_COLUMNS:

            counts = (
                s
                .dropna()
                .value_counts()
            )

            for value, count in counts.items():

                value_domain_rows.append(
                    {
                        "table":
                            table,

                        "column":
                            column,

                        "value":
                            clean_text(
                                value
                            ),

                        "count":
                            int(
                                count
                            ),

                        "pct_non_null":
                            pct(
                                int(
                                    count
                                ),
                                int(
                                    s.notna()
                                    .sum()
                                )
                            ),

                        "domain_status":
                            "OBSERVED",

                        "business_label":
                            clean_text(
                                value
                            ),

                        "notes":
                            ""
                    }
                )


    del df


table_catalog = pd.DataFrame(
    table_rows
)

column_catalog = pd.DataFrame(
    column_rows
)

value_domain = pd.DataFrame(
    value_domain_rows
)


table_catalog.to_csv(
    META
    /
    "table_catalog.csv",
    index=False
)


column_catalog.to_csv(
    META
    /
    "column_catalog.csv",
    index=False
)


value_domain.to_csv(
    META
    /
    "value_domain_registry.csv",
    index=False
)


# =============================================================================
# RELATIONSHIP REGISTRY
# =============================================================================

print(
    "[2/9] Construindo Relationship Registry..."
)


relationship_rows = []


for rel in RELATIONSHIPS:

    parent_table = rel[
        "parent_table"
    ]

    child_table = rel[
        "child_table"
    ]

    parent_column = rel[
        "parent_column"
    ]

    child_column = rel[
        "child_column"
    ]


    parent = pd.read_csv(
        RAW
        /
        FILES[
            parent_table
        ],
        usecols=[
            parent_column
        ],
        low_memory=False
    )


    child = pd.read_csv(
        RAW
        /
        FILES[
            child_table
        ],
        usecols=[
            child_column
        ],
        low_memory=False
    )


    parent_values = set(
        parent[
            parent_column
        ]
        .dropna()
        .astype(str)
    )


    child_series = (
        child[
            child_column
        ]
        .dropna()
        .astype(str)
    )


    orphan_mask = (
        ~child_series.isin(
            parent_values
        )
    )


    child_counts = (
        child_series
        .value_counts()
    )


    orphan_rows = int(
        orphan_mask.sum()
    )


    relationship_rows.append(
        {
            **rel,

            "parent_rows":
                len(
                    parent
                ),

            "parent_unique_keys":
                int(
                    parent[
                        parent_column
                    ]
                    .nunique()
                ),

            "child_rows":
                len(
                    child
                ),

            "child_unique_parent_keys":
                int(
                    child[
                        child_column
                    ]
                    .nunique()
                ),

            "average_child_rows_per_referenced_key":
                float(
                    child_counts.mean()
                )
                if len(
                    child_counts
                )
                else np.nan,

            "median_child_rows_per_referenced_key":
                float(
                    child_counts.median()
                )
                if len(
                    child_counts
                )
                else np.nan,

            "max_child_rows_per_referenced_key":
                int(
                    child_counts.max()
                )
                if len(
                    child_counts
                )
                else 0,

            "orphan_rows":
                orphan_rows,

            "orphan_pct":
                pct(
                    orphan_rows,
                    len(
                        child_series
                    )
                ),

            "join_risk":
                (
                    "ROW_MULTIPLICATION_RISK"
                    if rel[
                        "expected_relationship"
                    ].startswith(
                        "1:N"
                    )
                    else
                    "REVIEW"
                )
        }
    )


relationship_registry = pd.DataFrame(
    relationship_rows
)


relationship_registry.to_csv(
    META
    /
    "relationship_registry.csv",
    index=False
)


# =============================================================================
# TRUTH & PROVENANCE
# =============================================================================

print(
    "[3/9] Construindo Truth & Provenance Registry..."
)


truth_cols = [

    "table",
    "column",
    "entity",
    "description_pt",
    "description_status",

    "source_evidence",
    "meaning_evidence",
    "external_ground_truth",

    "event_stage",
    "available_at_t0",
    "availability_confidence",

    "allowed_as_baseline_feature",
    "allowed_as_target_source",
    "leakage_risk",

    "applicability_condition",
    "missing_reason_policy",

    "key_role",
    "foreign_key_target",

    "aliases_pt",
    "aliases_en",
    "research_terms"
]


truth_registry = (
    column_catalog[
        truth_cols
    ]
    .copy()
)


truth_registry.insert(
    0,
    "dataset",
    DATASET_ID
)


truth_registry[
    "source_file"
] = truth_registry[
    "table"
].map(
    FILES
)


truth_registry[
    "source_system"
] = (
    "Olist public dataset distributed through Kaggle"
)


truth_registry[
    "external_system_crosscheck"
] = (
    "NOT_AVAILABLE"
)


truth_registry[
    "project_prediction_time"
] = (
    PREDICTION_T0
)


truth_registry.to_csv(
    META
    /
    "truth_provenance_registry.csv",
    index=False
)


# =============================================================================
# TEMPORAL AVAILABILITY REGISTRY
# =============================================================================

print(
    "[4/9] Construindo Temporal Availability Registry..."
)


temporal_registry = truth_registry[
    [
        "table",
        "column",
        "entity",
        "event_stage",
        "available_at_t0",
        "availability_confidence",
        "allowed_as_baseline_feature",
        "allowed_as_target_source",
        "leakage_risk",
        "project_prediction_time"
    ]
].copy()


def stage_matrix(row):

    stage = row[
        "event_stage"
    ]

    t0 = row[
        "available_at_t0"
    ]


    if t0 in {
        "YES",
        "YES_CANDIDATE",
        "LIKELY_YES",
        "ASSUMED_AT_CHECKOUT_NEEDS_VERIFICATION"
    }:

        purchase = t0

    else:

        purchase = "NO_OR_UNVERIFIED"


    if "delivery" in stage:
        delivery = "YES"
    else:
        delivery = "LIKELY_OR_YES"


    return pd.Series(
        {
            "known_at_purchase":
                purchase,

            "known_at_approval":
                (
                    "YES_OR_LIKELY"
                    if purchase
                    !=
                    "NO_OR_UNVERIFIED"
                    else
                    "STAGE_DEPENDENT"
                ),

            "known_at_shipping":
                (
                    "YES_OR_LIKELY"
                    if "post_delivery"
                    not in stage
                    else
                    "NO"
                ),

            "known_at_delivery":
                delivery
        }
    )


stage_values = temporal_registry.apply(
    stage_matrix,
    axis=1
)


temporal_registry = pd.concat(
    [
        temporal_registry,
        stage_values
    ],
    axis=1
)


temporal_registry[
    "point_in_time_rule"
] = np.where(
    temporal_registry[
        "available_at_t0"
    ].isin(
        [
            "NO",
            "NO_AS_FINAL_RAW_VALUE"
        ]
    ),
    "FORBIDDEN_FOR_BASELINE_T0",
    "VERIFY_AVAILABILITY_BEFORE_FEATURE_USE"
)


temporal_registry.to_csv(
    META
    /
    "temporal_availability_registry.csv",
    index=False
)


# =============================================================================
# QUALITY OBSERVATION REGISTRY
# =============================================================================

print(
    "[5/9] Consolidando Gates de qualidade..."
)


quality_frames = []


for gate_name, path in GATE_PATHS.items():

    if not path.exists():
        continue

    q = pd.read_csv(
        path,
        low_memory=False
    )

    q[
        "registry_source_gate"
    ] = gate_name

    q[
        "registry_source_file"
    ] = str(
        path.relative_to(
            PROJECT
        )
    )

    quality_frames.append(
        q
    )


if quality_frames:

    quality_registry = pd.concat(
        quality_frames,
        ignore_index=True,
        sort=False
    )

else:

    quality_registry = pd.DataFrame()


quality_registry.to_csv(
    META
    /
    "quality_observation_registry.csv",
    index=False
)


# =============================================================================
# UNRESOLVED QUESTIONS
# =============================================================================

print(
    "[6/9] Construindo registro de decisões abertas..."
)


open_questions = [

    {
        "question_id": "OPEN-001",
        "topic": "target_semantics",
        "status": "OPEN",
        "priority": "CRITICAL",
        "question":
            "A promessa deve ser avaliada por timestamp exato ou por "
            "dia-calendário?",
        "current_evidence":
            "1,292 pedidos mudam de classe entre as duas definições.",
        "required_action":
            "Formalizar Label Contract e análise de sensibilidade.",
        "blocks":
            "final label definition"
    },

    {
        "question_id": "OPEN-002",
        "topic": "shipping_limit_date_provenance",
        "status": "OPEN",
        "priority": "HIGH",
        "question":
            "shipping_limit_date era conhecido exatamente no instante da compra?",
        "current_evidence":
            "Campo existe no RAW, mas disponibilidade em t0 não foi provada.",
        "required_action":
            "Manter fora do baseline até confirmação.",
        "blocks":
            "feature eligibility"
    },

    {
        "question_id": "OPEN-003",
        "topic": "conditional_missingness",
        "status": "OPEN",
        "priority": "HIGH",
        "question":
            "Qual a completude dos atributos condicionada à existência "
            "da entidade relacionada?",
        "current_evidence":
            "Gate 03 mistura relationship missingness e attribute missingness "
            "em alguns perfis mensais.",
        "required_action":
            "Executar DQ Gate 03B Conditional & Task Completeness.",
        "blocks":
            "final completeness assessment"
    },

    {
        "question_id": "OPEN-004",
        "topic": "external_ground_truth",
        "status": "OPEN_LIMITATION",
        "priority": "HIGH",
        "question":
            "É possível validar datas/valores contra sistemas operacionais "
            "originais da Olist?",
        "current_evidence":
            "Dataset público não fornece acesso ao sistema transacional fonte.",
        "required_action":
            "Documentar limite: internal consistency != external ground truth.",
        "blocks":
            "external factual verification"
    },

    {
        "question_id": "OPEN-005",
        "topic": "seller_history",
        "status": "OPEN",
        "priority": "HIGH",
        "question":
            "Como construir seller history sem conhecer outcomes futuros?",
        "current_evidence":
            "Compra anterior não implica label anterior já disponível.",
        "required_action":
            "Usar somente eventos cujo outcome estava observável antes do "
            "pedido atual; definir smoothing.",
        "blocks":
            "seller historical features"
    },

    {
        "question_id": "OPEN-006",
        "topic": "multi_seller",
        "status": "OPEN",
        "priority": "MEDIUM",
        "question":
            "Como representar pedidos contendo múltiplos sellers?",
        "current_evidence":
            "Dataset permite mais de um seller por pedido.",
        "required_action":
            "Comparar agregações min/mean/max/count e regra operacional.",
        "blocks":
            "order-level feature aggregation"
    },

    {
        "question_id": "OPEN-007",
        "topic": "geolocation",
        "status": "OPEN",
        "priority": "MEDIUM",
        "question":
            "Como consolidar CEPs com observações em múltiplos estados "
            "ou grande dispersão?",
        "current_evidence":
            "Gate 02 encontrou 8 prefixos associados a múltiplos estados.",
        "required_action":
            "Inspecionar exceções antes da consolidação definitiva.",
        "blocks":
            "geographic reference table"
    },

    {
        "question_id": "OPEN-008",
        "topic": "financial_reconciliation",
        "status": "OPEN",
        "priority": "MEDIUM",
        "question":
            "Qual a semântica dos pedidos com diferença relevante entre "
            "pagamentos e preço+frete?",
        "current_evidence":
            "Auditoria anterior encontrou pequena quantidade de exceções.",
        "required_action":
            "Inspecionar sem alterar valores RAW.",
        "blocks":
            "financial treatment decision"
    },

    {
        "question_id": "OPEN-009",
        "topic": "observation_window",
        "status": "OPEN",
        "priority": "HIGH",
        "question":
            "Como representar setembro/outubro de 2018 na população "
            "supervisionada?",
        "current_evidence":
            "Os últimos meses têm ausência de outcomes entregues observáveis.",
        "required_action":
            "Tratar como janela não madura/censura potencial; não converter "
            "ausência de label em on-time.",
        "blocks":
            "final supervised cohort"
    },

    {
        "question_id": "OPEN-010",
        "topic": "payment_availability",
        "status": "OPEN",
        "priority": "MEDIUM",
        "question":
            "payment_type/value/installments são operacionalmente disponíveis "
            "no exato t0 definido como purchase_timestamp?",
        "current_evidence":
            "São informações de checkout/pagamento, mas availability time "
            "não é explicitamente fornecido.",
        "required_action":
            "Manter como candidate e documentar hipótese.",
        "blocks":
            "baseline feature contract"
    }
]


# incorporar decisões da Auditoria 03
if PENDING_AUDIT_03.exists():

    try:

        with PENDING_AUDIT_03.open(
            "r",
            encoding="utf-8"
        ) as f:

            old_pending = json.load(f)

        existing_topics = {
            x[
                "topic"
            ]
            for x
            in open_questions
        }

        for key, value in old_pending.items():

            if key in existing_topics:
                continue

            open_questions.append(
                {
                    "question_id":
                        f"AUD03-{len(open_questions)+1:03d}",

                    "topic":
                        key,

                    "status":
                        "OPEN",

                    "priority":
                        "MEDIUM",

                    "question":
                        str(
                            value
                        ),

                    "current_evidence":
                        "Imported from Auditoria 03 pending decisions.",

                    "required_action":
                        "Review before final treatment.",

                    "blocks":
                        "data treatment plan"
                }
            )

    except Exception:
        pass


unresolved = pd.DataFrame(
    open_questions
)


unresolved.to_csv(
    META
    /
    "unresolved_questions.csv",
    index=False
)


# =============================================================================
# DATASET MANIFEST
# =============================================================================

print(
    "[7/9] Construindo Dataset Manifest..."
)


combined_fingerprint_source = "".join(
    sorted(
        x[
            "sha256"
        ]
        for x
        in file_fingerprints
    )
)


dataset_fingerprint = hashlib.sha256(
    combined_fingerprint_source.encode(
        "utf-8"
    )
).hexdigest()


dataset_manifest = {

    "project":
        PROJECT_NAME,

    "dataset": {
        "name":
            DATASET_NAME,

        "identifier":
            DATASET_ID,

        "publisher":
            "Olist",

        "distribution":
            "Kaggle",

        "data_nature":
            "real commercial data, anonymized according to dataset description",

        "time_coverage_description":
            "2016-2018",

        "files_expected":
            9,

        "columns_observed":
            int(
                table_catalog[
                    "columns"
                ].sum()
            ),

        "dataset_sha256_composite":
            dataset_fingerprint,

        "file_fingerprints":
            file_fingerprints
    },

    "intended_project_use": {
        "primary_problem":
            "early warning of violation of delivery promise",

        "prediction_time":
            PREDICTION_T0,

        "unit_of_analysis_target":
            "order_id",

        "main_model_grain":
            "one row per order",

        "future_modules": [
            "ETA prediction",
            "uncertainty / quantile prediction",
            "NLP customer impact analysis",
            "decision engine",
            "MLOps monitoring"
        ]
    },

    "ground_truth_statement": {
        "source_truth":
            "Values are observed in the public Olist dataset.",

        "internal_validation":
            "Structural, semantic and statistical gates available.",

        "external_truth":
            "Original Olist operational systems are not available for "
            "independent verification.",

        "important_distinction":
            "Internal consistency does not prove external ground-truth accuracy."
    },

    "quality_gate_status": {
        k:
            v.get(
                "status",
                "UNKNOWN"
            )
        for k, v
        in gate_summaries.items()
    },

    "known_constraints": [
        "orders may contain multiple items",
        "items may be associated with different sellers",
        "reviews are post-purchase/post-delivery information",
        "geolocation contains multiple observations per ZIP prefix",
        "some lifecycle timestamps have ordering anomalies",
        "late-label semantics require explicit contract",
        "late observation window is not mature in final raw months",
        "availability time is not explicitly provided for every field"
    ]
}


safe_json_dump(
    dataset_manifest,
    META
    /
    "dataset_manifest.json"
)


# =============================================================================
# SEARCH KNOWLEDGE JSONL
# =============================================================================

print(
    "[8/9] Construindo Search Knowledge..."
)


search_records = []


# dataset ----------------------------------------------------------------------

search_records.append(
    {
        "id":
            "dataset:olist",

        "type":
            "dataset",

        "title":
            DATASET_NAME,

        "dataset":
            DATASET_ID,

        "search_text":
            (
                "Olist Brazilian ecommerce Brazil orders delivery logistics "
                "shipping freight payments sellers products customers reviews "
                "delivery risk late delivery ETA delivery promise "
                "fulfillment data quality"
            ),

        "payload":
            dataset_manifest[
                "intended_project_use"
            ]
    }
)


# tables -----------------------------------------------------------------------

for _, row in table_catalog.iterrows():

    search_records.append(
        {
            "id":
                f"table:{row['table']}",

            "type":
                "table",

            "title":
                row[
                    "table"
                ],

            "table":
                row[
                    "table"
                ],

            "search_text":
                " ".join(
                    [
                        str(
                            row[
                                "table"
                            ]
                        ),

                        str(
                            row[
                                "entity"
                            ]
                        ),

                        str(
                            row[
                                "grain"
                            ]
                        ),

                        str(
                            row[
                                "description_pt"
                            ]
                        ),

                        str(
                            row[
                                "lifecycle_role"
                            ]
                        )
                    ]
                ),

            "payload": {
                "rows":
                    int(
                        row[
                            "rows"
                        ]
                    ),

                "columns":
                    int(
                        row[
                            "columns"
                        ]
                    ),

                "grain":
                    row[
                        "grain"
                    ]
            }
        }
    )


# columns ----------------------------------------------------------------------

for _, row in column_catalog.iterrows():

    search_text = " ".join(
        [
            str(
                row[
                    "table"
                ]
            ),

            str(
                row[
                    "column"
                ]
            ),

            str(
                row[
                    "description_pt"
                ]
            ),

            str(
                row[
                    "semantic_type"
                ]
            ),

            str(
                row[
                    "aliases_pt"
                ]
            ),

            str(
                row[
                    "aliases_en"
                ]
            ),

            str(
                row[
                    "research_terms"
                ]
            ),

            str(
                row[
                    "event_stage"
                ]
            ),

            str(
                row[
                    "leakage_risk"
                ]
            )
        ]
    )


    search_records.append(
        {
            "id":
                f"column:{row['table']}.{row['column']}",

            "type":
                "column",

            "title":
                f"{row['table']}.{row['column']}",

            "table":
                row[
                    "table"
                ],

            "column":
                row[
                    "column"
                ],

            "search_text":
                search_text,

            "payload": {
                "description":
                    row[
                        "description_pt"
                    ],

                "semantic_type":
                    row[
                        "semantic_type"
                    ],

                "available_at_t0":
                    row[
                        "available_at_t0"
                    ],

                "feature_policy":
                    row[
                        "allowed_as_baseline_feature"
                    ],

                "target_policy":
                    row[
                        "allowed_as_target_source"
                    ],

                "leakage_risk":
                    row[
                        "leakage_risk"
                    ],

                "missing_pct":
                    row[
                        "missing_pct"
                    ],

                "cardinality":
                    row[
                        "cardinality"
                    ]
            }
        }
    )


# relationships ---------------------------------------------------------------

for _, row in relationship_registry.iterrows():

    search_records.append(
        {
            "id":
                (
                    f"relationship:"
                    f"{row['parent_table']}.{row['parent_column']}"
                    f"->{row['child_table']}.{row['child_column']}"
                ),

            "type":
                "relationship",

            "title":
                (
                    f"{row['parent_table']} -> "
                    f"{row['child_table']}"
                ),

            "search_text":
                (
                    f"{row['parent_table']} {row['parent_column']} "
                    f"{row['child_table']} {row['child_column']} "
                    f"{row['expected_relationship']} "
                    f"{row['purpose']} "
                    f"{row['join_type_recommended']} "
                    f"{row['join_risk']}"
                ),

            "payload":
                row.to_dict()
        }
    )


# quality ---------------------------------------------------------------------

if not quality_registry.empty:

    for _, row in quality_registry.iterrows():

        status = str(
            row.get(
                "status",
                ""
            )
        )

        if status not in {
            "FAIL",
            "WARN",
            "INFO"
        }:
            continue

        search_records.append(
            {
                "id":
                    (
                        f"quality:"
                        f"{row.get('registry_source_gate','')}:"
                        f"{row.get('check_id','')}"
                    ),

                "type":
                    "quality_observation",

                "title":
                    str(
                        row.get(
                            "check_id",
                            ""
                        )
                    ),

                "search_text":
                    " ".join(
                        [
                            str(
                                row.get(
                                    "registry_source_gate",
                                    ""
                                )
                            ),

                            str(
                                row.get(
                                    "table",
                                    ""
                                )
                            ),

                            str(
                                row.get(
                                    "variable",
                                    ""
                                )
                            ),

                            str(
                                row.get(
                                    "dimension",
                                    ""
                                )
                            ),

                            str(
                                row.get(
                                    "status",
                                    ""
                                )
                            ),

                            str(
                                row.get(
                                    "details",
                                    ""
                                )
                            ),

                            str(
                                row.get(
                                    "observed",
                                    ""
                                )
                            )
                        ]
                    ),

                "payload":
                    {
                        k:
                            clean_text(v)
                        for k, v
                        in row.to_dict().items()
                    }
            }
        )


# questions -------------------------------------------------------------------

for _, row in unresolved.iterrows():

    search_records.append(
        {
            "id":
                f"question:{row['question_id']}",

            "type":
                "unresolved_question",

            "title":
                row[
                    "topic"
                ],

            "search_text":
                (
                    f"{row['topic']} "
                    f"{row['question']} "
                    f"{row['current_evidence']} "
                    f"{row['required_action']} "
                    f"{row['blocks']}"
                ),

            "payload":
                row.to_dict()
        }
    )


with (
    META
    /
    "search_knowledge.jsonl"
).open(
    "w",
    encoding="utf-8"
) as f:

    for record in search_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            )
            +
            "\n"
        )


# =============================================================================
# DATA CARD
# =============================================================================

print(
    "[9/9] Gerando Data Card..."
)


orders_profile = (
    table_catalog[
        table_catalog[
            "table"
        ]
        ==
        "orders"
    ]
    .iloc[0]
)


gate_lines = []

for gate_name, info in gate_summaries.items():

    gate_lines.append(
        f"- **{gate_name}:** `{info.get('status','UNKNOWN')}`"
    )


top_quality = ""

if not quality_registry.empty:

    nonpass = quality_registry[
        quality_registry[
            "status"
        ].isin(
            [
                "WARN",
                "FAIL"
            ]
        )
    ].copy()

    if len(
        nonpass
    ):

        rows = []

        for _, r in nonpass.head(
            30
        ).iterrows():

            rows.append(
                (
                    f"- `{r.get('registry_source_gate','')}` / "
                    f"`{r.get('check_id','')}` — "
                    f"{r.get('table','')}.{r.get('variable','')} — "
                    f"**{r.get('status','')}**"
                )
            )

        top_quality = "\n".join(
            rows
        )

    else:

        top_quality = (
            "- Nenhum WARN/FAIL disponível nos scorecards carregados."
        )


open_md = "\n".join(
    [
        (
            f"- **{r['question_id']} — {r['topic']}**: "
            f"{r['question']}"
        )
        for r
        in open_questions
    ]
)


card = f"""# Data Card — Olist Brazilian E-Commerce

## 1. Identificação

**Projeto:** {PROJECT_NAME}

**Dataset:** {DATASET_NAME}

**Identificador da distribuição:** `{DATASET_ID}`

**Arquivos RAW:** {len(FILES)}

**Colunas observadas:** {int(table_catalog["columns"].sum())}

**Pedidos na tabela central:** {int(orders_profile["rows"]):,}

**Fingerprint composto SHA-256:** `{dataset_fingerprint}`


## 2. Origem e natureza

A base utilizada é a distribuição pública da Olist.

O projeto considera os arquivos RAW como **Source Truth do dataset público**:
eles informam o que está registrado na distribuição, mas não fornecem acesso
aos sistemas transacionais privados originais.

Consequentemente:

> internal consistency != independently verified external ground truth


## 3. Uso pretendido

Problema principal:

> No momento da compra, estimar o risco de que o pedido seja entregue depois
> da promessa apresentada ao consumidor.

Instante de decisão candidato:

`{PREDICTION_T0}`

Granularidade futura da tabela de ML:

`1 row = 1 order_id`


## 4. Estrutura

| Tabela | Linhas | Colunas | Grão |
|---|---:|---:|---|
"""

for _, r in table_catalog.iterrows():

    card += (
        f"| {r['table']} | "
        f"{int(r['rows']):,} | "
        f"{int(r['columns'])} | "
        f"{r['grain']} |\n"
    )


card += """

## 5. Relacionamentos

"""

for _, r in relationship_registry.iterrows():

    card += (
        f"- `{r['parent_table']}.{r['parent_column']}` → "
        f"`{r['child_table']}.{r['child_column']}` | "
        f"{r['expected_relationship']} | "
        f"órfãos observados: {int(r['orphan_rows']):,} | "
        f"{r['join_type_recommended']}\n"
    )


card += f"""

## 6. Data Quality Gates

{chr(10).join(gate_lines)}

### Observações não-PASS registradas

{top_quality}


## 7. Verdade e proveniência

O registry distingue:

1. **Source truth** — o valor existe no arquivo público.
2. **Structural truth** — entidades/chaves/relações são estruturalmente válidas.
3. **Semantic truth** — valores respeitam regras de domínio.
4. **Process truth** — o evento é compatível com o estágio operacional.
5. **Measurement truth** — a variável mede adequadamente o fenômeno pretendido.
6. **ML task truth** — a informação seria realmente utilizável em `{PREDICTION_T0}`.
7. **External ground truth** — confirmação contra o mundo/sistema fonte.

O nível 7 não está disponível no dataset público.


## 8. Política de disponibilidade temporal

O arquivo:

`temporal_availability_registry.csv`

é a fonte central para definir quais variáveis podem entrar em modelos
no instante de compra.

Uma coluna existir no CSV final **não implica** que estivesse disponível em t0.


## 9. Missingness

Missing deverá ser classificado por mecanismo operacional, evitando que todos
os casos sejam reduzidos a um único NaN.

Taxonomia planejada:

- `OBSERVED`
- `ATTRIBUTE_MISSING`
- `RELATION_MISSING`
- `NOT_APPLICABLE`
- `NOT_YET_OBSERVED`
- `CENSORED`
- `UNKNOWN_PROVENANCE`


## 10. Label

A definição definitiva de atraso permanece sujeita ao Label Contract.

Alternativas já identificadas:

- timestamp estrito;
- dia-calendário.

A diferença entre essas definições deve ser tratada como questão de
**construct validity**, não como simples detalhe de implementação.


## 11. Leakage

São proibidas como features do baseline em t0 informações claramente futuras,
incluindo:

- entrega ao carrier;
- entrega ao cliente;
- reviews;
- estado final do pedido.

Campos cuja disponibilidade em t0 não foi comprovada permanecem em `HOLD`.


## 12. Limitações conhecidas

- dados anonimizados;
- ausência de acesso aos sistemas operacionais fonte;
- anomalias temporais já documentadas;
- múltiplos itens por pedido;
- múltiplos sellers possíveis;
- múltiplos registros de pagamento;
- geolocation com múltiplas observações por CEP;
- últimos meses sem maturidade adequada do outcome supervisionado;
- availability time não explicitamente fornecido para todas as variáveis.


## 13. Questões abertas

{open_md}


## 14. Artefatos de conhecimento

- `dataset_manifest.json`
- `table_catalog.csv`
- `column_catalog.csv`
- `relationship_registry.csv`
- `truth_provenance_registry.csv`
- `temporal_availability_registry.csv`
- `value_domain_registry.csv`
- `quality_observation_registry.csv`
- `unresolved_questions.csv`
- `search_knowledge.jsonl`


## 15. Regra de manutenção

Este documento deve evoluir junto com o dataset e com as decisões do projeto.

Alterações em:

- target;
- t0;
- disponibilidade das features;
- tratamento;
- definição de entidades;
- conhecimento de proveniência;
- novos problemas de qualidade;

devem ser refletidas nos registries correspondentes.
"""


(
    META
    /
    "DATA_CARD_OLIST.md"
).write_text(
    card,
    encoding="utf-8"
)


# =============================================================================
# BUILD MANIFEST
# =============================================================================

build_manifest = {

    "registry_version":
        "1.0.0",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "project":
        PROJECT_NAME,

    "dataset":
        DATASET_ID,

    "dataset_fingerprint":
        dataset_fingerprint,

    "python_version":
        sys.version,

    "platform":
        platform.platform(),

    "pandas_version":
        pd.__version__,

    "numpy_version":
        np.__version__,

    "tables":
        len(
            table_catalog
        ),

    "columns":
        len(
            column_catalog
        ),

    "relationships":
        len(
            relationship_registry
        ),

    "domain_values":
        len(
            value_domain
        ),

    "quality_observations":
        len(
            quality_registry
        ),

    "open_questions":
        len(
            unresolved
        ),

    "search_records":
        len(
            search_records
        ),

    "raw_modified":
        False
}


safe_json_dump(
    build_manifest,
    META
    /
    "00_registry_build_manifest.json"
)


# =============================================================================
# SUMMARY
# =============================================================================

summary_lines = [

    "=" * 100,

    "DATASET KNOWLEDGE & TRUTH REGISTRY",

    "=" * 100,

    "",

    f"Dataset              : {DATASET_NAME}",

    f"Dataset ID           : {DATASET_ID}",

    f"Tables mapped        : {len(table_catalog)}",

    f"Columns mapped       : {len(column_catalog)}",

    f"Relationships mapped : {len(relationship_registry)}",

    f"Domain values        : {len(value_domain)}",

    f"Quality observations : {len(quality_registry)}",

    f"Open questions       : {len(unresolved)}",

    f"Search records       : {len(search_records)}",

    "",

    "DQ GATES",

    "-" * 100,
]


for gate_name, info in gate_summaries.items():

    summary_lines.append(
        f"{gate_name:<25}: "
        f"{info.get('status','UNKNOWN')}"
    )


summary_lines.extend(
    [
        "",

        "ARTEFATOS",

        "-" * 100,
    ]
)


for p in sorted(
    META.iterdir()
):

    if p.is_file():

        summary_lines.append(
            f"- {p.name}"
        )


summary_lines.extend(
    [
        "",

        "PRINCÍPIOS REGISTRADOS",

        "-" * 100,

        "1. CSV observado != verdade externa comprovada.",

        "2. Missing de relação != missing de atributo.",

        "3. Valor existente no dataset != valor disponível em t0.",

        "4. Feature != target source.",

        "5. Campo futuro não entra no baseline.",

        "6. Questões não comprovadas permanecem OPEN.",

        "7. RAW não foi alterado.",

        "",

        "PRÓXIMA ETAPA RECOMENDADA:",

        "DQ Gate 03B — Conditional & Task Completeness.",

        "",

        "=" * 100,
    ]
)


summary_path = (
    META
    /
    "DATASET_KNOWLEDGE_SUMMARY.txt"
)


summary_path.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8"
)


# =============================================================================
# TERMINAL
# =============================================================================

print()

print(
    "=" * 100
)

print(
    "[OK] DATASET KNOWLEDGE REGISTRY CONSTRUÍDO"
)

print(
    "=" * 100
)

print()

print(
    f"Tabelas mapeadas        : {len(table_catalog)}"
)

print(
    f"Colunas mapeadas        : {len(column_catalog)}"
)

print(
    f"Relacionamentos         : {len(relationship_registry)}"
)

print(
    f"Valores de domínio      : {len(value_domain)}"
)

print(
    f"Observações de qualidade: {len(quality_registry)}"
)

print(
    f"Questões abertas        : {len(unresolved)}"
)

print(
    f"Registros para busca    : {len(search_records)}"
)

print()

print(
    "Arquivos:"
)

for p in sorted(
    META.iterdir()
):

    if p.is_file():

        print(
            f"  - metadata/{p.name}"
        )

print()

print(
    "[OK] Nenhum arquivo em data/raw foi modificado."
)

print(
    f"Resumo principal: {summary_path}"
)

