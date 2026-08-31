#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
CROISSANT 1.1 — MLCROISSANT 1.1.0 COMPATIBILITY LAYER
Delivery Risk Intelligence Platform
===============================================================================

OBJETIVO
--------
Preservar:

1. croissant_1_1_FIXED.json
   -> representação canônica segundo Croissant 1.1

e gerar:

2. croissant_1_1_MLCROISSANT_COMPAT.json
   -> adaptação exclusivamente para compatibilidade com o parser
      mlcroissant 1.1.0

IMPORTANTE
----------
O Croissant 1.1 oficial representa FK como:

    "references": {"@id": "recordset/field"}

O mlcroissant 1.1.0 modela internamente `references` como Source.
Nesta camada de compatibilidade usamos:

    "references": {
        "field": {"@id": "recordset/field"}
    }

Isso NÃO substitui o arquivo canônico.
"""

from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import sys

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

ROOT = (
    PROJECT
    / "metadata"
    / "registry_v1_1"
)

CANONICAL = (
    ROOT
    / "croissant_1_1_FIXED.json"
)

COMPAT = (
    ROOT
    / "croissant_1_1_MLCROISSANT_COMPAT.json"
)

MAP_OUT = (
    ROOT
    / "croissant_mlcompat_reference_map.json"
)

STATUS_OUT = (
    ROOT
    / "croissant_interoperability_status.json"
)


if not CANONICAL.exists():

    raise SystemExit(
        f"[ERRO] Arquivo canônico não encontrado: {CANONICAL}"
    )


# ============================================================================
# LOAD
# ============================================================================

with CANONICAL.open(
    "r",
    encoding="utf-8"
) as f:

    canonical = json.load(f)


compat = copy.deepcopy(
    canonical
)


# ============================================================================
# METADATA DE ATRIBUIÇÃO
#
# Não inventamos datePublished exato.
# A data pública exata permanece uma questão de proveniência.
# ============================================================================

compat.setdefault(
    "creator",
    {
        "@type": "sc:Organization",
        "name": "Olist"
    }
)


compat.setdefault(
    "publisher",
    {
        "@type": "sc:Organization",
        "name": "Kaggle"
    }
)


compat.setdefault(
    "citeAs",
    (
        "@misc{olist_brazilian_ecommerce,"
        " author={Olist},"
        " title={Brazilian E-Commerce Public Dataset by Olist},"
        " publisher={Kaggle},"
        " url={https://www.kaggle.com/datasets/"
        "olistbr/brazilian-ecommerce}"
        "}"
    )
)


compat[
    "sdVersion"
] = "registry-1.1.0"


# ============================================================================
# CONVERTER REFERENCES SOMENTE NA VERSÃO COMPAT
# ============================================================================

mapping = []


recordsets = compat.get(
    "recordSet",
    []
)


for rs in recordsets:

    rsid = rs.get(
        "@id",
        ""
    )

    for field in rs.get(
        "field",
        []
    ):

        source_field = field.get(
            "@id",
            ""
        )

        ref = field.get(
            "references"
        )

        if ref is None:
            continue


        # ---------------------------------------------------------------
        # Forma canônica:
        #
        # references:
        #     {"@id": "orders/order_id"}
        #
        # ---------------------------------------------------------------

        if (
            isinstance(
                ref,
                dict
            )
            and
            "@id" in ref
        ):

            target = ref[
                "@id"
            ]


            mapping.append(
                {
                    "recordset":
                        rsid,

                    "source_field":
                        source_field,

                    "canonical_reference":
                        {
                            "@id":
                                target
                        },

                    "mlcroissant_compat_reference":
                        {
                            "field": {
                                "@id":
                                    target
                            }
                        }
                }
            )


            field[
                "references"
            ] = {
                "field": {
                    "@id":
                        target
                }
            }


# ============================================================================
# VALIDAR QUANTIDADE
# ============================================================================

expected_references = 7


if len(
    mapping
) != expected_references:

    raise SystemExit(
        "[ERRO] Esperávamos converter "
        f"{expected_references} references, "
        f"mas encontramos {len(mapping)}."
    )


# ============================================================================
# GARANTIR QUE O CANÔNICO NÃO FOI ALTERADO
# ============================================================================

with CANONICAL.open(
    "r",
    encoding="utf-8"
) as f:

    canonical_after = json.load(f)


if (
    canonical
    !=
    canonical_after
):

    raise SystemExit(
        "[ERRO] O arquivo Croissant canônico foi alterado."
    )


# ============================================================================
# SAVE COMPAT
# ============================================================================

with COMPAT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        compat,
        f,
        ensure_ascii=False,
        indent=2
    )


with MAP_OUT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        mapping,
        f,
        ensure_ascii=False,
        indent=2
    )


status = {

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "canonical_file":
        str(
            CANONICAL
        ),

    "compat_file":
        str(
            COMPAT
        ),

    "croissant_spec":
        "1.1",

    "mlcroissant_version_target":
        "1.1.0",

    "canonical_reference_syntax":
        'references: {"@id": "recordset/field"}',

    "compat_reference_syntax":
        (
            'references: '
            '{"field": {"@id": "recordset/field"}}'
        ),

    "references_converted":
        len(
            mapping
        ),

    "expected_references":
        expected_references,

    "canonical_preserved":
        True,

    "raw_modified":
        False,

    "interpretation":
        (
            "Compatibility representation for current mlcroissant parser; "
            "canonical Croissant metadata remains unchanged."
        )
}


with STATUS_OUT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        status,
        f,
        ensure_ascii=False,
        indent=2
    )


print("=" * 100)

print(
    "CROISSANT 1.1 — MLCROISSANT COMPATIBILITY BUILD"
)

print("=" * 100)

print()

print(
    f"References convertidas : {len(mapping)}"
)

print(
    f"Esperadas              : {expected_references}"
)

print(
    "Canonical preservado   : SIM"
)

print(
    "RAW modificado         : NÃO"
)

print()

print(
    f"Canonical : {CANONICAL}"
)

print(
    f"Compat    : {COMPAT}"
)

print(
    f"Map       : {MAP_OUT}"
)

print(
    f"Status    : {STATUS_OUT}"
)

