#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import copy
import sys

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

ROOT = (
    PROJECT
    / "metadata"
    / "registry_v1_1"
)

SRC = ROOT / "croissant_1_1.json"

OUT_FULL = ROOT / "croissant_1_1_FIXED.json"
OUT_NO_FK = ROOT / "croissant_1_1_FIXED_NO_REFERENCES.json"
AUDIT = ROOT / "croissant_reference_audit.csv"

if not SRC.exists():
    raise SystemExit(f"[ERRO] Não encontrado: {SRC}")


# =====================================================================
# LOAD
# =====================================================================

with SRC.open("r", encoding="utf-8") as f:
    data = json.load(f)


# =====================================================================
# 1. CONTEXTO CROISSANT
#
# Corrige exatamente os aliases ausentes apontados pelo validator.
# Mantém os aliases que já existiam.
# =====================================================================

ctx = data.setdefault("@context", {})

required_context = {

    "@language":
        "en",

    "@vocab":
        "https://schema.org/",

    "sc":
        "https://schema.org/",

    "cr":
        "http://mlcommons.org/croissant/",

    "rai":
        "http://mlcommons.org/croissant/RAI/",

    "dct":
        "http://purl.org/dc/terms/",

    "citeAs":
        "cr:citeAs",

    "column":
        "cr:column",

    "conformsTo":
        "dct:conformsTo",

    "data": {
        "@id":
            "cr:data",

        "@type":
            "@json"
    },

    "dataType": {
        "@id":
            "cr:dataType",

        "@type":
            "@vocab"
    },

    "equivalentProperty":
        "cr:equivalentProperty",

    "examples": {
        "@id":
            "cr:examples",

        "@type":
            "@json"
    },

    "extract":
        "cr:extract",

    "field":
        "cr:field",

    "fileObject":
        "cr:fileObject",

    "fileProperty":
        "cr:fileProperty",

    "fileSet":
        "cr:fileSet",

    "format":
        "cr:format",

    "includes":
        "cr:includes",

    "isLiveDataset":
        "cr:isLiveDataset",

    "jsonPath":
        "cr:jsonPath",

    "key":
        "cr:key",

    "md5":
        "cr:md5",

    "parentField":
        "cr:parentField",

    "path":
        "cr:path",

    "recordSet":
        "cr:recordSet",

    "references":
        "cr:references",

    "regex":
        "cr:regex",

    "repeated":
        "cr:repeated",

    "replace":
        "cr:replace",

    "samplingRate":
        "cr:samplingRate",

    "separator":
        "cr:separator",

    "source":
        "cr:source",

    "subField":
        "cr:subField",

    "transform":
        "cr:transform"
}


# Reconstruir em ordem conhecida.
data["@context"] = required_context


# =====================================================================
# 2. CONFORMS TO
# =====================================================================

data["conformsTo"] = (
    "http://mlcommons.org/croissant/1.1"
)


# =====================================================================
# 3. INDEXAR TODOS OS RECORDSETS E FIELDS
# =====================================================================

recordsets = data.get(
    "recordSet",
    []
)

if not isinstance(recordsets, list):
    raise SystemExit(
        "[ERRO] recordSet deveria ser uma lista."
    )


recordset_ids = set()

field_ids = set()

field_locations = {}


for rs in recordsets:

    rsid = rs.get("@id")

    if not rsid:

        raise SystemExit(
            "[ERRO] RecordSet sem @id."
        )

    if rsid in recordset_ids:

        raise SystemExit(
            f"[ERRO] RecordSet duplicado: {rsid}"
        )

    recordset_ids.add(
        rsid
    )

    fields = rs.get(
        "field",
        []
    )

    if not isinstance(
        fields,
        list
    ):

        raise SystemExit(
            f"[ERRO] field não é lista em {rsid}"
        )

    for field in fields:

        fid = field.get("@id")

        if not fid:

            raise SystemExit(
                f"[ERRO] Field sem @id em {rsid}"
            )

        if fid in field_ids:

            raise SystemExit(
                f"[ERRO] Field duplicado: {fid}"
            )

        field_ids.add(
            fid
        )

        field_locations[
            fid
        ] = rsid


# =====================================================================
# 4. AUDITAR KEYS
# =====================================================================

key_rows = []

for rs in recordsets:

    rsid = rs["@id"]

    keys = rs.get(
        "key",
        []
    )

    if isinstance(
        keys,
        dict
    ):
        keys = [keys]

    if keys is None:
        keys = []

    for key in keys:

        kid = (
            key.get("@id")
            if isinstance(key, dict)
            else None
        )

        exists = (
            kid in field_ids
        )

        key_rows.append(
            {
                "kind":
                    "KEY",

                "source_recordset":
                    rsid,

                "source_field":
                    "",

                "target_field":
                    kid,

                "target_exists":
                    exists,

                "target_recordset":
                    (
                        field_locations.get(
                            kid,
                            ""
                        )
                    )
            }
        )


# =====================================================================
# 5. AUDITAR REFERENCES
# =====================================================================

reference_rows = []


for rs in recordsets:

    rsid = rs["@id"]

    for field in rs.get(
        "field",
        []
    ):

        fid = field["@id"]

        if "references" not in field:
            continue

        ref = field[
            "references"
        ]

        # O spec usa Reference via {"@id": "..."}.
        if not isinstance(
            ref,
            dict
        ):

            reference_rows.append(
                {
                    "kind":
                        "REFERENCE",

                    "source_recordset":
                        rsid,

                    "source_field":
                        fid,

                    "target_field":
                        "",

                    "target_exists":
                        False,

                    "target_recordset":
                        "",

                    "error":
                        (
                            "references is not an object"
                        )
                }
            )

            continue

        target = ref.get(
            "@id"
        )

        exists = (
            target in field_ids
        )

        reference_rows.append(
            {
                "kind":
                    "REFERENCE",

                "source_recordset":
                    rsid,

                "source_field":
                    fid,

                "target_field":
                    target,

                "target_exists":
                    exists,

                "target_recordset":
                    field_locations.get(
                        target,
                        ""
                    ),

                "error":
                    (
                        ""
                        if exists
                        else
                        "referenced Field does not exist"
                    )
            }
        )


# =====================================================================
# 6. SALVAR AUDITORIA
# =====================================================================

import pandas as pd

audit_df = pd.DataFrame(
    key_rows
    +
    reference_rows
)

audit_df.to_csv(
    AUDIT,
    index=False
)


bad = audit_df[
    audit_df[
        "target_exists"
    ]
    ==
    False
]


print("=" * 100)
print("CROISSANT — AUDITORIA DE KEYS / REFERENCES")
print("=" * 100)

print(
    f"RecordSets             : {len(recordset_ids)}"
)

print(
    f"Fields                 : {len(field_ids)}"
)

print(
    f"Keys auditadas         : {len(key_rows)}"
)

print(
    f"References auditadas   : {len(reference_rows)}"
)

print(
    f"Targets inválidos      : {len(bad)}"
)

if len(bad):

    print()
    print("REFERÊNCIAS/KEYS INVÁLIDAS:")
    print(
        bad.to_string(
            index=False
        )
    )

    raise SystemExit(
        2
    )


# =====================================================================
# 7. VERIFICAR QUE REFERÊNCIAS APONTAM PARA FIELDS
# =====================================================================

cross_rs = []

for row in reference_rows:

    source_rs = row[
        "source_recordset"
    ]

    target_rs = row[
        "target_recordset"
    ]

    cross_rs.append(
        {
            **row,

            "cross_recordset":
                source_rs
                !=
                target_rs
        }
    )


cross_df = pd.DataFrame(
    cross_rs
)

cross_df.to_csv(
    ROOT
    /
    "croissant_foreign_key_map.csv",
    index=False
)


# =====================================================================
# 8. SALVAR VERSÃO COMPLETA
# =====================================================================

with OUT_FULL.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )


# =====================================================================
# 9. SALVAR VERSÃO DIAGNÓSTICA SEM REFERENCES
#
# Se esta passar e a completa não, isolamos o problema no parser
# das relações Croissant/mlcroissant.
# =====================================================================

no_fk = copy.deepcopy(
    data
)


for rs in no_fk.get(
    "recordSet",
    []
):

    for field in rs.get(
        "field",
        []
    ):

        field.pop(
            "references",
            None
        )


with OUT_NO_FK.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        no_fk,
        f,
        ensure_ascii=False,
        indent=2
    )


# =====================================================================
# 10. RESUMO
# =====================================================================

print()
print("[PASS] Todas as keys/references apontam para Fields existentes.")

print()

print("Arquivos criados:")

print(
    f"  - {OUT_FULL}"
)

print(
    f"  - {OUT_NO_FK}"
)

print(
    f"  - {AUDIT}"
)

print(
    f"  - {ROOT / 'croissant_foreign_key_map.csv'}"
)

print()

print(
    "[OK] Nenhum arquivo RAW foi alterado."
)

