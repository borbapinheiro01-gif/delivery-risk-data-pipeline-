#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import tempfile
import unicodedata
import zipfile

import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
LANDING = ROOT / "data" / "external" / "landing"
ART = ROOT / "artifacts" / "spatiotemporal_logistics"
REP = ROOT / "reports" / "spatiotemporal_logistics"

YEARS = {2017, 2018}


def norm_text(value):
    if pd.isna(value):
        return ""

    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )

    s = s.upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_code(value, width):
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        s = str(int(value))

    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            s = str(int(value))
        else:
            return None

    else:
        s = str(value).strip()

        if re.fullmatch(r"-?\d+\.0+", s):
            s = s.split(".")[0]

        s = re.sub(r"\D", "", s)

    if not s:
        return None

    if len(s) > width:
        return None

    return s.zfill(width)


def parse_population(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return np.nan
        return int(round(float(value)))

    s = str(value).strip()

    if re.fullmatch(r"\d+\.0+", s):
        return int(s.split(".")[0])

    s = re.sub(r"\([^)]*\)", "", s)
    digits = re.sub(r"\D", "", s)

    return int(digits) if digits else np.nan


print("=" * 110)
print("STAGE 02 — EXTERNAL CONTEXT")
print("=" * 110)


# --------------------------------------------------------------------------------------------------
# 1. POPULAÇÃO
# --------------------------------------------------------------------------------------------------

pop_parts = []

for year, path in {
    2017:
        LANDING
        / "ibge_population"
        / "2017"
        / "estimativa_dou_2017.xls",

    2018:
        LANDING
        / "ibge_population"
        / "2018"
        / "estimativa_dou_2018_20181019.xls",
}.items():

    raw = pd.read_excel(
        path,
        sheet_name="Municípios",
        header=1,
        dtype=object,
    )

    raw.columns = [
        str(c).strip()
        for c in raw.columns
    ]

    # IMPORTANT:
    # Build the indexed columns first.
    # Assigning scalar year to an empty DataFrame would create
    # an empty year column and later yield NaN after index alignment.

    q = pd.DataFrame(
        {
            "uf": (
                raw["UF"]
                .astype("string")
                .str.upper()
                .str.strip()
            ),

            "city_norm": (
                raw["NOME DO MUNICÍPIO"]
                .map(norm_text)
            ),

            "cod_uf": (
                raw["COD. UF"]
                .map(lambda x: clean_code(x, 2))
            ),

            "cod_mun": (
                raw["COD. MUNIC"]
                .map(lambda x: clean_code(x, 5))
            ),
        }
    )

    q["year"] = int(year)

    q["id_municipio"] = np.where(
        q["cod_uf"].notna()
        &
        q["cod_mun"].notna(),
        q["cod_uf"] + q["cod_mun"],
        None,
    )

    q["population"] = (
        raw["POPULAÇÃO ESTIMADA"]
        .map(parse_population)
    )

    q = q[
        q["uf"].str.fullmatch(r"[A-Z]{2}", na=False)
        &
        q["id_municipio"].astype("string").str.fullmatch(r"\d{7}", na=False)
        &
        q["population"].gt(0)
    ].copy()

    q = q[
        [
            "year",
            "uf",
            "city_norm",
            "id_municipio",
            "population",
        ]
    ].drop_duplicates(
        ["year", "id_municipio"],
        keep="first",
    )

    if len(q) != 5570:
        raise RuntimeError(
            f"IBGE {year}: esperado 5570 municípios válidos; obtido {len(q)}"
        )

    q["log_population"] = np.log1p(
        q["population"]
    )

    # Strong uniqueness / year gates before concatenation.
    if q["year"].isna().any():
        raise RuntimeError(
            f"IBGE {year}: year contém NaN."
        )

    if not q["year"].eq(year).all():
        raise RuntimeError(
            f"IBGE {year}: year inconsistente."
        )

    if q.duplicated(
        ["year", "id_municipio"]
    ).any():
        dup = q.loc[
            q.duplicated(
                ["year", "id_municipio"],
                keep=False
            ),
            [
                "year",
                "uf",
                "city_norm",
                "id_municipio",
            ]
        ].head(20)

        raise RuntimeError(
            f"IBGE {year}: duplicatas município-ano:\n"
            f"{dup.to_string(index=False)}"
        )

    pop_parts.append(q)

    print(
        f"[PASS] população {year}: "
        f"{len(q):,} municípios | "
        f"year_unique={sorted(q['year'].unique().tolist())}"
    )

population = pd.concat(
    pop_parts,
    ignore_index=True,
)

print(
    "[INFO] population rows by year:",
    population.groupby("year").size().to_dict()
)

print(
    "[INFO] population year NA:",
    int(population["year"].isna().sum())
)

print(
    "[INFO] population duplicate municipality-year:",
    int(
        population.duplicated(
            ["year", "id_municipio"]
        ).sum()
    )
)

if population["year"].isna().any():
    raise RuntimeError(
        "Population consolidated: year contém NaN."
    )

if set(population["year"].unique()) != {2017, 2018}:
    raise RuntimeError(
        "Population consolidated: anos inesperados: "
        f"{sorted(population['year'].unique().tolist())}"
    )

if population.duplicated(
    ["year", "id_municipio"]
).any():
    raise RuntimeError(
        "Population consolidated: município-ano duplicado."
    )

if len(population) != 11140:
    raise RuntimeError(
        "Population consolidated: esperado 11.140 linhas "
        f"(5570 x 2), observado {len(population)}."
    )

print(
    "[PASS] POPULATION CONSOLIDATED | "
    "2017=5570 | 2018=5570 | total=11140 | duplicates=0"
)


# --------------------------------------------------------------------------------------------------
# 2. PIB
# --------------------------------------------------------------------------------------------------

gdp_zip = (
    LANDING
    / "ibge_gdp"
    / "retrospective"
    / "base_de_dados_2010_2018_xls.zip"
)

with zipfile.ZipFile(gdp_zip) as z:

    members = [
        x
        for x in z.namelist()
        if x.lower().endswith((".xls", ".xlsx"))
        and not x.startswith("__MACOSX")
    ]

    if len(members) != 1:
        raise RuntimeError(
            f"PIB: workbook inesperado: {members}"
        )

    with tempfile.TemporaryDirectory() as td:

        z.extract(members[0], td)

        gdp_raw = pd.read_excel(
            Path(td) / members[0],
            sheet_name="PIB_dos_Municípios",
            header=0,
        )


def normalized_col(col):
    return norm_text(col)


colmap = {
    normalized_col(c): c
    for c in gdp_raw.columns
}

year_col = colmap["ANO"]
id_col = colmap["CODIGO DO MUNICIPIO"]

pc_candidates = [
    original
    for norm, original in colmap.items()
    if (
        "PRODUTO INTERNO BRUTO PER CAPITA" in norm
    )
]

total_candidates = [
    original
    for norm, original in colmap.items()
    if (
        "PRODUTO INTERNO BRUTO" in norm
        and
        "PER CAPITA" not in norm
        and
        "PERCENTUAL" not in norm
    )
]

if not pc_candidates:
    raise RuntimeError("PIB per capita não localizado.")

if not total_candidates:
    raise RuntimeError("PIB total não localizado.")

gdp = pd.DataFrame()

gdp["year"] = pd.to_numeric(
    gdp_raw[year_col],
    errors="coerce",
).astype("Int64")

gdp["id_municipio"] = (
    gdp_raw[id_col]
    .map(lambda x: clean_code(x, 7))
)

gdp["gdp_current"] = pd.to_numeric(
    gdp_raw[total_candidates[0]],
    errors="coerce",
)

gdp["gdp_per_capita"] = pd.to_numeric(
    gdp_raw[pc_candidates[0]],
    errors="coerce",
)

gdp = gdp[
    gdp["year"].isin(YEARS)
    &
    gdp["id_municipio"].notna()
].copy()

gdp = gdp.drop_duplicates(
    ["year", "id_municipio"],
    keep="first",
)

if gdp.duplicated(
    ["year", "id_municipio"]
).any():
    raise RuntimeError(
        "PIB município-ano duplicado."
    )

gdp["log_gdp_per_capita"] = np.log1p(
    gdp["gdp_per_capita"].where(
        gdp["gdp_per_capita"] >= 0
    )
)

muni = population.merge(
    gdp,
    on=[
        "year",
        "id_municipio",
    ],
    how="left",
    validate="1:1",
)

muni.to_csv(
    ART / "03_MUNICIPAL_CONTEXT.csv",
    index=False,
)

print(f"[PASS] PIB 2017-2018: {len(gdp):,} município-ano")


# --------------------------------------------------------------------------------------------------
# 3. ANP
# --------------------------------------------------------------------------------------------------

STATE_NAME_TO_UF = {
    "ACRE": "AC",
    "ALAGOAS": "AL",
    "AMAPA": "AP",
    "AMAZONAS": "AM",
    "BAHIA": "BA",
    "CEARA": "CE",
    "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES",
    "GOIAS": "GO",
    "MARANHAO": "MA",
    "MATO GROSSO": "MT",
    "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG",
    "PARA": "PA",
    "PARAIBA": "PB",
    "PARANA": "PR",
    "PERNAMBUCO": "PE",
    "PIAUI": "PI",
    "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS",
    "RONDONIA": "RO",
    "RORAIMA": "RR",
    "SANTA CATARINA": "SC",
    "SAO PAULO": "SP",
    "SERGIPE": "SE",
    "TOCANTINS": "TO",
}

anp_path = (
    LANDING
    / "anp_fuel"
    / "monthly_municipality_2016_2018"
    / "mensal_municipios_2016_2018.xlsx"
)

anp = pd.read_excel(
    anp_path,
    sheet_name="MUNICÍPIOS - JAN 16 - DEZ 18 ",
    header=16,
    usecols=[
        "MÊS",
        "PRODUTO",
        "ESTADO",
        "MUNICÍPIO",
        "NÚMERO DE POSTOS PESQUISADOS",
        "PREÇO MÉDIO REVENDA",
    ],
)

date = pd.to_datetime(
    anp["MÊS"],
    errors="coerce",
)

anp["year"] = date.dt.year
anp["month"] = date.dt.month

anp["uf"] = (
    anp["ESTADO"]
    .map(norm_text)
    .map(STATE_NAME_TO_UF)
)

anp["city_norm"] = (
    anp["MUNICÍPIO"]
    .map(norm_text)
)

anp["product_norm"] = (
    anp["PRODUTO"]
    .map(norm_text)
)

anp["price_mean"] = pd.to_numeric(
    anp["PREÇO MÉDIO REVENDA"],
    errors="coerce",
)

anp["n_posts"] = pd.to_numeric(
    anp["NÚMERO DE POSTOS PESQUISADOS"],
    errors="coerce",
)

anp = anp[
    anp["year"].isin(YEARS)
    &
    anp["product_norm"].str.contains(
        r"\bDIESEL\b",
        regex=True,
        na=False,
    )
].copy()

anp["diesel_type"] = np.where(
    anp["product_norm"].str.contains(
        "S10",
        na=False,
    ),
    "DIESEL_S10",
    "DIESEL_COMMON",
)

anp = anp[
    anp["price_mean"].gt(0)
    &
    anp["uf"].notna()
    &
    anp["city_norm"].ne("")
].copy()

grouped = (
    anp.groupby(
        [
            "year",
            "month",
            "uf",
            "city_norm",
            "diesel_type",
        ],
        as_index=False,
    )
    .agg(
        price_mean=("price_mean", "mean"),
        n_posts=("n_posts", lambda x: x.sum(min_count=1)),
    )
)

price_wide = grouped.pivot(
    index=[
        "year",
        "month",
        "uf",
        "city_norm",
    ],
    columns="diesel_type",
    values="price_mean",
).reset_index()

price_wide.columns.name = None

price_wide = price_wide.rename(
    columns={
        "DIESEL_COMMON":
            "diesel_common_mean",
        "DIESEL_S10":
            "diesel_s10_mean",
    }
)

posts_wide = grouped.pivot(
    index=[
        "year",
        "month",
        "uf",
        "city_norm",
    ],
    columns="diesel_type",
    values="n_posts",
).reset_index()

posts_wide.columns.name = None

posts_wide = posts_wide.rename(
    columns={
        "DIESEL_COMMON":
            "diesel_common_n_posts",
        "DIESEL_S10":
            "diesel_s10_n_posts",
    }
)

anp_muni = price_wide.merge(
    posts_wide,
    on=[
        "year",
        "month",
        "uf",
        "city_norm",
    ],
    how="outer",
    validate="1:1",
)

anp_muni.to_csv(
    ART / "04_ANP_CONTEXT.csv",
    index=False,
)


# --------------------------------------------------------------------------------------------------
# 4. STATE ANP — weighted by number of surveyed stations where possible
# --------------------------------------------------------------------------------------------------

anp["weighted_price_num"] = (
    anp["price_mean"]
    *
    anp["n_posts"]
)

state_long = (
    anp.groupby(
        [
            "year",
            "month",
            "uf",
            "diesel_type",
        ],
        as_index=False,
    )
    .agg(
        weighted_num=("weighted_price_num", lambda x: x.sum(min_count=1)),
        n_posts=("n_posts", lambda x: x.sum(min_count=1)),
        simple_mean=("price_mean", "mean"),
    )
)

state_long["state_price_mean"] = np.where(
    state_long["n_posts"].gt(0),
    (
        state_long["weighted_num"]
        /
        state_long["n_posts"]
    ),
    state_long["simple_mean"],
)

state_price = state_long.pivot(
    index=["year", "month", "uf"],
    columns="diesel_type",
    values="state_price_mean",
).reset_index()

state_price.columns.name = None

state_price = state_price.rename(
    columns={
        "DIESEL_COMMON":
            "diesel_common_state_mean",
        "DIESEL_S10":
            "diesel_s10_state_mean",
    }
)

state_posts = state_long.pivot(
    index=["year", "month", "uf"],
    columns="diesel_type",
    values="n_posts",
).reset_index()

state_posts.columns.name = None

state_posts = state_posts.rename(
    columns={
        "DIESEL_COMMON":
            "diesel_common_state_n_posts",
        "DIESEL_S10":
            "diesel_s10_state_n_posts",
    }
)

anp_state = state_price.merge(
    state_posts,
    on=["year", "month", "uf"],
    how="outer",
    validate="1:1",
)

anp_state.to_csv(
    ART / "04b_ANP_STATE_CONTEXT.csv",
    index=False,
)

print(f"[PASS] ANP municipal: {len(anp_muni):,}")
print(f"[PASS] ANP estadual : {len(anp_state):,}")
print("[PASS 02] EXTERNAL CONTEXT COMPLETE")
