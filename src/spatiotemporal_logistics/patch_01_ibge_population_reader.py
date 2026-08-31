#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

TARGET = Path(
    "src/spatiotemporal_logistics/"
    "01_build_fast_spatiotemporal_core.py"
)

text = TARGET.read_text(encoding="utf-8")

START_MARKER = "pop_parts = []"
END_MARKER = "population = pd.concat("

start = text.find(START_MARKER)
end = text.find(END_MARKER, start)

if start < 0:
    raise SystemExit(
        "[FAIL] marcador inicial pop_parts não encontrado."
    )

if end < 0:
    raise SystemExit(
        "[FAIL] marcador population = pd.concat não encontrado."
    )

replacement = r'''# ---------------------------------------------------------------------
# IBGE POPULATION READER — ROBUST STRUCTURAL PARSER
#
# Os XLS históricos do IBGE possuem linhas de título/notas e podem
# apresentar cabeçalhos formatados/mesclados.
#
# Em vez de depender da posição física do header, identificamos
# diretamente as linhas municipais pela estrutura oficial:
#
# UF | COD. UF | COD. MUNIC | NOME DO MUNICÍPIO | POPULAÇÃO
#
# Isso preserva o arquivo original e evita tratamento manual.
# ---------------------------------------------------------------------

def parse_ibge_population_value(value):

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (int, np.integer)
    ):
        value = int(value)

        return (
            value
            if value > 0
            else None
        )

    if isinstance(
        value,
        (float, np.floating)
    ):

        if not np.isfinite(value):
            return None

        value = int(round(value))

        return (
            value
            if value > 0
            else None
        )

    s = str(value).strip()

    # Ex.: 12.345(1)
    s = re.sub(
        r"\([^)]*\)",
        "",
        s
    )

    digits = re.sub(
        r"\D",
        "",
        s
    )

    if not digits:
        return None

    value = int(digits)

    return (
        value
        if value > 0
        else None
    )


def read_ibge_population_structural(
    path,
    year
):

    excel = pd.ExcelFile(path)

    records = []
    sheet_stats = []

    for sheet in excel.sheet_names:

        raw = pd.read_excel(
            excel,
            sheet_name=sheet,
            header=None,
            dtype=object
        )

        before = len(records)

        for row in raw.itertuples(
            index=False,
            name=None
        ):

            values = [
                value
                for value in row
                if (
                    pd.notna(value)
                    and
                    str(value).strip()
                )
            ]

            if len(values) < 5:
                continue

            # Procura qualquer janela consecutiva de 5 valores úteis.
            # Isso também tolera colunas vazias/formatativas no XLS.
            for pos in range(
                0,
                len(values) - 4
            ):

                (
                    uf_raw,
                    coduf_raw,
                    codmun_raw,
                    city_raw,
                    population_raw,
                ) = values[
                    pos:pos + 5
                ]

                uf = norm_text(
                    uf_raw
                )

                if not re.fullmatch(
                    r"[A-Z]{2}",
                    uf
                ):
                    continue

                cod_uf = clean_code(
                    coduf_raw,
                    2
                )

                cod_mun = clean_code(
                    codmun_raw,
                    5
                )

                if (
                    cod_uf is None
                    or
                    cod_mun is None
                ):
                    continue

                city = str(
                    city_raw
                ).strip()

                if (
                    not city
                    or
                    norm_text(city)
                    in {
                        "",
                        "NOME DO MUNICIPIO",
                        "MUNICIPIO",
                    }
                ):
                    continue

                population_value = (
                    parse_ibge_population_value(
                        population_raw
                    )
                )

                if population_value is None:
                    continue

                records.append(
                    {
                        "year":
                            int(year),

                        "uf":
                            uf,

                        "city":
                            city,

                        "city_norm":
                            norm_text(city),

                        "cod_uf":
                            cod_uf,

                        "cod_mun":
                            cod_mun,

                        "id_municipio":
                            cod_uf + cod_mun,

                        "population":
                            population_value,

                        "source_sheet":
                            str(sheet),
                    }
                )

                break

        sheet_stats.append(
            (
                str(sheet),
                len(records) - before
            )
        )

    if not records:

        raise RuntimeError(
            f"Nenhuma linha municipal reconhecida em {path}"
        )

    df_pop = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------
    # Remover somente duplicação física da mesma unidade
    # eventualmente causada por páginas/blocos repetidos no XLS.
    # -------------------------------------------------------------

    df_pop = (

        df_pop.sort_values(
            [
                "id_municipio",
                "city_norm",
            ]
        )

        .drop_duplicates(
            subset=[
                "year",
                "id_municipio",
            ],
            keep="first"
        )

        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Validações estruturais
    # -------------------------------------------------------------

    duplicate_codes = int(
        df_pop[
            "id_municipio"
        ].duplicated().sum()
    )

    bad_codes = int(
        ~df_pop[
            "id_municipio"
        ].astype(str).str.fullmatch(
            r"\d{7}"
        )
    .sum()
    ) if False else 0

    # forma explícita para evitar qualquer ambiguidade
    bad_code_mask = (
        ~df_pop[
            "id_municipio"
        ]
        .astype(str)
        .str.fullmatch(
            r"\d{7}"
        )
    )

    bad_codes = int(
        bad_code_mask.sum()
    )

    bad_population = int(
        (
            pd.to_numeric(
                df_pop[
                    "population"
                ],
                errors="coerce"
            )
            <= 0
        ).sum()
    )

    print(
        f"[INFO] IBGE {year} "
        f"sheets={sheet_stats}"
    )

    print(
        f"[INFO] IBGE {year} "
        f"linhas estruturais únicas="
        f"{len(df_pop):,}"
    )

    print(
        f"[INFO] IBGE {year} "
        f"duplicate_codes="
        f"{duplicate_codes}"
    )

    print(
        f"[INFO] IBGE {year} "
        f"bad_codes="
        f"{bad_codes}"
    )

    print(
        f"[INFO] IBGE {year} "
        f"bad_population="
        f"{bad_population}"
    )

    if duplicate_codes != 0:
        raise RuntimeError(
            f"IBGE {year}: códigos municipais duplicados."
        )

    if bad_codes != 0:
        raise RuntimeError(
            f"IBGE {year}: código municipal inválido."
        )

    if bad_population != 0:
        raise RuntimeError(
            f"IBGE {year}: população inválida."
        )

    # -------------------------------------------------------------
    # Sanity check.
    #
    # A divulgação oficial possui aproximadamente 5.570 unidades.
    # Usamos intervalo estreito para detectar parsing incorreto,
    # sem esconder eventual particularidade histórica.
    # -------------------------------------------------------------

    if not (
        5550
        <= len(df_pop)
        <= 5590
    ):
        raise RuntimeError(
            f"IBGE {year}: quantidade inesperada "
            f"de unidades: {len(df_pop):,}. "
            "Esperado aproximadamente 5.570."
        )

    return df_pop


pop_parts = []


for year, path in pop_files.items():

    if not path.exists():

        raise SystemExit(
            f"[FAIL] população ausente: {path}"
        )

    q = read_ibge_population_structural(
        path,
        year
    )

    pop_parts.append(
        q
    )

    print(
        f"[PASS] população {year}: "
        f"{len(q):,} unidades"
    )


'''

new_text = (
    text[:start]
    +
    replacement
    +
    text[end:]
)

TARGET.write_text(
    new_text,
    encoding="utf-8"
)

print(
    "[PASS] PATCH_IBGE_POPULATION_APPLIED"
)

print(
    f"[PASS] target={TARGET}"
)

print(
    "RAW_MODIFIED = false"
)
