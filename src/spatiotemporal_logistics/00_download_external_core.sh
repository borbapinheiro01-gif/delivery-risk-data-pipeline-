#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/workspace/Delivery_Risk_Intelligence"
cd "$ROOT"

MANIFEST="data/external/registry/download_manifest.tsv"

HTTPS="https:"
IBGE_HOST="//ftp.ibge.gov.br"
ANP_HOST="//www.gov.br"
MJ_HOST="//dados.mj.gov.br"

ACCESS_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo "======================================================================"
echo "SPATIOTEMPORAL LOGISTICS — OFFICIAL EXTERNAL DATA ACQUISITION"
echo "======================================================================"
echo "UTC: $ACCESS_UTC"
echo

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERRO: manifest não existe."
    echo "Rode primeiro create_spatiotemporal_logistics_pipeline.sh"
    exit 1
fi

download_file () {

    local source_id="$1"
    local url="$2"
    local dest="$3"

    mkdir -p "$(dirname "$dest")"

    echo
    echo "---------------------------------------------------------------------"
    echo "SOURCE : $source_id"
    echo "DEST   : $dest"
    echo "---------------------------------------------------------------------"

    if [[ -s "$dest" ]]; then
        echo "[EXISTS] arquivo já existe; não será sobrescrito."
    else
        rm -f "${dest}.part"

        curl \
            --fail \
            --location \
            --retry 4 \
            --retry-delay 3 \
            --connect-timeout 30 \
            --output "${dest}.part" \
            "$url"

        if [[ ! -s "${dest}.part" ]]; then
            echo "ERRO: download vazio para $source_id"
            rm -f "${dest}.part"
            exit 1
        fi

        mv "${dest}.part" "$dest"

        echo "[OK] download concluído."
    fi

    local bytes
    local sha
    local when

    bytes="$(stat -c '%s' "$dest")"
    sha="$(sha256sum "$dest" | awk '{print $1}')"
    when="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$source_id" \
        "$(basename "$dest")" \
        "$url" \
        "$when" \
        "$bytes" \
        "$sha" \
        "DOWNLOADED_HASHED" \
        >> "$MANIFEST"

    echo "bytes  = $bytes"
    echo "sha256 = $sha"
}

# ======================================================================
# IBGE — POPULATION 2017
# Original contemporary publication file
# ======================================================================

URL="${HTTPS}${IBGE_HOST}/Estimativas_de_Populacao/Estimativas_2017/estimativa_dou_2017.xls"

download_file \
    "IBGE_POP_2017_ORIGINAL" \
    "$URL" \
    "data/external/landing/ibge_population/2017/estimativa_dou_2017.xls"

# ======================================================================
# IBGE — POPULATION 2017
# Later revised version kept separately for retrospective comparison
# ======================================================================

URL="${HTTPS}${IBGE_HOST}/Estimativas_de_Populacao/Estimativas_2017/POP2017_20220905.xls"

download_file \
    "IBGE_POP_2017_REVISED_2022" \
    "$URL" \
    "data/external/landing/ibge_population/2017/POP2017_20220905.xls"

# ======================================================================
# IBGE — POPULATION 2018
# Official DOU estimate file available in historical directory
# ======================================================================

URL="${HTTPS}${IBGE_HOST}/Estimativas_de_Populacao/Estimativas_2018/estimativa_dou_2018_20181019.xls"

download_file \
    "IBGE_POP_2018_DOU" \
    "$URL" \
    "data/external/landing/ibge_population/2018/estimativa_dou_2018_20181019.xls"

# ======================================================================
# IBGE — POPULATION 2018
# Later revised version, never to be confused with point-in-time source
# ======================================================================

URL="${HTTPS}${IBGE_HOST}/Estimativas_de_Populacao/Estimativas_2018/POP2018_20220905.xls"

download_file \
    "IBGE_POP_2018_REVISED_2022" \
    "$URL" \
    "data/external/landing/ibge_population/2018/POP2018_20220905.xls"

# ======================================================================
# IBGE — GDP MUNICIPAL
# 2010–2018 historical database.
# RETROSPECTIVE_ONLY until further temporal audit.
# ======================================================================

URL="${HTTPS}${IBGE_HOST}/Pib_Municipios/2018/base/base_de_dados_2010_2018_xls.zip"

download_file \
    "IBGE_GDP_2010_2018" \
    "$URL" \
    "data/external/landing/ibge_gdp/retrospective/base_de_dados_2010_2018_xls.zip"

# ======================================================================
# ANP — MONTHLY MUNICIPAL FUEL PRICES 2016–2018
# ======================================================================

URL="${HTTPS}${ANP_HOST}/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/precos-revenda-e-de-distribuicao-combustiveis/shlp/mensal/mensal-municipios-2016-a-2018.xlsx"

download_file \
    "ANP_FUEL_MONTHLY_MUNICIPAL_2016_2018" \
    "$URL" \
    "data/external/landing/anp_fuel/monthly_municipality_2016_2018/mensal_municipios_2016_2018.xlsx"

# ======================================================================
# CONSUMIDOR.GOV.BR
#
# Instead of hard-coding unstable resource IDs, retrieve the official
# CKAN metadata and select the historical 2017/2018 resources.
# ======================================================================

echo
echo "======================================================================"
echo "CONSUMIDOR.GOV.BR — RESOLVING OFFICIAL CKAN RESOURCES"
echo "======================================================================"

CKAN_META="$(mktemp)"
CKAN_LIST="$(mktemp)"

trap 'rm -f "$CKAN_META" "$CKAN_LIST"' EXIT

CKAN_URL="${HTTPS}${MJ_HOST}/api/3/action/package_show?id=0182f1bf-e73d-42b1-ae8c-fa94d9ce9451"

curl \
    --fail \
    --location \
    --retry 4 \
    --retry-delay 3 \
    --connect-timeout 30 \
    --output "$CKAN_META" \
    "$CKAN_URL"

python3 - "$CKAN_META" "$CKAN_LIST" <<'PY'
import json
import sys
import unicodedata

meta_path, out_path = sys.argv[1], sys.argv[2]

with open(meta_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if not payload.get("success"):
    raise SystemExit("CKAN package_show returned success=false")

resources = payload["result"]["resources"]

def norm(text):
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().replace(" ", "").replace("-", "").replace("_", "")

wanted = {
    "CONSUMIDOR_2017": None,
    "CONSUMIDOR_2018_H1": None,
    "CONSUMIDOR_2018_H2": None,
}

for r in resources:
    name = r.get("name", "")
    fmt = str(r.get("format", "")).lower()
    url = r.get("url", "")

    n = norm(name)

    if not url:
        continue

    if fmt and "csv" not in fmt:
        continue

    if "2017" in n and "2018" not in n:
        wanted["CONSUMIDOR_2017"] = (name, url)

    if "2018" in n and (
        "1semestre" in n
        or "1osemestre" in n
        or "primeirosemestre" in n
    ):
        wanted["CONSUMIDOR_2018_H1"] = (name, url)

    if "2018" in n and (
        "2semestre" in n
        or "2osemestre" in n
        or "segundosemestre" in n
    ):
        wanted["CONSUMIDOR_2018_H2"] = (name, url)

missing = [k for k, v in wanted.items() if v is None]

if missing:
    print("Resources discovered:", file=sys.stderr)
    for r in resources:
        print(
            r.get("name"),
            r.get("format"),
            r.get("url"),
            file=sys.stderr,
        )
    raise SystemExit(
        "Could not resolve required resources: " + ", ".join(missing)
    )

with open(out_path, "w", encoding="utf-8") as f:
    for source_id, (name, url) in wanted.items():
        f.write(f"{source_id}\t{name}\t{url}\n")
PY

while IFS=$'\t' read -r source_id resource_name resource_url
do

    case "$source_id" in

        CONSUMIDOR_2017)
            dest="data/external/landing/consumidor_gov/2017/consumidor_gov_2017.csv"
            ;;

        CONSUMIDOR_2018_H1)
            dest="data/external/landing/consumidor_gov/2018/consumidor_gov_2018_h1.csv"
            ;;

        CONSUMIDOR_2018_H2)
            dest="data/external/landing/consumidor_gov/2018/consumidor_gov_2018_h2.csv"
            ;;

        *)
            echo "ERRO: source_id inesperado: $source_id"
            exit 1
            ;;
    esac

    echo
    echo "CKAN resource: $resource_name"

    download_file \
        "$source_id" \
        "$resource_url" \
        "$dest"

done < "$CKAN_LIST"

# ======================================================================
# FINAL AUDIT — NO TRANSFORMATION
# ======================================================================

echo
echo "======================================================================"
echo "DOWNLOAD AUDIT"
echo "======================================================================"

echo
echo "FILES:"
find data/external/landing \
    -type f \
    -printf '%s bytes\t%p\n' \
    | sort -n

echo
echo "TOTAL LANDING:"
du -sh data/external/landing

echo
echo "LAST MANIFEST RECORDS:"
tail -n 20 "$MANIFEST"

echo
echo "RAW OLIST:"
du -sh data/raw/olist 2>/dev/null || true

echo
echo "======================================================================"
echo "STATUS = DOWNLOAD_STAGE_COMPLETE"
echo "TRANSFORMATIONS = 0"
echo "JOINS = 0"
echo "RAW_MODIFIED = false"
echo "NEXT = INVENTORY_AND_SCHEMA_AUDIT"
echo "PARAR AQUI"
echo "======================================================================"
