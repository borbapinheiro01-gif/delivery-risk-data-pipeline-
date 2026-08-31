#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/workspace/Delivery_Risk_Intelligence"
cd "$ROOT"

HOST="dados.mj.gov.br"
PACKAGE_ID="0182f1bf-e73d-42b1-ae8c-fa94d9ce9451"

MANIFEST="data/external/registry/download_manifest.tsv"
META_DIR="metadata/spatiotemporal_logistics"

mkdir -p "$META_DIR"

NETWORK_LOG="$META_DIR/consumidor_gov_download_network.log"

: > "$NETWORK_LOG"

echo "======================================================================"
echo "CONSUMIDOR.GOV.BR — HISTORICAL 2017/2018 ACQUISITION"
echo "DNS-RESILIENT OFFICIAL DOWNLOAD"
echo "======================================================================"
echo

if [[ ! -f "$MANIFEST" ]]; then
    echo "[FAIL] Manifest não encontrado:"
    echo "$MANIFEST"
    exit 1
fi

# ---------------------------------------------------------------------
# HTTP FETCH
#
# 1. tenta resolução DNS normal;
# 2. se falhar, consulta DNS público via HTTPS;
# 3. mantém dados.mj.gov.br como hostname TLS usando --resolve.
# ---------------------------------------------------------------------

RESOLVE_IP=""

get_public_ip () {

    if [[ -n "$RESOLVE_IP" ]]; then
        return 0
    fi

    echo "[INFO] Consultando DNS público por HTTPS..." | tee -a "$NETWORK_LOG"

    local tmp_dns
    tmp_dns="$(mktemp)"

    if ! curl \
        --fail \
        --silent \
        --show-error \
        --location \
        --retry 3 \
        --connect-timeout 20 \
        "https://dns.google/resolve?name=${HOST}&type=A" \
        -o "$tmp_dns"
    then
        rm -f "$tmp_dns"

        echo "[FAIL] Não foi possível consultar DNS público." \
            | tee -a "$NETWORK_LOG"

        return 1
    fi

    RESOLVE_IP="$(
        python3 - "$tmp_dns" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)

answers = obj.get("Answer", [])

ips = [
    str(x.get("data", "")).strip()
    for x in answers
    if x.get("type") == 1
]

ips = [x for x in ips if x]

if not ips:
    raise SystemExit(2)

print(ips[0])
PY
    )"

    rm -f "$tmp_dns"

    if [[ -z "$RESOLVE_IP" ]]; then
        echo "[FAIL] DNS público não retornou IPv4." \
            | tee -a "$NETWORK_LOG"
        return 1
    fi

    echo "[OK] IPv4 público obtido para $HOST" \
        | tee -a "$NETWORK_LOG"

    echo "[INFO] IP registrado apenas no log de rede desta execução." \
        | tee -a "$NETWORK_LOG"
}


fetch_official () {

    local url="$1"
    local output="$2"

    rm -f "${output}.part"

    echo
    echo "[TRY] DNS normal:"
    echo "$url"

    if curl \
        --fail \
        --silent \
        --show-error \
        --location \
        --retry 2 \
        --retry-delay 2 \
        --connect-timeout 20 \
        --max-time 180 \
        -o "${output}.part" \
        "$url"
    then

        mv "${output}.part" "$output"

        echo "[OK] Download usando DNS normal." \
            | tee -a "$NETWORK_LOG"

        return 0
    fi

    rm -f "${output}.part"

    echo
    echo "[WARN] DNS normal falhou."
    echo "[INFO] Ativando fallback DNS-over-HTTPS."

    get_public_ip

    if curl \
        --fail \
        --silent \
        --show-error \
        --location \
        --retry 3 \
        --retry-delay 2 \
        --connect-timeout 20 \
        --max-time 300 \
        --resolve "${HOST}:443:${RESOLVE_IP}" \
        -o "${output}.part" \
        "$url"
    then

        mv "${output}.part" "$output"

        echo "[OK] Download usando fallback DNS público." \
            | tee -a "$NETWORK_LOG"

        return 0
    fi

    rm -f "${output}.part"

    echo "[FAIL] Não foi possível obter:"
    echo "$url"

    return 1
}


# ---------------------------------------------------------------------
# FETCH CKAN METADATA
# ---------------------------------------------------------------------

TMP_META="$(mktemp)"
TMP_LIST="$(mktemp)"

cleanup () {
    rm -f "$TMP_META" "$TMP_LIST"
}

trap cleanup EXIT

CKAN_URL="https://${HOST}/api/3/action/package_show?id=${PACKAGE_ID}"

echo
echo "======================================================================"
echo "1. CKAN METADATA"
echo "======================================================================"

fetch_official "$CKAN_URL" "$TMP_META"


# ---------------------------------------------------------------------
# DISCOVER EXACT 2017 / 2018 RESOURCES
# ---------------------------------------------------------------------

echo
echo "======================================================================"
echo "2. DISCOVERING HISTORICAL RESOURCES"
echo "======================================================================"

python3 - "$TMP_META" "$TMP_LIST" <<'PY'
import json
import sys
import unicodedata

meta_path = sys.argv[1]
output_path = sys.argv[2]

with open(meta_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if payload.get("success") is not True:
    raise SystemExit(
        "FAIL: CKAN package_show returned success != true"
    )

resources = payload["result"].get("resources", [])


def norm(value):
    text = str(value or "")

    text = (
        text
        .replace("º", "o")
        .replace("°", "o")
        .replace("ª", "a")
    )

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    return "".join(
        ch.lower()
        for ch in text
        if ch.isalnum()
    )


wanted = {
    "CONSUMIDOR_2017": None,
    "CONSUMIDOR_2018_H1": None,
    "CONSUMIDOR_2018_H2": None,
}


for resource in resources:

    name = str(resource.get("name", ""))
    url = str(resource.get("url", ""))
    fmt = str(resource.get("format", ""))

    n = norm(name)

    if not url:
        continue

    # -------------------------------------------
    # 2017
    # -------------------------------------------

    if (
        "2017" in n
        and "2018" not in n
        and (
            "dadosconsumidor" in n
            or "consumidor" in n
        )
    ):
        wanted["CONSUMIDOR_2017"] = {
            "name": name,
            "url": url,
            "id": resource.get("id", ""),
            "format": fmt,
        }

    # -------------------------------------------
    # 2018 first semester
    # -------------------------------------------

    if "2018" in n and (
        "1osemestre" in n
        or "1semestre" in n
        or "primeirosemestre" in n
    ):
        wanted["CONSUMIDOR_2018_H1"] = {
            "name": name,
            "url": url,
            "id": resource.get("id", ""),
            "format": fmt,
        }

    # -------------------------------------------
    # 2018 second semester
    # -------------------------------------------

    if "2018" in n and (
        "2osemestre" in n
        or "2semestre" in n
        or "segundosemestre" in n
    ):
        wanted["CONSUMIDOR_2018_H2"] = {
            "name": name,
            "url": url,
            "id": resource.get("id", ""),
            "format": fmt,
        }


missing = [
    source_id
    for source_id, resource in wanted.items()
    if resource is None
]

if missing:

    print(
        "\nFAIL: historical resources not resolved:",
        ", ".join(missing),
        file=sys.stderr,
    )

    print(
        "\nRESOURCES FOUND IN OFFICIAL PACKAGE:",
        file=sys.stderr,
    )

    for r in resources:
        print(
            repr(r.get("name")),
            "|",
            r.get("format"),
            "|",
            r.get("id"),
            file=sys.stderr,
        )

    raise SystemExit(3)


with open(output_path, "w", encoding="utf-8") as f:

    for source_id, r in wanted.items():

        print(
            f"[FOUND] {source_id}",
            "|",
            r["name"],
            "|",
            r["id"],
        )

        f.write(
            "\t".join([
                source_id,
                r["id"],
                r["format"],
                r["name"].replace("\t", " "),
                r["url"],
            ])
            + "\n"
        )
PY


# ---------------------------------------------------------------------
# VALIDATE DOWNLOADED RESOURCE
# ---------------------------------------------------------------------

validate_download () {

    local file="$1"

    python3 - "$file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])

if not path.exists():
    raise SystemExit("FAIL: file does not exist")

size = path.stat().st_size

if size <= 100:
    raise SystemExit(
        f"FAIL: suspiciously small file: {size} bytes"
    )

head = path.read_bytes()[:4096].lstrip().lower()

bad_signatures = (
    b"<!doctype html",
    b"<html",
    b"<?xml",
)

if head.startswith(bad_signatures):
    raise SystemExit(
        "FAIL: downloaded resource looks like HTML/XML, not CSV"
    )

print(f"[PASS] file payload validation | bytes={size}")
PY
}


# ---------------------------------------------------------------------
# MANIFEST DUPLICATE CHECK
# ---------------------------------------------------------------------

manifest_has () {

    local source_id="$1"
    local filename="$2"

    awk -F '\t' \
        -v sid="$source_id" \
        -v fn="$filename" \
        '
        NR > 1 && $1 == sid && $2 == fn {
            found=1
        }
        END {
            exit(found ? 0 : 1)
        }
        ' \
        "$MANIFEST"
}


# ---------------------------------------------------------------------
# DOWNLOAD HISTORICAL FILES
# ---------------------------------------------------------------------

echo
echo "======================================================================"
echo "3. DOWNLOAD HISTORICAL CSV FILES"
echo "======================================================================"

while IFS=$'\t' read -r \
    source_id \
    resource_id \
    resource_format \
    resource_name \
    resource_url

do

    case "$source_id" in

        CONSUMIDOR_2017)
            DEST="data/external/landing/consumidor_gov/2017/consumidor_gov_2017.csv"
            ;;

        CONSUMIDOR_2018_H1)
            DEST="data/external/landing/consumidor_gov/2018/consumidor_gov_2018_h1.csv"
            ;;

        CONSUMIDOR_2018_H2)
            DEST="data/external/landing/consumidor_gov/2018/consumidor_gov_2018_h2.csv"
            ;;

        *)
            echo "[FAIL] source_id inesperado: $source_id"
            exit 1
            ;;
    esac

    mkdir -p "$(dirname "$DEST")"

    echo
    echo "---------------------------------------------------------------------"
    echo "SOURCE       : $source_id"
    echo "RESOURCE ID  : $resource_id"
    echo "RESOURCE     : $resource_name"
    echo "FORMAT       : $resource_format"
    echo "DEST         : $DEST"
    echo "---------------------------------------------------------------------"

    if [[ -s "$DEST" ]]; then

        echo "[EXISTS] arquivo já existe."
        echo "[INFO] Não será sobrescrito."

    else

        fetch_official "$resource_url" "$DEST"

    fi

    validate_download "$DEST"

    BYTES="$(stat -c '%s' "$DEST")"
    SHA256="$(sha256sum "$DEST" | awk '{print $1}')"
    WHEN="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    FILENAME="$(basename "$DEST")"

    echo "bytes        = $BYTES"
    echo "sha256       = $SHA256"

    if manifest_has "$source_id" "$FILENAME"; then

        echo "[INFO] registro já existe no manifesto; não duplicando."

    else

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$source_id" \
            "$FILENAME" \
            "$resource_url" \
            "$WHEN" \
            "$BYTES" \
            "$SHA256" \
            "DOWNLOADED_HASHED" \
            >> "$MANIFEST"

        echo "[PASS] manifest atualizado."

    fi

done < "$TMP_LIST"


# ---------------------------------------------------------------------
# FINAL AUDIT
# ---------------------------------------------------------------------

echo
echo "======================================================================"
echo "4. FINAL CONSUMIDOR.GOV AUDIT"
echo "======================================================================"

find data/external/landing/consumidor_gov \
    -type f \
    -printf '%s bytes\t%p\n' \
    | sort -n

echo
echo "SHA256:"
find data/external/landing/consumidor_gov \
    -type f \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum

echo
echo "MANIFEST:"
grep '^CONSUMIDOR_' "$MANIFEST" || true

echo
echo "RAW OLIST:"
du -sh data/raw/olist

echo
echo "EXTERNAL LANDING TOTAL:"
du -sh data/external/landing

echo
echo "======================================================================"
echo "STATUS = PASS"
echo "CONSUMIDOR_GOV_2017_2018 = ACQUIRED"
echo "TRANSFORMATIONS = 0"
echo "JOINS = 0"
echo "RAW_MODIFIED = false"
echo "NEXT = 01_INVENTORY_EXTERNAL"
echo "PARAR AQUI"
echo "======================================================================"
