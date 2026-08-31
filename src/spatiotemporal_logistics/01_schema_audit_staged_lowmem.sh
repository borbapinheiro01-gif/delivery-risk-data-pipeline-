#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/workspace/Delivery_Risk_Intelligence"
cd "$ROOT"

OUT="reports/spatiotemporal_logistics/schema_audit"
mkdir -p "$OUT"

echo "===================================================================================================="
echo "SPATIOTEMPORAL LOGISTICS — 4-STAGE LOW-MEMORY SCHEMA AUDIT"
echo "===================================================================================================="

###############################################################################
# STAGE 1 — OLIST
###############################################################################

echo
echo "===================================================================================================="
echo "STAGE 1/4 — OLIST / ROUTE INPUTS"
echo "===================================================================================================="

python3 - <<'PY'
from pathlib import Path
import csv
import json

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
RAW = ROOT / "data" / "raw" / "olist"
OUT = ROOT / "reports" / "spatiotemporal_logistics" / "schema_audit"

patterns = {
    "orders": "*orders_dataset.csv",
    "items": "*order_items_dataset.csv",
    "customers": "*customers_dataset.csv",
    "sellers": "*sellers_dataset.csv",
    "products": "*products_dataset.csv",
    "geolocation": "*geolocation_dataset.csv",
}

result = {}

for name, pattern in patterns.items():

    files = sorted(RAW.glob(pattern))

    if len(files) != 1:
        raise SystemExit(
            f"[FAIL] {name}: esperado 1 arquivo, encontrados {len(files)}"
        )

    path = files[0]

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:
        reader = csv.reader(f)
        header = next(reader)

        rows = 0
        for _ in reader:
            rows += 1

    result[name] = {
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": header,
    }

    print()
    print(f"[PASS] {name}")
    print(f"  rows    = {rows:,}")
    print(f"  columns = {header}")

(OUT / "01_olist_schema.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("[PASS] STAGE 1 COMPLETE")
PY

echo
echo "[MEMORY AFTER STAGE 1]"
free -h | head -2

###############################################################################
# STAGE 2 — IBGE POPULATION
###############################################################################

echo
echo "===================================================================================================="
echo "STAGE 2/4 — IBGE POPULATION 2017/2018"
echo "===================================================================================================="

python3 - <<'PY'
from pathlib import Path
import json
import xlrd

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"
BASE = ROOT / "data" / "external" / "landing" / "ibge_population"
OUT = ROOT / "reports" / "spatiotemporal_logistics" / "schema_audit"

files = {
    "2017": BASE / "2017" / "estimativa_dou_2017.xls",
    "2018": BASE / "2018" / "estimativa_dou_2018_20181019.xls",
}

result = {}

for year, path in files.items():

    if not path.exists():
        raise SystemExit(f"[FAIL] arquivo ausente: {path}")

    book = xlrd.open_workbook(
        str(path),
        on_demand=True
    )

    year_info = {
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sheets": [],
    }

    print()
    print(f"--- IBGE POP {year} ---")

    for sheet_name in book.sheet_names():

        sheet = book.sheet_by_name(sheet_name)

        info = {
            "sheet": sheet_name,
            "nrows": sheet.nrows,
            "ncols": sheet.ncols,
            "first_nonempty_rows": [],
        }

        print(
            f"[SHEET] {sheet_name!r} "
            f"rows={sheet.nrows:,} cols={sheet.ncols}"
        )

        found = 0

        for r in range(min(sheet.nrows, 80)):

            values = [
                sheet.cell_value(r, c)
                for c in range(sheet.ncols)
            ]

            useful = [
                str(v).strip()
                for v in values
                if str(v).strip()
            ]

            if not useful:
                continue

            info["first_nonempty_rows"].append(
                {
                    "row_index_zero_based": r,
                    "values": useful,
                }
            )

            print(
                f"  row {r}: {useful[:10]}"
            )

            found += 1

            if found >= 15:
                break

        year_info["sheets"].append(info)

    book.release_resources()

    result[year] = year_info

(OUT / "02_ibge_population_schema.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("[PASS] STAGE 2 COMPLETE")
PY

echo
echo "[MEMORY AFTER STAGE 2]"
free -h | head -2

###############################################################################
# STAGE 3 — IBGE GDP ZIP
###############################################################################

echo
echo "===================================================================================================="
echo "STAGE 3/4 — IBGE GDP 2010–2018"
echo "===================================================================================================="

python3 - <<'PY'
from pathlib import Path
import json
import tempfile
import zipfile
import xlrd

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

ZIP = (
    ROOT
    / "data"
    / "external"
    / "landing"
    / "ibge_gdp"
    / "retrospective"
    / "base_de_dados_2010_2018_xls.zip"
)

OUT = ROOT / "reports" / "spatiotemporal_logistics" / "schema_audit"

if not ZIP.exists():
    raise SystemExit(f"[FAIL] ZIP PIB ausente: {ZIP}")

with zipfile.ZipFile(ZIP) as z:

    members = [
        i for i in z.infolist()
        if i.filename.lower().endswith((".xls", ".xlsx"))
        and not i.filename.startswith("__MACOSX")
    ]

    print(f"ZIP members XLS/XLSX = {len(members)}")

    for info in members:
        print(
            f"  {info.file_size:,} bytes | {info.filename}"
        )

    if not members:
        raise SystemExit("[FAIL] nenhum workbook encontrado no ZIP")

    member = max(
        members,
        key=lambda x: x.file_size
    )

    print()
    print(f"[SELECTED] {member.filename}")

    with tempfile.TemporaryDirectory() as td:

        z.extract(member, td)

        path = Path(td) / member.filename

        if path.suffix.lower() != ".xls":
            print(
                "[INFO] Workbook principal não é .xls; "
                "será auditado posteriormente com openpyxl."
            )

            result = {
                "zip": str(ZIP.relative_to(ROOT)),
                "selected_member": member.filename,
                "selected_bytes": member.file_size,
                "format": path.suffix.lower(),
            }

        else:

            book = xlrd.open_workbook(
                str(path),
                on_demand=True
            )

            result = {
                "zip": str(ZIP.relative_to(ROOT)),
                "selected_member": member.filename,
                "selected_bytes": member.file_size,
                "sheets": [],
            }

            for sheet_name in book.sheet_names():

                sheet = book.sheet_by_name(sheet_name)

                sinfo = {
                    "sheet": sheet_name,
                    "nrows": sheet.nrows,
                    "ncols": sheet.ncols,
                    "sample_rows": [],
                }

                print()
                print(
                    f"[SHEET] {sheet_name!r} "
                    f"rows={sheet.nrows:,} cols={sheet.ncols}"
                )

                found = 0

                for r in range(min(sheet.nrows, 50)):

                    values = [
                        sheet.cell_value(r, c)
                        for c in range(sheet.ncols)
                    ]

                    useful = [
                        str(v).strip()
                        for v in values
                        if str(v).strip()
                    ]

                    if not useful:
                        continue

                    sinfo["sample_rows"].append(
                        {
                            "row_index_zero_based": r,
                            "values": useful[:40],
                        }
                    )

                    print(
                        f"  row {r}: {useful[:20]}"
                    )

                    found += 1

                    if found >= 12:
                        break

                result["sheets"].append(sinfo)

            book.release_resources()

(OUT / "03_ibge_gdp_schema.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("[PASS] STAGE 3 COMPLETE")
PY

echo
echo "[MEMORY AFTER STAGE 3]"
free -h | head -2

###############################################################################
# STAGE 4 — ANP XLSX — READ ONLY
###############################################################################

echo
echo "===================================================================================================="
echo "STAGE 4/4 — ANP MUNICIPAL MONTHLY 2016–2018"
echo "===================================================================================================="

python3 - <<'PY'
from pathlib import Path
import json

from openpyxl import load_workbook

ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

PATH = (
    ROOT
    / "data"
    / "external"
    / "landing"
    / "anp_fuel"
    / "monthly_municipality_2016_2018"
    / "mensal_municipios_2016_2018.xlsx"
)

OUT = ROOT / "reports" / "spatiotemporal_logistics" / "schema_audit"

if not PATH.exists():
    raise SystemExit(f"[FAIL] ANP ausente: {PATH}")

# read_only=True:
# leitura lazy/streaming para reduzir RAM.
wb = load_workbook(
    PATH,
    read_only=True,
    data_only=True,
)

result = {
    "file": str(PATH.relative_to(ROOT)),
    "bytes": PATH.stat().st_size,
    "sheets": [],
}

for ws in wb.worksheets:

    print()
    print(
        f"[SHEET] {ws.title!r} "
        f"dimension={ws.calculate_dimension()}"
    )

    rows_saved = []

    for i, row in enumerate(
        ws.iter_rows(
            values_only=True
        )
    ):

        useful = [
            str(v).strip()
            for v in row
            if v is not None
            and str(v).strip()
        ]

        if useful:

            rows_saved.append(
                {
                    "row_index_zero_based": i,
                    "values": useful[:40],
                }
            )

            print(
                f"  row {i}: {useful[:20]}"
            )

        if len(rows_saved) >= 15:
            break

    result["sheets"].append(
        {
            "sheet": ws.title,
            "dimension": ws.calculate_dimension(),
            "sample_rows": rows_saved,
        }
    )

wb.close()

(OUT / "04_anp_schema.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("[PASS] STAGE 4 COMPLETE")
PY

echo
echo "[MEMORY AFTER STAGE 4]"
free -h | head -2

###############################################################################
# CONSOLIDATED STATUS
###############################################################################

echo
echo "===================================================================================================="
echo "4-STAGE AUDIT RESULTS"
echo "===================================================================================================="

ls -lh "$OUT"

echo
echo "CGROUP NOW:"
if [[ -r /sys/fs/cgroup/memory.current ]]; then
    cat /sys/fs/cgroup/memory.current
fi

echo
echo "MEMORY NOW:"
free -h

echo
echo "===================================================================================================="
echo "STATUS = PASS"
echo "STAGE_1_OLIST_SCHEMA = PASS"
echo "STAGE_2_IBGE_POP_SCHEMA = PASS"
echo "STAGE_3_IBGE_GDP_SCHEMA = PASS"
echo "STAGE_4_ANP_SCHEMA = PASS"
echo "RAW_MODIFIED = false"
echo "DATA_TRANSFORMED = false"
echo "NEXT = BUILD_4_STAGED_CURATED_TABLES"
echo "PARAR AQUI"
echo "===================================================================================================="
