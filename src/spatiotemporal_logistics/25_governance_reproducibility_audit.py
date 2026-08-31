#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import py_compile
import sys

import numpy as np
import pandas as pd


ROOT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts" / "spatiotemporal_logistics"
SCI = ROOT / "reports" / "spatiotemporal_logistics" / "scientific"
GN = SCI / "gnadmm_delivery"
OUT = ROOT / "reports" / "final_closeout"
SRC = ROOT / "src" / "spatiotemporal_logistics"

OUT.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 105)
print("MODULE 25 — GOVERNANCE / REPRODUCIBILITY AUDIT")
print("=" * 105)


# ==============================================================================================
# HELPERS
# ==============================================================================================

def sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def file_gate(path, label=None):
    path = Path(path)

    return {
        "gate":
            label
            or f"FILE::{path.name}",

        "status":
            "PASS"
            if path.exists() and path.stat().st_size > 0
            else "FAIL",

        "detail":
            str(
                path.relative_to(ROOT)
            ),
    }


# ==============================================================================================
# 1. RAW MANIFEST
# ==============================================================================================

raw_files = sorted(
    p
    for p in RAW.rglob("*")
    if p.is_file()
)

raw_rows = []

for p in raw_files:

    raw_rows.append(
        {
            "relative_path":
                str(
                    p.relative_to(ROOT)
                ),

            "size_bytes":
                int(
                    p.stat().st_size
                ),

            "sha256":
                sha256(p),
        }
    )


raw_manifest = pd.DataFrame(
    raw_rows
)

raw_manifest.to_csv(
    OUT
    / "25a_current_raw_sha256_manifest.csv",
    index=False,
)


# ==============================================================================================
# 2. LOOK FOR HISTORICAL HASH EVIDENCE
# ==============================================================================================

historical_hash_candidates = []

for base in [
    ROOT / "reports",
    ROOT / "artifacts",
]:

    if not base.exists():
        continue

    for p in base.rglob("*"):

        if not p.is_file():
            continue

        name = p.name.lower()

        if (
            "sha256" in name
            or
            "hash" in name
        ):

            if (
                p
                !=
                OUT
                / "25a_current_raw_sha256_manifest.csv"
            ):

                historical_hash_candidates.append(
                    str(
                        p.relative_to(ROOT)
                    )
                )


# ==============================================================================================
# 3. REQUIRED FILE GATES
# ==============================================================================================

required_files = [
    ART
    / "01_ROUTE_SELLER_ORDER.csv",

    ART
    / "02_ROUTE_ORDER_LEVEL.csv",

    ART
    / "03_MUNICIPAL_CONTEXT.csv",

    ART
    / "04_ANP_CONTEXT.csv",

    ART
    / "04b_ANP_STATE_CONTEXT.csv",

    ART
    / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv",

    ART
    / "06_EXPECTED_FREIGHT_OOT.csv",

    GN
    / "19d_PAIRED_TEMPORAL_DECISION.json",

    GN
    / "20d_CONSTRAINT_ABLATION_DECISION.json",

    GN
    / "21d_CALIBRATION_DECISION.json",

    GN
    / "22e_GNADMM_SCIENTIFIC_FREEZE.json",

    OUT
    / "23d_SCIENTIFIC_EVIDENCE_FREEZE.json",

    OUT
    / "24d_PREDICTIVE_MODEL_AUDIT.json",
]


gates = [
    file_gate(p)
    for p in required_files
]


# ==============================================================================================
# 4. ORDER MASTER ROW CONSISTENCY
# ==============================================================================================

core_n = None
freight_n = None

row_status = "FAIL"

try:

    core_n = len(
        pd.read_csv(
            ART
            / "05_SPATIOTEMPORAL_CONTEXT_CORE.csv",
            usecols=["order_id"],
        )
    )

    freight_n = len(
        pd.read_csv(
            ART
            / "06_EXPECTED_FREIGHT_OOT.csv",
            usecols=["order_id"],
        )
    )

    if (
        core_n == freight_n
        and
        core_n == 96211
    ):
        row_status = "PASS"

    elif core_n == freight_n:
        row_status = "CAVEAT"

except Exception as exc:

    row_status = "FAIL"


gates.append(
    {
        "gate":
            "ORDER_MASTER_ROW_CONSISTENCY",

        "status":
            row_status,

        "detail":
            (
                f"context_core_N={core_n}; "
                f"expected_freight_N={freight_n}"
            ),
    }
)


# ==============================================================================================
# 5. GN DESIGN FILE — ALLOW_PICKLE FALSE
# ==============================================================================================

npz_path = (
    ART
    / "gnadmm_delivery"
    / "15_delivery_gnadmm_design.npz"
)

npz_status = "FAIL"
npz_detail = "NPZ missing."

if npz_path.exists():

    try:

        d = np.load(
            npz_path,
            allow_pickle=False,
        )

        X = d["X"]
        y = d["y"]
        months = d["months"]

        if (
            X.ndim == 2
            and
            y.ndim == 1
            and
            months.ndim == 1
            and
            len(X) == len(y)
            and
            len(y) == len(months)
            and
            X.dtype != object
            and
            y.dtype != object
            and
            months.dtype != object
        ):

            npz_status = "PASS"

        else:
            npz_status = "FAIL"

        npz_detail = (
            f"X={X.shape}/{X.dtype}; "
            f"y={y.shape}/{y.dtype}; "
            f"months={months.shape}/{months.dtype}"
        )

    except Exception as exc:

        npz_status = "FAIL"

        npz_detail = (
            f"{type(exc).__name__}: {exc}"
        )


gates.append(
    {
        "gate":
            "GN_DESIGN_ALLOW_PICKLE_FALSE",

        "status":
            npz_status,

        "detail":
            npz_detail,
    }
)


# ==============================================================================================
# 6. GN FEATURE GUARDRAIL
# ==============================================================================================

metadata_path = (
    ART
    / "gnadmm_delivery"
    / "15_delivery_gnadmm_metadata.json"
)

allowed_features = {
    "intercept",
    "z_freight",
    "z_freight_sq",
    "log_distance",
    "log_weight",
    "log_volume",
    "promised_days",
    "interstate",
}

feature_status = "FAIL"
feature_detail = "Metadata missing."

if metadata_path.exists():

    meta = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    features = set(
        meta.get(
            "feature_names",
            []
        )
    )

    unexpected = sorted(
        features
        -
        allowed_features
    )

    missing = sorted(
        allowed_features
        -
        features
    )

    if (
        not unexpected
        and
        not missing
    ):

        feature_status = "PASS"

        feature_detail = (
            "Frozen GN feature set matches expected specification."
        )

    else:

        feature_status = "FAIL"

        feature_detail = (
            f"unexpected={unexpected}; "
            f"missing={missing}"
        )


gates.append(
    {
        "gate":
            "GN_FEATURE_GUARDRAIL",

        "status":
            feature_status,

        "detail":
            feature_detail,
    }
)


# ==============================================================================================
# 7. TEMPORAL OOT EVALUATION
# ==============================================================================================

fold_path = (
    GN
    / "18b_gnadmm_oot_fold_metrics.csv"
)

temporal_status = "FAIL"
temporal_detail = "OOT fold table missing."

if fold_path.exists():

    folds = pd.read_csv(
        fold_path
    )

    months = sorted(
        folds[
            "test_month"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(months) == 14:
        temporal_status = "PASS"

    elif len(months) > 0:
        temporal_status = "CAVEAT"

    temporal_detail = (
        f"OOT months={len(months)}; "
        f"first={months[0] if months else None}; "
        f"last={months[-1] if months else None}"
    )


gates.append(
    {
        "gate":
            "TEMPORAL_OOT_EVALUATION_PRESENT",

        "status":
            temporal_status,

        "detail":
            temporal_detail,
    }
)


# ==============================================================================================
# 8. PYTHON COMPILE 15–25
# ==============================================================================================

compile_failures = []

compiled_files = []

for n in range(
    15,
    26
):

    candidates = sorted(
        SRC.glob(
            f"{n:02d}_*.py"
        )
    )

    for p in candidates:

        try:

            py_compile.compile(
                str(p),
                doraise=True,
            )

            compiled_files.append(
                str(
                    p.relative_to(ROOT)
                )
            )

        except Exception as exc:

            compile_failures.append(
                {
                    "file":
                        str(
                            p.relative_to(ROOT)
                        ),

                    "error":
                        str(exc),
                }
            )


gates.append(
    {
        "gate":
            "PYTHON_COMPILE_15_25",

        "status":
            "PASS"
            if not compile_failures
            else "FAIL",

        "detail":
            (
                f"compiled={len(compiled_files)}"
                if not compile_failures
                else
                json.dumps(
                    compile_failures,
                    ensure_ascii=False,
                )
            ),
    }
)


# ==============================================================================================
# 9. GOVERNANCE GUARDRAILS
# ==============================================================================================

guardrails = {
    "prediction_time":
        "ORDER_PURCHASE_TIMESTAMP",

    "target":
        "LATE_DELIVERY_CALENDAR_DAY",

    "expected_freight":
        (
            "Out-of-time model-derived quantity; "
            "downstream analyses inherit upstream uncertainty."
        ),

    "distance":
        (
            "Great-circle/Haversine geographic proxy; "
            "not road-network distance."
        ),

    "volume":
        (
            "Product-volume-sum proxy; "
            "not observed parcel cubage."
        ),

    "GDP":
        (
            "Retrospective contextual variable; "
            "not point-in-time operational information."
        ),

    "population":
        (
            "Annual municipal estimate with publication-timing caveat."
        ),

    "ANP":
        (
            "Municipal missingness is preserved; "
            "state-level context remains separate."
        ),

    "causality":
        (
            "Observational associations only; "
            "no causal identification."
        ),

    "GNADMM":
        (
            "No claim of global convergence "
            "for the complete external nonlinear sequence."
        ),

    "curvature_constraint":
        (
            "beta_z2 >= 0 is an imposed modeling hypothesis "
            "and is not evidence of a true U-shaped mechanism."
        ),
}


(
    OUT
    / "25b_governance_guardrails.json"
).write_text(
    json.dumps(
        guardrails,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==============================================================================================
# 10. ENVIRONMENT SNAPSHOT
# ==============================================================================================

packages = {}

for package in [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "patsy",
]:

    try:

        packages[
            package
        ] = importlib.metadata.version(
            package
        )

    except Exception:

        packages[
            package
        ] = None


environment = {
    "python":
        sys.version,

    "platform":
        platform.platform(),

    "packages":
        packages,
}


(
    OUT
    / "25c_environment.json"
).write_text(
    json.dumps(
        environment,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==============================================================================================
# 11. FINAL GATE TABLE
# ==============================================================================================

gate_df = pd.DataFrame(
    gates
)

gate_df.to_csv(
    OUT
    / "25d_governance_gates.csv",
    index=False,
)


n_fail = int(
    (
        gate_df[
            "status"
        ]
        ==
        "FAIL"
    ).sum()
)

n_caveat = int(
    (
        gate_df[
            "status"
        ]
        ==
        "CAVEAT"
    ).sum()
)


if n_fail > 0:

    final_status = "FAIL"

elif n_caveat > 0:

    final_status = "PASS_WITH_CAVEATS"

else:

    final_status = "PASS"


# ==============================================================================================
# 12. FINAL GOVERNANCE AUDIT
# ==============================================================================================

audit = {
    "status":
        final_status,

    "raw_file_count":
        int(
            len(raw_manifest)
        ),

    "raw_manifest_created":
        True,

    "historical_hash_evidence_candidates":
        historical_hash_candidates[:50],

    "hash_guardrail":
        (
            "The current SHA256 manifest is an integrity checkpoint. "
            "An unchanged-from-original claim requires comparison with "
            "an independent earlier machine-readable hash record."
        ),

    "context_core_n":
        core_n,

    "expected_freight_n":
        freight_n,

    "gn_design_allow_pickle_false":
        npz_status == "PASS",

    "temporal_oot_gate":
        temporal_status,

    "failed_gates":
        n_fail,

    "caveat_gates":
        n_caveat,

    "raw_modified_by_module_25":
        False,
}


(
    OUT
    / "25e_FINAL_GOVERNANCE_AUDIT.json"
).write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ==============================================================================================
# 13. TERMINAL OUTPUT
# ==============================================================================================

print()
print("GOVERNANCE GATES")
print("-" * 105)

print(
    gate_df.to_string(
        index=False
    )
)

print()

print(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)

print()


if final_status == "FAIL":

    print(
        "[FAIL 25] GOVERNANCE AUDIT CONTAINS FAIL GATES."
    )

    raise SystemExit(2)


print(
    f"[PASS 25] GOVERNANCE AUDIT = {final_status}"
)
