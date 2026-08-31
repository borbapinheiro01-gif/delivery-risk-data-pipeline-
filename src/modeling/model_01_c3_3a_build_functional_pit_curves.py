#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-A
FUNCTIONAL POINT-IN-TIME CURVES — RECOVERY / VALIDATED BUILD
===============================================================================

Constrói, para cada pedido supervisionado:

    V_i(b) = número de pedidos históricos no lag diário b

    F_i(b) = soma do freight histórico no lag diário b

com:

    b = 1,...,90

e contrato temporal estrito:

    event_time < prediction_time

Intervalos:

    lag_01 = [t0 - 1 dia,  t0)
    lag_02 = [t0 - 2 dias, t0 - 1 dia)
    ...
    lag_90 = [t0 - 90 dias, t0 - 89 dias)

NÃO:
- aplica smoothing;
- aplica spline;
- aplica FPCA;
- seleciona janela;
- seleciona K;
- cria Silver;
- treina modelo;
- altera RAW.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

RAW = PROJECT / "data" / "raw" / "olist"

MATRIX = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "pretraining"
    / "ORDER_CORE_V1_AUDIT_MATRIX.csv"
)

GATE05 = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_05_point_in_time"
    / "dq_gate_05_summary.json"
)

C321 = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
    / "01ad_common_scale_nullspace_summary.json"
)

OUT = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

ART = (
    PROJECT
    / "artifacts"
    / "model_01_order_logistic"
    / "functional_feasibility"
)

OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OUTPUTS — nomes esperados pelo C3.3-B
# =============================================================================

P_ORDER_INDEX = OUT / "02a_functional_pit_order_index.csv"
P_MANIFEST = OUT / "02b_functional_pit_curve_manifest.csv"
P_WINDOW = OUT / "02c_functional_pit_window_summary.csv"
P_BRUTE = OUT / "02e_functional_pit_bruteforce_validation.csv"

P_VOLUME = ART / "02g_purchase_volume_curve_90d.npy"
P_FREIGHT = ART / "02h_purchase_freight_curve_90d.npy"

P_VALIDATION = OUT / "02i_functional_pit_validation.csv"
P_SUMMARY = OUT / "02j_functional_pit_summary.json"
P_REPORT = OUT / "02k_functional_pit_report.txt"


# =============================================================================
# CONSTANTS
# =============================================================================

N_EXPECTED = 96470
MAX_LAG = 90
WINDOWS = [30, 60, 90]

# 24 horas em nanossegundos.
# Constante explícita para evitar o warning de timedelta visto no outro script.
DAY_NS = 86_400_000_000_000

BRUTE_SAMPLE = 60


# =============================================================================
# HELPERS
# =============================================================================

def add_check(rows, check, condition, observed, expected):
    rows.append({
        "check": check,
        "status": "PASS" if bool(condition) else "FAIL",
        "observed": observed,
        "expected": expected,
    })


def unweighted_curves(target_ns, event_ns):
    """
    V_i,b = número de eventos em:
            [t_i-(b+1)d, t_i-bd)
    """

    result = np.zeros(
        (len(target_ns), MAX_LAG),
        dtype=np.int32
    )

    for b in range(MAX_LAG):

        lower = target_ns - (b + 1) * DAY_NS
        upper = target_ns - b * DAY_NS

        left = np.searchsorted(
            event_ns,
            lower,
            side="left"
        )

        right = np.searchsorted(
            event_ns,
            upper,
            side="left"
        )

        result[:, b] = (right - left).astype(np.int32)

    return result


def weighted_curves(target_ns, event_ns, weights):
    """
    F_i,b = soma dos pesos de eventos em:
            [t_i-(b+1)d, t_i-bd)
    """

    result = np.zeros(
        (len(target_ns), MAX_LAG),
        dtype=np.float64
    )

    prefix = np.zeros(
        len(weights) + 1,
        dtype=np.float64
    )

    np.cumsum(
        weights,
        dtype=np.float64,
        out=prefix[1:]
    )

    for b in range(MAX_LAG):

        lower = target_ns - (b + 1) * DAY_NS
        upper = target_ns - b * DAY_NS

        left = np.searchsorted(
            event_ns,
            lower,
            side="left"
        )

        right = np.searchsorted(
            event_ns,
            upper,
            side="left"
        )

        result[:, b] = (
            prefix[right]
            -
            prefix[left]
        )

    return result


def direct_curve(prediction_ns, event_ns, weights=None):
    """
    Implementação independente para auditoria.

    Não reutiliza o cálculo bin-a-bin baseado em searchsorted.
    """

    lower = prediction_ns - MAX_LAG * DAY_NS

    mask = (
        (event_ns >= lower)
        &
        (event_ns < prediction_ns)
    )

    selected_times = event_ns[mask]

    delta = prediction_ns - selected_times

    # Exatamente 24h atrás pertence ao lag 1.
    bin_index = (
        (delta - 1)
        //
        DAY_NS
    ).astype(np.int64)

    valid = (
        (bin_index >= 0)
        &
        (bin_index < MAX_LAG)
    )

    bin_index = bin_index[valid]

    if weights is None:

        return np.bincount(
            bin_index,
            minlength=MAX_LAG
        ).astype(np.int64)

    selected_weights = weights[mask][valid]

    return np.bincount(
        bin_index,
        weights=selected_weights,
        minlength=MAX_LAG
    ).astype(np.float64)


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

start = time.perf_counter()

print()
print("=" * 112)
print("MODEL 01.0-C3.3-A — FUNCTIONAL PIT CURVES")
print("=" * 112)

required = [
    (MATRIX, "ORDER_CORE_V1"),
    (GATE05, "Gate 05 summary"),
    (C321, "C3.2.1 summary"),
    (RAW / "olist_orders_dataset.csv", "orders RAW"),
    (RAW / "olist_order_items_dataset.csv", "order_items RAW"),
]

for path, label in required:

    if not path.exists():
        raise SystemExit(
            f"[FAIL] {label} ausente: {path}"
        )

    print(f"[PASS] {label}")


with GATE05.open(encoding="utf-8") as f:
    gate05 = json.load(f)

with C321.open(encoding="utf-8") as f:
    c321 = json.load(f)


if gate05.get("status") != "PASS":
    raise SystemExit("[FAIL] Gate 05 não está PASS.")

if gate05.get("model_01_unlocked") is not True:
    raise SystemExit("[FAIL] MODEL_01 não está liberado.")

if c321.get("status") != "PASS":
    raise SystemExit("[FAIL] C3.2.1 não está PASS.")

if (
    c321.get("structural_nullspace_assessment")
    !=
    "INVARIANT_IDENTITIES_CONFIRMED"
):
    raise SystemExit(
        "[FAIL] Nullspace estrutural ainda não confirmado."
    )

print("[PASS] Governança temporal validada.")


# =============================================================================
# 2. PEDIDOS SUPERVISIONADOS
# =============================================================================

task = pd.read_csv(
    MATRIX,
    usecols=[
        "order_id",
        "order_purchase_timestamp",
        "late_delivery_calendar_day",
    ]
)

task["order_purchase_timestamp"] = pd.to_datetime(
    task["order_purchase_timestamp"],
    errors="coerce"
)

if task["order_purchase_timestamp"].isna().any():
    raise RuntimeError("Timestamp inválido na matriz.")

task = (
    task
    .sort_values(
        ["order_purchase_timestamp", "order_id"]
    )
    .reset_index(drop=True)
)

if len(task) != N_EXPECTED:
    raise RuntimeError(
        f"Esperado {N_EXPECTED}; obtido {len(task)}."
    )

if not task["order_id"].is_unique:
    raise RuntimeError("order_id não é único.")

task["curve_row_index"] = np.arange(
    len(task),
    dtype=np.int64
)

task["purchase_month"] = (
    task["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

target_ns = (
    task["order_purchase_timestamp"]
    .astype("int64")
    .to_numpy()
)

task[
    [
        "curve_row_index",
        "order_id",
        "order_purchase_timestamp",
        "purchase_month",
        "late_delivery_calendar_day",
    ]
].to_csv(
    P_ORDER_INDEX,
    index=False
)

print()
print("Supervised orders:", f"{len(task):,}")


# =============================================================================
# 3. PURCHASE EVENT STREAM
# =============================================================================

orders = pd.read_csv(
    RAW / "olist_orders_dataset.csv",
    usecols=[
        "order_id",
        "order_purchase_timestamp",
    ]
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

purchase_events = (
    orders.loc[
        orders["order_purchase_timestamp"].notna(),
        [
            "order_id",
            "order_purchase_timestamp",
        ]
    ]
    .sort_values(
        ["order_purchase_timestamp", "order_id"]
    )
    .reset_index(drop=True)
)

purchase_ns = (
    purchase_events["order_purchase_timestamp"]
    .astype("int64")
    .to_numpy()
)

if np.any(purchase_ns[1:] < purchase_ns[:-1]):
    raise RuntimeError("Purchase stream não está ordenado.")

print("Purchase events  :", f"{len(purchase_events):,}")


# =============================================================================
# 4. FREIGHT EVENT STREAM
# =============================================================================

items = pd.read_csv(
    RAW / "olist_order_items_dataset.csv",
    usecols=[
        "order_id",
        "freight_value",
    ]
)

items["freight_value"] = pd.to_numeric(
    items["freight_value"],
    errors="coerce"
)

if items["freight_value"].isna().any():
    raise RuntimeError(
        "freight_value possui missing inesperado."
    )

freight_order = (
    items.groupby(
        "order_id",
        as_index=False,
        sort=False
    )
    .agg(
        total_freight=(
            "freight_value",
            "sum"
        )
    )
)

freight_events = (
    orders.merge(
        freight_order,
        on="order_id",
        how="inner",
        validate="one_to_one"
    )
    .loc[
        lambda x:
            x["order_purchase_timestamp"].notna()
    ]
    .sort_values(
        ["order_purchase_timestamp", "order_id"]
    )
    .reset_index(drop=True)
)

freight_ns = (
    freight_events["order_purchase_timestamp"]
    .astype("int64")
    .to_numpy()
)

freight_weight = (
    freight_events["total_freight"]
    .to_numpy(dtype=np.float64)
)

if np.any(freight_ns[1:] < freight_ns[:-1]):
    raise RuntimeError("Freight stream não está ordenado.")

if not np.isfinite(freight_weight).all():
    raise RuntimeError("Freight stream possui NaN/Inf.")

print("Freight events   :", f"{len(freight_events):,}")


# =============================================================================
# 5. CONSTRUIR CURVAS
# =============================================================================

print()
print("=" * 112)
print("CONSTRUINDO CURVAS PIT")
print("=" * 112)

t = time.perf_counter()

volume_curve = unweighted_curves(
    target_ns,
    purchase_ns
)

print(
    "Volume :",
    volume_curve.shape,
    "| dtype:",
    volume_curve.dtype,
    f"| {time.perf_counter() - t:.3f}s"
)

t = time.perf_counter()

freight_curve = weighted_curves(
    target_ns,
    freight_ns,
    freight_weight
)

print(
    "Freight:",
    freight_curve.shape,
    "| dtype:",
    freight_curve.dtype,
    f"| {time.perf_counter() - t:.3f}s"
)


# =============================================================================
# 6. SALVAR MATRIZES
# =============================================================================

np.save(
    P_VOLUME,
    volume_curve,
    allow_pickle=False
)

np.save(
    P_FREIGHT,
    freight_curve,
    allow_pickle=False
)


# =============================================================================
# 7. MANIFEST
# =============================================================================

manifest = pd.DataFrame([
    {
        "channel": "purchase_volume",
        "availability_rule": "purchase_j < purchase_i",
        "event_time": "orders.order_purchase_timestamp",
        "weight": "1",
        "lags": 90,
        "bin": "[t0-(b+1)*24h, t0-b*24h)",
        "shape": str(volume_curve.shape),
        "dtype": str(volume_curve.dtype),
        "status": "DIAGNOSTIC_ONLY",
    },
    {
        "channel": "purchase_freight",
        "availability_rule": "purchase_j < purchase_i",
        "event_time": "orders.order_purchase_timestamp",
        "weight": "SUM(order_items.freight_value) per historical order",
        "lags": 90,
        "bin": "[t0-(b+1)*24h, t0-b*24h)",
        "shape": str(freight_curve.shape),
        "dtype": str(freight_curve.dtype),
        "status": "DIAGNOSTIC_ONLY",
    },
])

manifest.to_csv(
    P_MANIFEST,
    index=False
)


# =============================================================================
# 8. RESUMO 30/60/90
# =============================================================================

window_rows = []

for window in WINDOWS:

    volume_total = volume_curve[:, :window].sum(
        axis=1,
        dtype=np.int64
    )

    freight_total = freight_curve[:, :window].sum(
        axis=1,
        dtype=np.float64
    )

    for channel, values in [
        ("purchase_volume", volume_total),
        ("purchase_freight", freight_total),
    ]:

        window_rows.append({
            "channel": channel,
            "window_days": window,
            "orders": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
            "zero_orders": int((values == 0).sum()),
            "zero_pct": float(100.0 * (values == 0).mean()),
        })

window_df = pd.DataFrame(window_rows)

window_df.to_csv(
    P_WINDOW,
    index=False
)


# =============================================================================
# 9. BRUTE-FORCE INDEPENDENTE
# =============================================================================

print()
print("=" * 112)
print("BRUTE-FORCE INDEPENDENTE")
print("=" * 112)

sample_index = np.unique(
    np.linspace(
        0,
        len(task) - 1,
        BRUTE_SAMPLE,
        dtype=np.int64
    )
)

brute_rows = []

for idx in sample_index:

    pred = int(target_ns[idx])

    expected_v = direct_curve(
        pred,
        purchase_ns,
        weights=None
    )

    expected_f = direct_curve(
        pred,
        freight_ns,
        weights=freight_weight
    )

    observed_v = volume_curve[idx].astype(np.int64)
    observed_f = freight_curve[idx].astype(np.float64)

    volume_bad = (
        observed_v
        !=
        expected_v
    )

    freight_bad = (
        ~np.isclose(
            observed_f,
            expected_f,
            rtol=1e-10,
            atol=1e-8
        )
    )

    brute_rows.append({
        "curve_row_index": int(idx),
        "order_id": task.iloc[idx]["order_id"],
        "prediction_time":
            task.iloc[idx]["order_purchase_timestamp"],

        "volume_bin_mismatches":
            int(volume_bad.sum()),

        "volume_max_abs_diff":
            float(
                np.max(
                    np.abs(
                        observed_v
                        -
                        expected_v
                    )
                )
            ),

        "freight_bin_mismatches":
            int(freight_bad.sum()),

        "freight_max_abs_diff":
            float(
                np.max(
                    np.abs(
                        observed_f
                        -
                        expected_f
                    )
                )
            ),

        "pit_rule":
            "event_time < prediction_time",
    })


brute_df = pd.DataFrame(brute_rows)

brute_df.to_csv(
    P_BRUTE,
    index=False
)

volume_mismatches = int(
    brute_df[
        "volume_bin_mismatches"
    ].sum()
)

freight_mismatches = int(
    brute_df[
        "freight_bin_mismatches"
    ].sum()
)

print("Orders audited           :", len(brute_df))
print("Volume bin mismatches    :", volume_mismatches)
print("Freight bin mismatches   :", freight_mismatches)


# =============================================================================
# 10. VALIDATION
# =============================================================================

checks = []

add_check(
    checks,
    "rows",
    len(task) == N_EXPECTED,
    len(task),
    N_EXPECTED
)

add_check(
    checks,
    "order_id_unique",
    task["order_id"].is_unique,
    int(task["order_id"].duplicated().sum()),
    0
)

add_check(
    checks,
    "volume_shape",
    volume_curve.shape == (N_EXPECTED, MAX_LAG),
    str(volume_curve.shape),
    "(96470, 90)"
)

add_check(
    checks,
    "freight_shape",
    freight_curve.shape == (N_EXPECTED, MAX_LAG),
    str(freight_curve.shape),
    "(96470, 90)"
)

add_check(
    checks,
    "volume_nonnegative",
    bool((volume_curve >= 0).all()),
    int((volume_curve < 0).sum()),
    0
)

add_check(
    checks,
    "freight_nonnegative",
    bool((freight_curve >= 0).all()),
    int((freight_curve < 0).sum()),
    0
)

add_check(
    checks,
    "volume_finite",
    bool(np.isfinite(volume_curve).all()),
    int((~np.isfinite(volume_curve)).sum()),
    0
)

add_check(
    checks,
    "freight_finite",
    bool(np.isfinite(freight_curve).all()),
    int((~np.isfinite(freight_curve)).sum()),
    0
)

v30 = volume_curve[:, :30].sum(axis=1, dtype=np.int64)
v60 = volume_curve[:, :60].sum(axis=1, dtype=np.int64)
v90 = volume_curve[:, :90].sum(axis=1, dtype=np.int64)

f30 = freight_curve[:, :30].sum(axis=1, dtype=np.float64)
f60 = freight_curve[:, :60].sum(axis=1, dtype=np.float64)
f90 = freight_curve[:, :90].sum(axis=1, dtype=np.float64)

volume_nesting_bad = int(
    (~(
        (v30 <= v60)
        &
        (v60 <= v90)
    )).sum()
)

freight_nesting_bad = int(
    (~(
        (f30 <= f60 + 1e-8)
        &
        (f60 <= f90 + 1e-8)
    )).sum()
)

add_check(
    checks,
    "volume_window_nesting",
    volume_nesting_bad == 0,
    volume_nesting_bad,
    0
)

add_check(
    checks,
    "freight_window_nesting",
    freight_nesting_bad == 0,
    freight_nesting_bad,
    0
)

add_check(
    checks,
    "bruteforce_volume",
    volume_mismatches == 0,
    volume_mismatches,
    0
)

add_check(
    checks,
    "bruteforce_freight",
    freight_mismatches == 0,
    freight_mismatches,
    0
)


validation = pd.DataFrame(checks)

validation.to_csv(
    P_VALIDATION,
    index=False
)

failures = int(
    validation["status"].eq("FAIL").sum()
)


# =============================================================================
# 11. SUMMARY JSON
# =============================================================================

runtime = float(
    time.perf_counter()
    -
    start
)

summary = {
    "step":
        "MODEL_01_0_C3_3A_FUNCTIONAL_PIT_CURVES",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "stage_assessment":
        (
            "FUNCTIONAL_CURVE_CONSTRUCTION_VIABLE"
            if failures == 0
            else "REVIEW_REQUIRED"
        ),

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "task_orders":
        int(len(task)),

    "purchase_event_rows":
        int(len(purchase_events)),

    "freight_event_rows":
        int(len(freight_events)),

    "curve_channels": [
        "purchase_volume",
        "purchase_freight"
    ],

    "curve_shape_each": [
        N_EXPECTED,
        MAX_LAG
    ],

    "candidate_windows_days":
        WINDOWS,

    "point_in_time_rule":
        "event_time < prediction_time",

    "volume_dtype":
        str(volume_curve.dtype),

    "freight_dtype":
        str(freight_curve.dtype),

    "bruteforce_sample_orders":
        int(len(brute_df)),

    "volume_bruteforce_bin_mismatches":
        volume_mismatches,

    "freight_bruteforce_bin_mismatches":
        freight_mismatches,

    "runtime_seconds":
        runtime,

    "smoothing_applied":
        False,

    "basis_applied":
        False,

    "fpca_applied":
        False,

    "functional_module_unlocked":
        False,

    "window_selected":
        False,

    "component_count_selected":
        False,

    "folds_frozen":
        False,

    "model_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures,
}

P_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =============================================================================
# 12. REPORT
# =============================================================================

report = f"""
====================================================================================================
MODEL 01.0-C3.3-A — FUNCTIONAL POINT-IN-TIME CURVES
====================================================================================================

STATUS                         : {summary["status"]}
STAGE ASSESSMENT               : {summary["stage_assessment"]}

TASK ORDERS                    : {len(task):,}
PURCHASE EVENTS                : {len(purchase_events):,}
FREIGHT EVENTS                 : {len(freight_events):,}

CURVE SHAPE                    : {volume_curve.shape}
VOLUME DTYPE                   : {volume_curve.dtype}
FREIGHT DTYPE                  : {freight_curve.dtype}

POINT-IN-TIME RULE
----------------------------------------------------------------------------------------------------
event_time < prediction_time

lag_01 = [t0 - 1 day,  t0)
lag_90 = [t0 - 90 days, t0 - 89 days)

INDEPENDENT VALIDATION
----------------------------------------------------------------------------------------------------
Sample orders                  : {len(brute_df)}
Volume bin mismatches          : {volume_mismatches}
Freight bin mismatches         : {freight_mismatches}

VALIDATION FAILURES            : {failures}

IMPORTANT
----------------------------------------------------------------------------------------------------
No smoothing applied.
No basis expansion applied.
No FPCA applied.
No window selected.
No component count selected.
No feature added to MODEL_01.
No fold frozen.
No model trained.
No Silver created.
RAW not modified.

RUNTIME                        : {runtime:.3f} s
====================================================================================================
""".strip()

P_REPORT.write_text(
    report,
    encoding="utf-8"
)


# =============================================================================
# 13. PRINT
# =============================================================================

print()
print("=" * 112)
print("WINDOW SUMMARY")
print("=" * 112)

print(
    window_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("=" * 112)
print("VALIDATION")
print("=" * 112)

print(
    validation.to_string(
        index=False
    )
)

print()
print("=" * 112)
print("RESULTADO C3.3-A")
print("=" * 112)

print("FAILURES                 =", failures)
print("TASK ORDERS              =", len(task))
print("CURVE SHAPE              =", volume_curve.shape)
print("VOLUME DTYPE             =", volume_curve.dtype)
print("FREIGHT DTYPE            =", freight_curve.dtype)
print("VOLUME BRUTE MISMATCHES  =", volume_mismatches)
print("FREIGHT BRUTE MISMATCHES =", freight_mismatches)
print("RUNTIME SECONDS          =", runtime)

print()
print(
    "STAGE ASSESSMENT         =",
    summary["stage_assessment"]
)

print()
print("WINDOW SELECTED           = NÃO")
print("K SELECTED                = NÃO")
print("SMOOTHING APPLIED         = NÃO")
print("FPCA APPLIED              = NÃO")
print("FUNCTIONAL MODULE         = AINDA NÃO LIBERADO")
print("FOLDS FROZEN              = NÃO")
print("MODEL TRAINED             = NÃO")
print("SILVER CREATED            = NÃO")
print("RAW MODIFIED              = NÃO")

if failures:
    raise SystemExit(2)

print()
print("[PASS] C3.3-A validado.")
print("[PASS] PARAR AQUI — não executar C3.3-B ainda.")
