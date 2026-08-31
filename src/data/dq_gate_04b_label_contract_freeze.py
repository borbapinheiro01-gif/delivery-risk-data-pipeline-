#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys

import pandas as pd


PROJECT = Path.home() / "workspace" / "Delivery_Risk_Intelligence"

GATE04_DIR = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04_label_construct"
)

GATE03B_DIR = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_03b_conditional_task"
)

REG11 = (
    PROJECT
    / "metadata"
    / "registry_v1_1"
)

REG12 = (
    PROJECT
    / "metadata"
    / "registry_v1_2"
)

CONTRACT_DIR = PROJECT / "contracts"

OUT = (
    PROJECT
    / "reports"
    / "data_quality"
    / "gate_04b_label_contract"
)

REG12.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path):

    if not path.exists():
        raise SystemExit(
            f"[ERRO] Arquivo obrigatório ausente:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, obj):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )


# =============================================================================
# 1. PRÉ-REQUISITOS
# =============================================================================

print("=" * 104)
print("DQ GATE 04B — LABEL CONTRACT FREEZE & REGISTRY SYNC")
print("=" * 104)
print()


gate04 = load_json(
    GATE04_DIR
    / "dq_gate_04_summary.json"
)

gate03b = load_json(
    GATE03B_DIR
    / "dq_gate_03b_summary.json"
)

reg11_manifest = load_json(
    REG11
    / "registry_v1_1_manifest.json"
)


if gate04.get("status") != "PASS":

    raise SystemExit(
        "[BLOQUEADO] Gate 04 não está PASS."
    )


if gate03b.get("status") != "PASS":

    raise SystemExit(
        "[BLOQUEADO] Gate 03B não está PASS."
    )


if reg11_manifest.get(
    "internal_registry_status"
) != "PASS":

    raise SystemExit(
        "[BLOQUEADO] Registry 1.1 não está PASS."
    )


print("[PASS] DQ Gate 03B")
print("[PASS] DQ Gate 04")
print("[PASS] Registry 1.1")
print()


# =============================================================================
# 2. VALIDAR A EVIDÊNCIA DO LABEL
# =============================================================================

task_n = int(
    gate04[
        "task_candidate_orders"
    ]
)

late_n = int(
    gate04[
        "calendar_day_late_orders"
    ]
)

late_rate = float(
    gate04[
        "calendar_day_late_rate_pct"
    ]
)

disagreements = int(
    gate04[
        "label_disagreements"
    ]
)

same_day_disagreements = int(
    gate04[
        "same_day_disagreements"
    ]
)

same_day_pct = float(
    gate04[
        "same_day_disagreement_pct"
    ]
)

midnight_pct = float(
    gate04[
        "estimated_midnight_pct"
    ]
)

calendar_late_strict_ontime = int(
    gate04[
        "calendar_late_strict_ontime"
    ]
)


checks = []


def check(name, condition, observed):

    status = (
        "PASS"
        if condition
        else
        "FAIL"
    )

    checks.append(
        {
            "check":
                name,

            "status":
                status,

            "observed":
                observed
        }
    )


check(
    "gate04_pass",
    gate04.get("status") == "PASS",
    gate04.get("status")
)

check(
    "estimated_time_is_calendar_semantic",
    midnight_pct == 100.0,
    f"{midnight_pct:.6f}% midnight"
)

check(
    "all_label_disagreements_are_same_day",
    (
        disagreements > 0
        and
        same_day_disagreements == disagreements
        and
        same_day_pct == 100.0
    ),
    (
        f"{same_day_disagreements}/"
        f"{disagreements}"
    )
)

check(
    "no_calendar_late_strict_ontime_cases",
    calendar_late_strict_ontime == 0,
    calendar_late_strict_ontime
)


checks_df = pd.DataFrame(
    checks
)


checks_df.to_csv(
    OUT
    / "01_label_contract_validation.csv",
    index=False
)


failures = int(
    (
        checks_df[
            "status"
        ]
        ==
        "FAIL"
    )
    .sum()
)


if failures:

    print(
        checks_df.to_string(
            index=False
        )
    )

    raise SystemExit(
        "[FAIL] Evidência insuficiente para congelar o contrato."
    )


print(
    checks_df.to_string(
        index=False
    )
)

print()


# =============================================================================
# 3. LABEL CONTRACT V1.0
# =============================================================================

prevalence = (
    late_n
    /
    task_n
)


label_contract = {

    "contract_id":
        "DELIVERY_RISK_LABEL_V1",

    "contract_version":
        "1.0.0",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "status":
        "FROZEN_FOR_MODELING",

    "scope":
        "Delivery Risk Intelligence Platform using the public Olist dataset",

    "unit_of_analysis":
        "order_id",

    "prediction_time":
        "order_purchase_timestamp",

    "target_concept":
        "delivery after the promised calendar date",

    "primary_label": {

        "name":
            "late_delivery_calendar_day",

        "formula":
            (
                "Y_i = 1[date(order_delivered_customer_date_i) "
                "> date(order_estimated_delivery_date_i)]"
            ),

        "positive_value":
            1,

        "negative_value":
            0
    },

    "eligible_supervised_population": {

        "requirements": [
            "order_status == delivered",
            "order_purchase_timestamp observed",
            "order_estimated_delivery_date observed",
            "order_delivered_customer_date observed"
        ],

        "orders":
            task_n
    },

    "class_distribution": {

        "positive_orders":
            late_n,

        "negative_orders":
            task_n - late_n,

        "positive_rate":
            prevalence,

        "positive_rate_pct":
            late_rate,

        "no_skill_pr_auc":
            prevalence
    },

    "construct_evidence": {

        "estimated_timestamp_midnight_pct":
            midnight_pct,

        "timestamp_vs_calendar_disagreements":
            disagreements,

        "same_calendar_day_disagreements":
            same_day_disagreements,

        "same_day_share_pct":
            same_day_pct,

        "calendar_late_strict_ontime":
            calendar_late_strict_ontime
    },

    "sensitivity_labels": [

        {
            "name":
                "timestamp_strict",

            "role":
                "sensitivity_only",

            "late_orders":
                int(
                    gate04[
                        "timestamp_strict_late_orders"
                    ]
                ),

            "late_rate_pct":
                float(
                    gate04[
                        "timestamp_strict_late_rate_pct"
                    ]
                )
        },

        {
            "name":
                "calendar_day_plus_1_grace",

            "role":
                "sensitivity_only",

            "late_orders":
                int(
                    gate04[
                        "calendar_day_plus_1_grace_late_orders"
                    ]
                )
        }
    ],

    "unobserved_outcome_policy": {

        "rule":
            "DO_NOT_CONVERT_TO_ON_TIME",

        "interpretation":
            (
                "Orders without an observed delivery outcome "
                "are excluded from the supervised label population "
                "unless a separate modeling framework is defined."
            )
    },

    "external_business_truth": {

        "olist_private_historical_sla_verified":
            False,

        "status":
            "NOT_VERIFIED",

        "interpretation":
            (
                "The calendar-day label is frozen as the "
                "modeling contract for this case. "
                "This does not claim independent verification "
                "of Olist's private historical SLA semantics."
            )
    },

    "raw_modified":
        False
}


LABEL_CONTRACT_PATH = (
    CONTRACT_DIR
    / "DELIVERY_RISK_LABEL_CONTRACT_V1.json"
)


save_json(
    LABEL_CONTRACT_PATH,
    label_contract
)


# =============================================================================
# 4. PRESERVAR REGISTRY 1.1 E CRIAR REGISTRY 1.2
# =============================================================================

for src in REG11.iterdir():

    if src.is_file():

        shutil.copy2(
            src,
            REG12
            / src.name
        )


canonical_path = (
    REG11
    / "canonical_issue_registry.csv"
)


issues = pd.read_csv(
    canonical_path,
    low_memory=False
)


issues[
    "resolution_status"
] = "OPEN"


issues[
    "resolved_by"
] = ""


issues[
    "resolution_evidence"
] = ""


issues[
    "resolution_decision"
] = ""


issues[
    "external_truth_status"
] = ""


issues[
    "last_updated_utc"
] = datetime.now(
    timezone.utc
).isoformat()


# -----------------------------------------------------------------------------
# ISSUE-MISSINGNESS-001
# -----------------------------------------------------------------------------

mask = (
    issues[
        "canonical_issue_id"
    ]
    ==
    "ISSUE-MISSINGNESS-001"
)


issues.loc[
    mask,
    "resolution_status"
] = "RESOLVED_ASSESSMENT"


issues.loc[
    mask,
    "resolved_by"
] = "DQ_GATE_03B"


issues.loc[
    mask,
    "resolution_evidence"
] = (
    "DQ Gate 03B PASS; relation missingness, "
    "attribute missingness, source quality and "
    "task quality quantified separately."
)


issues.loc[
    mask,
    "resolution_decision"
] = (
    "Assessment closed. Residual missingness "
    "continues to treatment-policy stage."
)


# -----------------------------------------------------------------------------
# ISSUE-OBSERVATION-WINDOW-001
# -----------------------------------------------------------------------------

mask = (
    issues[
        "canonical_issue_id"
    ]
    ==
    "ISSUE-OBSERVATION-WINDOW-001"
)


issues.loc[
    mask,
    "resolution_status"
] = "RESOLVED_POLICY"


issues.loc[
    mask,
    "resolved_by"
] = "DQ_GATE_04"


issues.loc[
    mask,
    "resolution_evidence"
] = (
    "Observation-window audit completed; "
    "unobserved outcomes were separated from observed deliveries."
)


issues.loc[
    mask,
    "resolution_decision"
] = (
    "Unobserved outcomes must not be encoded as on-time."
)


# -----------------------------------------------------------------------------
# ISSUE-TARGET-001
# -----------------------------------------------------------------------------

mask = (
    issues[
        "canonical_issue_id"
    ]
    ==
    "ISSUE-TARGET-001"
)


issues.loc[
    mask,
    "resolution_status"
] = "RESOLVED_MODELING_CONTRACT"


issues.loc[
    mask,
    "resolved_by"
] = "DQ_GATE_04B"


issues.loc[
    mask,
    "resolution_evidence"
] = (
    f"estimated_midnight_pct={midnight_pct:.6f}; "
    f"label_disagreements={disagreements}; "
    f"same_day_share={same_day_pct:.6f}%."
)


issues.loc[
    mask,
    "resolution_decision"
] = (
    "Use CALENDAR_DAY as primary modeling label. "
    "Timestamp strict and +1-day grace remain sensitivity analyses."
)


issues.loc[
    mask,
    "external_truth_status"
] = (
    "OLIST_PRIVATE_SLA_NOT_VERIFIED"
)


# -----------------------------------------------------------------------------
# EXTERNAL GROUND TRUTH — manter explicitamente aberto
# -----------------------------------------------------------------------------

mask = (
    issues[
        "canonical_issue_id"
    ]
    ==
    "ISSUE-GROUND-TRUTH-001"
)


issues.loc[
    mask,
    "resolution_status"
] = "OPEN_LIMITATION"


issues.loc[
    mask,
    "external_truth_status"
] = "NOT_AVAILABLE"


# =============================================================================
# SAVE REGISTRY 1.2
# =============================================================================

issues.to_csv(
    REG12
    / "canonical_issue_registry.csv",
    index=False
)


issues.to_csv(
    REG12
    / "canonical_issue_registry_v1_2.csv",
    index=False
)


resolution_registry = issues[
    [
        "canonical_issue_id",
        "canonical_topic",
        "priority",
        "resolution_status",
        "resolved_by",
        "resolution_evidence",
        "resolution_decision",
        "external_truth_status",
        "last_updated_utc"
    ]
].copy()


resolution_registry.to_csv(
    REG12
    / "issue_resolution_registry.csv",
    index=False
)


resolved_count = int(
    issues[
        "resolution_status"
    ]
    .str.startswith(
        "RESOLVED",
        na=False
    )
    .sum()
)


open_count = int(
    len(
        issues
    )
    -
    resolved_count
)


manifest12 = dict(
    reg11_manifest
)


manifest12.update(
    {
        "registry_version":
            "1.2.0",

        "parent_registry_version":
            "1.1.0",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "internal_registry_status":
            "PASS",

        "label_contract":
            "contracts/DELIVERY_RISK_LABEL_CONTRACT_V1.json",

        "resolved_canonical_issues":
            resolved_count,

        "non_resolved_canonical_issues":
            open_count,

        "updates": [
            "Label contract frozen for modeling",
            "ISSUE-MISSINGNESS-001 synchronized",
            "ISSUE-OBSERVATION-WINDOW-001 synchronized",
            "ISSUE-TARGET-001 synchronized",
            "External Olist SLA truth explicitly remains unverified"
        ],

        "raw_modified":
            False
    }
)


save_json(
    REG12
    / "registry_v1_2_manifest.json",
    manifest12
)


# =============================================================================
# 5. REPORT
# =============================================================================

report = f"""
========================================================================================================
DQ GATE 04B — LABEL CONTRACT FREEZE & REGISTRY SYNC
========================================================================================================

STATUS                              : PASS

PRIMARY LABEL                       : CALENDAR_DAY

FORMULA:
Y_i = 1[
    date(order_delivered_customer_date_i)
    >
    date(order_estimated_delivery_date_i)
]

UNIT OF ANALYSIS                    : order_id
PREDICTION TIME                     : order_purchase_timestamp

SUPERVISED ORDERS                   : {task_n:,}
LATE ORDERS                         : {late_n:,}
ON-TIME ORDERS                      : {task_n - late_n:,}
LATE RATE                           : {late_rate:.6f}%
NO-SKILL PR-AUC                     : {prevalence:.8f}

CONSTRUCT EVIDENCE
--------------------------------------------------------------------------------------------------------
Estimated timestamps at midnight   : {midnight_pct:.6f}%
Label disagreements                 : {disagreements:,}
Same-day disagreements              : {same_day_disagreements:,}
Same-day share                      : {same_day_pct:.6f}%
Calendar-late / strict-on-time      : {calendar_late_strict_ontime:,}

CONTRACT STATUS
--------------------------------------------------------------------------------------------------------
Modeling label contract             : FROZEN
External Olist historical SLA       : NOT VERIFIED

ISSUES SYNCHRONIZED
--------------------------------------------------------------------------------------------------------
ISSUE-MISSINGNESS-001               : RESOLVED_ASSESSMENT
ISSUE-OBSERVATION-WINDOW-001        : RESOLVED_POLICY
ISSUE-TARGET-001                    : RESOLVED_MODELING_CONTRACT

IMPORTANT
--------------------------------------------------------------------------------------------------------
Registry 1.1 was preserved.
Registry 1.2 was created.
Gate 04 results were preserved.
RAW was not modified.
Silver was not created.

========================================================================================================
"""


REPORT_PATH = (
    OUT
    / "DQ_GATE_04B_LABEL_CONTRACT_REPORT.txt"
)


REPORT_PATH.write_text(
    report.strip()
    +
    "\n",
    encoding="utf-8"
)


summary = {

    "gate":
        "DQ_GATE_04B_LABEL_CONTRACT_FREEZE",

    "status":
        "PASS",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "primary_label":
        "CALENDAR_DAY",

    "label_contract_status":
        "FROZEN_FOR_MODELING",

    "external_olist_sla_status":
        "NOT_VERIFIED",

    "supervised_orders":
        task_n,

    "late_orders":
        late_n,

    "late_rate_pct":
        late_rate,

    "no_skill_pr_auc":
        prevalence,

    "registry_before":
        "1.1.0",

    "registry_after":
        "1.2.0",

    "resolved_issues":
        [
            "ISSUE-MISSINGNESS-001",
            "ISSUE-OBSERVATION-WINDOW-001",
            "ISSUE-TARGET-001"
        ],

    "raw_modified":
        False,

    "silver_created":
        False
}


save_json(
    OUT
    / "dq_gate_04b_summary.json",
    summary
)


print(report)

print("[OK] Label Contract:")
print(LABEL_CONTRACT_PATH)

print()
print("[OK] Registry 1.2:")
print(REG12)

print()
print("[OK] RAW não modificado.")

