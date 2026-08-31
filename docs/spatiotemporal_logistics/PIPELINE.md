# SPATIOTEMPORAL_LOGISTICS_AUDIT

## Scientific scope

Audit associations among:

- product price
- freight
- freight burden
- origin/destination
- distance
- package characteristics
- municipal context
- fuel context
- complaint context
- delivery speed
- delivery reliability
- late delivery

## Immutable source rule

`data/raw/olist/` must never be modified by this pipeline.

External original files are stored under:

`data/external/landing/`

Landing files are immutable after successful acquisition and hashing.

## Pipeline

A0 Acquisition
-> A1 Inventory/schema audit
-> A2 Temporal eligibility
-> A3 Geographic harmonization
-> A4 Source-specific curation
-> A5 Join coverage audit
-> A6 Context feature construction
-> A7 Spatiotemporal logistics audit

## Temporal classes

PIT_ELIGIBLE
PIT_ELIGIBLE_WITH_LAG
RETROSPECTIVE_ONLY
HOLD

No external variable becomes a predictive feature before temporal eligibility
is explicitly audited.

## Geographic rule

Do not blindly join on municipality name.

Build and audit:

Olist city + UF
-> normalized municipality identity
-> IBGE municipality code.

## Analysis layers

1. Cost
2. Speed
3. Reliability

No causal interpretation is allowed without an appropriate causal design.

No carrier identity shall be inferred from anonymized Olist identifiers.
