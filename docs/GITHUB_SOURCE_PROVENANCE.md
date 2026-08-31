# GitHub Source Provenance

## Status

The historical monolithic script:

`src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py`

is retained only as legacy provenance.

It is not considered part of the current reproducible execution path.

## Reason

The file contains a historical syntax defect associated with a duplicated
`try:` block. Equivalent historical copies contain the same defect.

The scientific artifacts currently used by the project were generated later
by the checkpoint-based spatiotemporal pipeline.

## Current spatiotemporal execution path

1. `00_environment_and_registry.py`
2. `01_build_route_checkpoint.py`
3. `02_build_external_context_checkpoint.py`
4. `03_build_master_checkpoint.py`
5. `04_run_four_analyses.py`
6. `05_integrated_validation_report.py`

Subsequent scientific analyses consume the artifacts created by this pipeline.

## Core derived artifacts

- `01_ROUTE_SELLER_ORDER.csv`
- `02_ROUTE_ORDER_LEVEL.csv`
- `03_MUNICIPAL_CONTEXT.csv`
- `04_ANP_CONTEXT.csv`
- `04b_ANP_STATE_CONTEXT.csv`
- `05_SPATIOTEMPORAL_CONTEXT_CORE.csv`
- `06_EXPECTED_FREIGHT_OOT.csv`

## Publication policy

The four broken monolithic copies are classified as:

`LEGACY_SUPERSEDED_NONRUNNABLE`

They should not appear as active executable source in the public repository.

No scientific result was modified by this classification.
