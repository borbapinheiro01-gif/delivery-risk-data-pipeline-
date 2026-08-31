# Expected Freight — Exact Producer Audit

## Why Block 06 was required

Block 05 ranked files by references to `expected_freight_oot`. A high reference
count can identify a consumer rather than the producer.

Block 06 therefore uses stronger evidence:

- assignment to `expected_freight_oot`;
- model `.fit(...)`;
- model `.predict(...)`;
- write of `06_EXPECTED_FREIGHT_OOT.csv`;
- active source only, excluding backup/snapshot paths.

## Execution specification

The output `expected_freight_runtime_execution_spec_v1.json` records:

- primary producer;
- source execution objects;
- estimator-constructor expressions;
- fit and predict expressions;
- candidate feature vectors;
- intersections with forbidden future/outcome fields;
- temporal markers required for out-of-time reproduction.

## Release rule

Expected Freight remains blocked in this block.

Runtime reproduction is permitted only when the exact producer has model, fit,
predict, feature-vector and temporal evidence sufficient to reconstruct the
historical OOT process independently.

M1 remains unreleased.
