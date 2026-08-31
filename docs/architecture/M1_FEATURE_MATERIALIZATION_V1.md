# M1 Feature Materialization V1

## Corrected formula freeze

Block 09 initially reproduced B and Q exactly but failed Z because the source
uses a symbolic `EPS` constant and the first parser did not resolve its literal
assignment.

Block 09R resolves `EPS` directly from the active Shipping Friction source AST.

The frozen incremental M1 features are:

- `expected_freight_oot`
- `freight_burden = F/(P+F)`
- `freight_expected_ratio = F/E`
- `freight_log_ratio = log((F+EPS)/(E+EPS))`

Q and Z share the source `valid_expected_pair` guard. The production
recomputation therefore uses the same source-equivalent validity mask that was
already established by Q's exact historical null-pattern parity.

## Availability

The frozen supervised cohort remains 96,470 rows.

No row is silently dropped.

Orders with the complete M1 feature vector are eligible for paired M0-vs-M1
evaluation. Orders without Expected Freight remain in the operational
population and use the M0 fallback.

## Governance

`critical_shipping_friction` remains ex-post and forbidden as a same-order
predictor.

This stage freezes features only. M1 remains unreleased.
