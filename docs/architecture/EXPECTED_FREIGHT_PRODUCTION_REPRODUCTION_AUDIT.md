# Expected Freight — Production Reproduction Audit

## Why this stage exists

Historical `expected_freight_oot` values have been validated as artifacts, but a
production feature must be reproducible at `order_purchase_timestamp`.

Artifact equality alone is not sufficient.

## Block 05

Block 05 ranks the actual producer files, extracts feature/model/temporal
evidence, and classifies discovered data fields under the Point-in-Time
contract.

The audit intentionally distinguishes:

- a column name appearing anywhere in source code;
- a column actually entering the model feature vector.

Only the second can determine PIT eligibility.

## Release rule

Expected Freight remains blocked until an independent runtime reproduction
establishes all of the following:

1. exact input feature vector;
2. exact preprocessing;
3. exact model family and hyperparameters;
4. exact past-only temporal training rule;
5. reproduced OOT predictions;
6. prediction-parity diagnostics versus the historical artifact;
7. explicit treatment of the 267 Gold supervised orders without historical
   Expected Freight coverage.

## Current state

M1 is not released and no production model is trained in this block.
