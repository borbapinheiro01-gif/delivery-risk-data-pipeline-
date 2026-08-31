# M1 Shipping Intelligence — PIT Lineage Audit

## Prediction time

`order_purchase_timestamp`

## Purpose

M1 is intended to test whether Shipping Intelligence adds predictive
information beyond the 13-feature M0 core.

This block does not train M1. It audits whether candidate features can be
reproduced without violating Point-in-Time constraints.

## Candidate classes

### Freight burden

`B = total_freight / (total_price + total_freight)`

This is derived only from already-released M0 fields. Its artifact parity is
audited separately.

### Expected Freight OOT

Historical out-of-time values are evidence, but they are not by themselves a
production-serving proof. The expected-freight producer and all of its inputs
must be independently reproduced under the purchase-time contract.

### Expected-freight ratios

`Q = total_freight / expected_freight_oot`

Q remains blocked while Expected Freight remains blocked.

The log-ratio feature additionally requires exact epsilon provenance before
formula parity is asserted.

## Coverage

Expected-freight coverage is measured against the exact 96,470-order frozen
Gold supervised cohort. Missing expected-freight rows must not be silently
dropped; a benchmark protocol must explicitly govern missingness.

## Forbidden diagnostic

Critical Shipping Friction contains observed delivery outcome and remains
forbidden as a same-order predictor.

## Current decision

M1 remains unreleased.

The next stage is an independent production reproduction of Expected Freight
and its Point-in-Time dependencies, informed by the source-lineage evidence
collected in this block.
