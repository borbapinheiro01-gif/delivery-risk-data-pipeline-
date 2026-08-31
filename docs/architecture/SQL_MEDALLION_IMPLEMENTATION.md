# SQL / Medallion Implementation

## Architecture

Olist RAW
→ BRONZE
→ SILVER
→ GOLD
→ Production Feature Contract
→ Late Risk M0 vs M1

## Bronze

Preserves source semantics and row multiplicity. No silent data repair.

## Silver

Creates typed and validated reusable entities while preserving missingness,
Point-in-Time availability, and outcome/predictor distinctions.

## Gold

Creates one-row-per-order product entities.

First predictive comparison:

M0 = Core Point-in-Time features

M1 = Core Point-in-Time features + Point-in-Time-safe Shipping Intelligence

## Guardrail

Critical Shipping Friction contains observed delivery outcome and therefore is
diagnostic, not a same-order purchase-time predictor.
