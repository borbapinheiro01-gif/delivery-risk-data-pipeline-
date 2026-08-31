# Delivery Risk Intelligence

End-to-end data science and decision-intelligence project for delivery-risk
prediction using the Brazilian E-Commerce Public Dataset by Olist.

## Objective

At order purchase time, estimate the risk that a delivery will occur after
the promised delivery date.

Prediction time:

\[
t_0 = \texttt{order\_purchase\_timestamp}
\]

Primary target:

\[
Y_i =
\mathbf{1}
\left[
\operatorname{date}(T_i^{delivery})
>
\operatorname{date}(T_i^{estimated})
\right]
\]

## Main analytical grain

\[
1\ observation = 1\ order\_id
\]

## Architecture

The repository contains:

- data-quality and semantic governance;
- point-in-time feature governance;
- ORDER_CORE_V1 feature construction;
- temporal-drift diagnostics;
- functional historical curves;
- PCA / smoothing / BCV / temporal bootstrap experiments;
- supervised temporal Model 01;
- spatiotemporal logistics analysis;
- expected-freight analysis;
- GN / GN-ADMM delivery experiments;
- Shipping Friction analysis;
- scientific evidence freezes;
- reproducibility and migration manifests.

## Reproducibility status

Migration audit:

- Python inventory: 125
- historical/legacy exclusions: 5
- publishable Python source: 120
- compile PASS: 120
- compile FAIL: 0
- required pipeline entries missing: 0

Formal status:

`SOURCE_TREE_READY_FOR_PLATFORM_MIGRATION`

## Data

The original Olist RAW layer is intentionally not mixed with source code.

See:

`data/README.md`

and:

`docs/EXTERNAL_AND_LARGE_FILE_MANIFEST.csv`

for provenance, sizes and hashes.

## Documentation

See `docs/` and `reports/migration_audit/` for the detailed methodological,
scientific and reproducibility records.
