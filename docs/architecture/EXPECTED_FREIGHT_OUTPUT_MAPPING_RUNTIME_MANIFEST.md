# Expected Freight — Output Mapping Runtime Manifest

## Final source resolution

The Expected Freight producer uses a generic transformation loop over
`["f0", "f1", "f2"]`.

For each model name, the code creates:

`pred_freight_{name} = max(0, expm1(pred_log_{name}))`.

Instantiating the generic source expression at `name="f2"` yields:

`pred_freight_f2 = max(0, expm1(pred_log_f2))`.

The downstream production policy is then explicit and unique:

`pred_freight_oot = pred_freight_f2`.

Therefore the final source chain is:

1. F2 produces `pred_log_f2`;
2. generic transform instantiated at F2 produces `pred_freight_f2`;
3. F2 is hard-wired as `pred_freight_oot`;
4. seller-level predictions are summed by `order_id`;
5. the result is `expected_freight_oot`.

The `metrics_f` table evaluates F0/F1/F2 but does not select the production
column. No global metric-driven selector is used.

## Exact production specification

- Input: `01_ROUTE_SELLER_ORDER.csv`
- Features:
  - `log1p(great_circle_distance_km)`
  - `log1p(seller_weight_g)`
  - `log1p(seller_volume_proxy_cm3)`
  - `log1p(seller_price)`
  - `seller_item_count`
- Target: `log1p(seller_freight)`
- Model: `HistGradientBoostingRegressor(max_iter=100, random_state=42)`
- Minimum training months: `3`
- Validation: expanding monthly OOT
- Output transform: `maximum(0, expm1(pred_log_f2))`
- Aggregation: seller predictions summed to order level.

Expected Freight is still not released. The next gate is independent runtime
reproduction and row-level parity against the historical OOT artifact.
