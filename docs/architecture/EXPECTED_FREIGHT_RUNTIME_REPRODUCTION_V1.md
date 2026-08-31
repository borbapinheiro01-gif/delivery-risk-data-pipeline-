# Expected Freight Runtime Reproduction V1

## Independent implementation

This stage independently reproduced the frozen F2 runtime contract without
calling the historical producer.

### Model

- Features: `['log_distance', 'log_weight', 'log_volume', 'log_price', 'seller_item_count']`
- Target: `log1p(seller_freight)`
- Estimator: `HistGradientBoostingRegressor(max_iter=100, random_state=42)`
- Minimum training months: `3`
- Temporal protocol: expanding monthly out-of-time
- Output transform: `maximum(0, expm1(pred_log_f2))`
- Seller-to-order aggregation: sum with `min_count=1`

## Historical parity

- Historical rows: 96211
- Reproduced rows: 96211
- Historical non-null predictions: 91262
- Reproduced non-null predictions: 91262
- Membership mismatch: 0
- Null-pattern mismatch: 0
- Numeric mismatch: 0
- Maximum absolute difference: 5.684341886080802e-14
- Strict parity: PASS

## Gold supervised serving coverage

- Gold rows: 96470
- Expected Freight available: 91254
- Expected Freight unavailable: 5216
- Available share: 0.9459313776
- Warm-up months: `['2017-01', '2017-02', '2017-03']`
- Unexpected post-warmup missing: 0
- Missing time key: 0

No order is silently dropped.

Orders with Expected Freight available are eligible for M1.
Orders without Expected Freight use the M0 fallback.

## Release

Expected Freight candidate release:
`PASS`

M1 remains unreleased and no late-risk model is trained in this stage.
