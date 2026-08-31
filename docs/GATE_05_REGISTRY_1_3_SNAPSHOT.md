# DQ Gate 05 — Point-in-Time & Registry 1.3

## Estado

- Gate 05: **PASS**
- Registry ativo: **1.3.0**
- Prediction time: `order_purchase_timestamp`
- RAW modificado: **NÃO**
- Silver criada: **NÃO**
- Modelo treinado: **NÃO**

## Regra point-in-time

\[
\boxed{t_{\mathrm{available}}(X_{ij}) \leq t_{0,i}}
\]

com

\[
\boxed{t_{0,i}=\texttt{order\_purchase\_timestamp}_i}
\]

## Target

\[
\boxed{Y_i=\mathbf{1}[\operatorname{date}(T_i^d)>\operatorname{date}(T_i^p)]}
\]

- Pedidos supervisionados: **96.470**
- Atrasos: **6.534**
- On-time: **89.936**
- Prevalência de atraso: **6,77309%**

## Seller History Point-in-Time

Histórico de compras:

\[
\boxed{T_j^{purchase}<T_i^{purchase}}
\]

Histórico dependente de outcome:

\[
\boxed{T_j^{delivery}<T_i^{purchase}}
\]

- Order-seller rows: **97.811**
- Sellers únicos: **2.970**
- Purchase-time leakage: **0**
- Outcome-time leakage: **0**
- Brute-force validation: **300 casos**
- Brute-force mismatches: **0**

## Registry 1.3

- Issues canônicas: **13**
- Resolvidas: **4**
- Não resolvidas: **9**
- OPEN: **8**
- OPEN_LIMITATION: **1**

\[
\boxed{13=4+9}
\]

## Estado modular

- `MODEL_01_ORDER_LOGISTIC`: **UNLOCKED_FOR_TEMPORAL_VALIDATION**
- `MODEL_02_ORDER_CATBOOST`: **LOCKED**
- `MODEL_03_SELLER_EXPERT`: **LOCKED**
- `MODEL_04_GEO_PROMISE_EXPERT`: **LOCKED**
- `MODEL_05_TEMPORAL_EXPERT`: **LOCKED**

## Bloqueios preservados

- Payments: **HOLD_PROVENANCE**
- `shipping_limit_date`: **HOLD_PROVENANCE**
- Coordenadas geográficas: **HOLD_DATA_QUALITY**

## ORDER_CORE_V1

Primeiro conjunto autorizado: **13 features**.

- `item_count`: `COUNT(order_item_id)`
- `unique_product_count`: `COUNT_DISTINCT(product_id)`
- `unique_seller_count`: `COUNT_DISTINCT(seller_id)`
- `total_price`: `SUM(price)`
- `mean_price`: `MEAN(price)`
- `max_price`: `MAX(price)`
- `min_price`: `MIN(price)`
- `price_range`: `MAX(price) - MIN(price)`
- `total_freight`: `SUM(freight_value)`
- `mean_freight`: `MEAN(freight_value)`
- `max_freight`: `MAX(freight_value)`
- `merchandise_plus_freight`: `SUM(price) + SUM(freight_value)`
- `freight_price_ratio`: `SUM(freight_value) / SUM(price)`

O desbloqueio do Modelo 01 permite apenas construção da matriz e validação temporal.

**MODEL_02 permanece bloqueado até PROMOTE / HOLD / REJECT do Modelo 01.**
