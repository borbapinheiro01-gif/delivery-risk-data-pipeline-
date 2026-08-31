# Data Card — Olist Brazilian E-Commerce

## 1. Identificação

**Projeto:** Delivery Risk Intelligence Platform

**Dataset:** Brazilian E-Commerce Public Dataset by Olist

**Identificador da distribuição:** `olistbr/brazilian-ecommerce`

**Arquivos RAW:** 9

**Colunas observadas:** 52

**Pedidos na tabela central:** 99,441

**Fingerprint composto SHA-256:** `3a2d46493a1d746f1840068264755660a30705a7ff80475eaee89b470387a7e3`


## 2. Origem e natureza

A base utilizada é a distribuição pública da Olist.

O projeto considera os arquivos RAW como **Source Truth do dataset público**:
eles informam o que está registrado na distribuição, mas não fornecem acesso
aos sistemas transacionais privados originais.

Consequentemente:

> internal consistency != independently verified external ground truth


## 3. Uso pretendido

Problema principal:

> No momento da compra, estimar o risco de que o pedido seja entregue depois
> da promessa apresentada ao consumidor.

Instante de decisão candidato:

`order_purchase_timestamp`

Granularidade futura da tabela de ML:

`1 row = 1 order_id`


## 4. Estrutura

| Tabela | Linhas | Colunas | Grão |
|---|---:|---:|---|
| customers | 99,441 | 5 | 1 row per order-specific customer_id |
| geolocation | 1,000,163 | 5 | 1 row per geolocation observation for ZIP prefix |
| order_items | 112,650 | 7 | 1 row per item position inside an order |
| payments | 103,886 | 5 | 1 row per payment sequence |
| reviews | 99,224 | 7 | 1 row per review record |
| orders | 99,441 | 8 | 1 row per order |
| products | 32,951 | 9 | 1 row per product_id |
| sellers | 3,095 | 4 | 1 row per seller_id |
| translation | 71 | 2 | 1 row per Portuguese product category |


## 5. Relacionamentos

- `customers.customer_id` → `orders.customer_id` | 1:1 order-specific customer identity | órfãos observados: 0 | many_to_one from orders perspective
- `orders.order_id` → `order_items.order_id` | 1:N | órfãos observados: 0 | aggregate child before order-level ML join
- `orders.order_id` → `payments.order_id` | 1:N | órfãos observados: 0 | aggregate child before order-level ML join
- `orders.order_id` → `reviews.order_id` | 1:N possible | órfãos observados: 0 | keep outside baseline risk features
- `products.product_id` → `order_items.product_id` | 1:N | órfãos observados: 0 | many_to_one item -> product
- `sellers.seller_id` → `order_items.seller_id` | 1:N | órfãos observados: 0 | many_to_one item -> seller
- `translation.product_category_name` → `products.product_category_name` | 1:N with known translation gaps | órfãos observados: 13 | LEFT JOIN only


## 6. Data Quality Gates

- **gate_01_structural:** `PASS`
- **gate_02_semantic:** `PASS`
- **gate_03_statistical:** `PASS`

### Observações não-PASS registradas

- `gate_01_structural` / `geolocation-ROWDUP` — geolocation.nan — **WARN**
- `gate_02_semantic` / `SEM-030` — payments.payment_value — **WARN**
- `gate_02_semantic` / `SEM-031` — payments.payment_installments — **WARN**
- `gate_02_semantic` / `SEM-032` — products.product_weight_g — **WARN**
- `gate_02_semantic` / `SEM-051` — orders.approval_after_carrier — **WARN**
- `gate_02_semantic` / `SEM-052` — orders.carrier_after_customer_delivery — **WARN**
- `gate_02_semantic` / `SEM-060` — products.product_category_name — **WARN**
- `gate_02_semantic` / `SEM-061` — geolocation.geolocation_zip_code_prefix — **WARN**


## 7. Verdade e proveniência

O registry distingue:

1. **Source truth** — o valor existe no arquivo público.
2. **Structural truth** — entidades/chaves/relações são estruturalmente válidas.
3. **Semantic truth** — valores respeitam regras de domínio.
4. **Process truth** — o evento é compatível com o estágio operacional.
5. **Measurement truth** — a variável mede adequadamente o fenômeno pretendido.
6. **ML task truth** — a informação seria realmente utilizável em `order_purchase_timestamp`.
7. **External ground truth** — confirmação contra o mundo/sistema fonte.

O nível 7 não está disponível no dataset público.


## 8. Política de disponibilidade temporal

O arquivo:

`temporal_availability_registry.csv`

é a fonte central para definir quais variáveis podem entrar em modelos
no instante de compra.

Uma coluna existir no CSV final **não implica** que estivesse disponível em t0.


## 9. Missingness

Missing deverá ser classificado por mecanismo operacional, evitando que todos
os casos sejam reduzidos a um único NaN.

Taxonomia planejada:

- `OBSERVED`
- `ATTRIBUTE_MISSING`
- `RELATION_MISSING`
- `NOT_APPLICABLE`
- `NOT_YET_OBSERVED`
- `CENSORED`
- `UNKNOWN_PROVENANCE`


## 10. Label

A definição definitiva de atraso permanece sujeita ao Label Contract.

Alternativas já identificadas:

- timestamp estrito;
- dia-calendário.

A diferença entre essas definições deve ser tratada como questão de
**construct validity**, não como simples detalhe de implementação.


## 11. Leakage

São proibidas como features do baseline em t0 informações claramente futuras,
incluindo:

- entrega ao carrier;
- entrega ao cliente;
- reviews;
- estado final do pedido.

Campos cuja disponibilidade em t0 não foi comprovada permanecem em `HOLD`.


## 12. Limitações conhecidas

- dados anonimizados;
- ausência de acesso aos sistemas operacionais fonte;
- anomalias temporais já documentadas;
- múltiplos itens por pedido;
- múltiplos sellers possíveis;
- múltiplos registros de pagamento;
- geolocation com múltiplas observações por CEP;
- últimos meses sem maturidade adequada do outcome supervisionado;
- availability time não explicitamente fornecido para todas as variáveis.


## 13. Questões abertas

- **OPEN-001 — target_semantics**: A promessa deve ser avaliada por timestamp exato ou por dia-calendário?
- **OPEN-002 — shipping_limit_date_provenance**: shipping_limit_date era conhecido exatamente no instante da compra?
- **OPEN-003 — conditional_missingness**: Qual a completude dos atributos condicionada à existência da entidade relacionada?
- **OPEN-004 — external_ground_truth**: É possível validar datas/valores contra sistemas operacionais originais da Olist?
- **OPEN-005 — seller_history**: Como construir seller history sem conhecer outcomes futuros?
- **OPEN-006 — multi_seller**: Como representar pedidos contendo múltiplos sellers?
- **OPEN-007 — geolocation**: Como consolidar CEPs com observações em múltiplos estados ou grande dispersão?
- **OPEN-008 — financial_reconciliation**: Qual a semântica dos pedidos com diferença relevante entre pagamentos e preço+frete?
- **OPEN-009 — observation_window**: Como representar setembro/outubro de 2018 na população supervisionada?
- **OPEN-010 — payment_availability**: payment_type/value/installments são operacionalmente disponíveis no exato t0 definido como purchase_timestamp?
- **AUD03-011 — target_definition**: Comparar timestamp_strict versus calendar_date e decidir a semântica correta da promessa.
- **AUD03-012 — outliers**: Investigar extremos plausíveis e suspeitos. Nenhuma exclusão automática.
- **AUD03-013 — numeric_transformations**: Log, clipping ou winsorization somente após EDA. Se adotados, parâmetros serão ajustados somente no treino.
- **AUD03-014 — numeric_imputation**: Método e estatísticas somente depois do split. Fit apenas no conjunto de treino.
- **AUD03-015 — shipping_limit_date**: Confirmar se a informação existia no instante t0 antes de liberar como feature.
- **AUD03-016 — multi_seller_orders**: Definir regra definitiva para sellers e distâncias quando um pedido possui vários sellers.
- **AUD03-017 — payment_inconsistencies**: Revisar a distribuição de payment_delta antes de qualquer correção ou exclusão.
- **AUD03-018 — geolocation_spread**: Inspecionar CEPs com vários estados/cidades ou grande dispersão antes da consolidação definitiva.


## 14. Artefatos de conhecimento

- `dataset_manifest.json`
- `table_catalog.csv`
- `column_catalog.csv`
- `relationship_registry.csv`
- `truth_provenance_registry.csv`
- `temporal_availability_registry.csv`
- `value_domain_registry.csv`
- `quality_observation_registry.csv`
- `unresolved_questions.csv`
- `search_knowledge.jsonl`


## 15. Regra de manutenção

Este documento deve evoluir junto com o dataset e com as decisões do projeto.

Alterações em:

- target;
- t0;
- disponibilidade das features;
- tratamento;
- definição de entidades;
- conhecimento de proveniência;
- novos problemas de qualidade;

devem ser refletidas nos registries correspondentes.
