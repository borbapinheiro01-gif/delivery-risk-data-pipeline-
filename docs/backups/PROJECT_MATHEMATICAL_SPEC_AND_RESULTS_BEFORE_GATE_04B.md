# Delivery Risk Intelligence Platform

## Mathematical Specification, Data Contract & Experimental State

**Última geração:** 2026-08-30T10:43:24.890813+00:00

> Este documento é gerado automaticamente a partir dos artefatos do projeto. Ele deve ser tratado como o mapa central da formulação matemática, das decisões metodológicas e dos resultados de auditoria.

## 1. Objetivo do sistema

No instante da compra, estimar a probabilidade de que um pedido seja entregue após a promessa de entrega informada ao consumidor.

A saída futura do sistema será um **score de risco de atraso**, utilizado para priorizar ações preventivas.

## 2. Unidade estatística e instante de decisão

A unidade de análise é:

\[
\boxed{\text{1 observação} = \text{1 order\_id}}
\]

O instante de previsão é:

\[
\boxed{t_0 = \text{order\_purchase\_timestamp}}
\]

Portanto, uma feature só é elegível quando sua disponibilidade em produção pode ser demonstrada em ou antes de \(t_0\).

## 3. Notação matemática

Para cada pedido \(i\):

- \(X_i(t_0)\): vetor de informações disponíveis no instante da compra;
- \(T_i^{p}\): timestamp/data prometida de entrega;
- \(T_i^{d}\): timestamp observado de entrega ao cliente;
- \(M_i=T_i^{d}-T_i^{p}\): margem temporal da entrega em relação à promessa;
- \(Y_i\): indicador de violação da promessa.

## 4. Formulações candidatas do target

### 4.1 Definição por timestamp estrito

\[
Y_i^{(\mathrm{ts})}=\mathbf{1}\left(T_i^{d}>T_i^{p}\right)
\]

Nesta definição, qualquer entrega posterior ao instante exato armazenado em `order_estimated_delivery_date` é classificada como atraso.

### 4.2 Definição por dia-calendário

\[
Y_i^{(\mathrm{day})}=\mathbf{1}\left(\operatorname{date}(T_i^{d})>\operatorname{date}(T_i^{p})\right)
\]

Nesta definição, uma entrega realizada durante o próprio dia prometido permanece classificada como on-time.

### 4.3 Sensibilidade com um dia de tolerância

\[
Y_i^{(+1)}=\mathbf{1}\left(\operatorname{date}(T_i^{d})>\operatorname{date}(T_i^{p})+1\right)
\]

Esta terceira formulação é apenas análise de sensibilidade. Ela não possui, neste momento, justificativa empresarial para uso primário.

## 5. Modelo probabilístico

O modelo futuro estima:

\[
\hat p_i=f_\theta\left(X_i(t_0)\right)\approxP\left(Y_i=1\midX_i(t_0)\right)
\]

onde \(\hat p_i\in[0,1]\) representa o risco estimado de atraso.

A decisão operacional poderá usar um threshold \(\tau\):

\[
\hat Y_i=\mathbf{1}\left(\hat p_i\ge\tau\right)
\]

O valor de \(\tau\) não será escolhido por conveniência estatística; deverá refletir capacidade operacional e custo dos erros.

## 6. Função de custo operacional

Uma formulação futura possível é:

\[
C(\tau)=c_{FN}FN(\tau)+c_{FP}FP(\tau)+c_A A(\tau)
\]

onde \(c_{FN}\) representa o custo de não detectar um atraso, \(c_{FP}\) o custo de agir sobre um pedido que não atrasaria, e \(c_A\) o custo da própria intervenção.

## 7. Métricas previstas para os modelos

### Precision

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
\]

### Recall

\[
\mathrm{Recall}=\frac{TP}{TP+FN}
\]

### \(F_\beta\)

\[
F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}
\]

Se a prioridade operacional for capturar atrasos, \(\beta>1\) poderá ser utilizado.

### Brier Score

\[
\mathrm{Brier}=\frac{1}{n}\sum_{i=1}^{n}(\hat p_i-Y_i)^2
\]

### Log Loss

\[
\mathrm{LogLoss}=-\frac1n\sum_i\left[Y_i\log\hat p_i+(1-Y_i)\log(1-\hat p_i)\right]
\]

### Capture@K

\[
\mathrm{Capture@K}=\frac{\text{atrasos presentes nos K pedidos de maior risco}}{\text{total de atrasos}}
\]

A métrica técnica primária prevista para comparação de classificadores é **PR-AUC**, acompanhada de recall, precision e calibração.

## 8. Formulação matemática da qualidade dos dados

### 8.1 Cobertura relacional

\[
C_R=1-\frac{N_{\mathrm{relation\ missing}}}{N_{\mathrm{applicable}}}
\]

### 8.2 Completude condicional de atributo

\[
C_{A\mid R}=1-P\left(A\text{ ausente}\midR\text{ existe}\right)
\]

Portanto:

\[
\boxed{\mathrm{RELATION\_MISSING}\neq\mathrm{ATTRIBUTE\_MISSING}}
\]

\[
\boxed{\mathrm{SOURCE\ QUALITY}\neq\mathrm{TASK\ QUALITY}}
\]

## 9. Estado dos controles de Data Quality

| Componente | Status |
| --- | --- |
| DQ Gate 01 — Structural | PASS |
| DQ Gate 02 — Semantic / Validity | PASS |
| DQ Gate 03 — Statistical / Completeness | PASS |
| Knowledge & Truth Registry 1.1 | PASS |
| DQ Gate 03B — Conditional / Task | PASS |
| DQ Gate 04 — Label / Construct | PASS |

## 10. Dataset Knowledge Registry

- Tabelas mapeadas: **9**
- Colunas mapeadas: **52**
- Relacionamentos: **7**
- Questões brutas: **18**
- Questões canônicas: **13**
- Search records: **181**

### Cardinalidades semânticas e observadas

| parent_table | child_table | semantic_cardinality | observed_cardinality | child_rows_per_parent_mean | child_rows_per_parent_max | child_orphan_rows_recomputed |
| --- | --- | --- | --- | --- | --- | --- |
| customers | orders | 1:1 order-specific customer identity | 1:0..1 OBSERVED | 1.000000 | 1 | 0 |
| orders | order_items | 1:N | 1:N OBSERVED | 1.141731 | 21 | 0 |
| orders | payments | 1:N | 1:N OBSERVED | 1.044710 | 29 | 0 |
| orders | reviews | 1:N possible | 1:N OBSERVED | 1.005584 | 3 | 0 |
| products | order_items | 1:N | 1:N OBSERVED | 3.418713 | 527 | 0 |
| sellers | order_items | 1:N | 1:N OBSERVED | 36.397415 | 2033 | 0 |
| translation | products | 1:N with known translation gaps | 1:N OBSERVED | 443.027397 | 3029 | 13 |

## 11. DQ Gate 02 — warnings semânticos conhecidos

| status | check_id | table | variable | affected_rows | affected_pct | observed |
| --- | --- | --- | --- | --- | --- | --- |
| WARN | SEM-030 | payments | payment_value | 9 | 0.008663 | 9 zero value(s) |
| WARN | SEM-031 | payments | payment_installments | 2 | 0.001925 | 2 zero value(s) |
| WARN | SEM-032 | products | product_weight_g | 4 | 0.012139 | 4 zero value(s) |
| WARN | SEM-051 | orders | approval_after_carrier | 1359 | 1.391791 | 1359 violation(s) |
| WARN | SEM-052 | orders | carrier_after_customer_delivery | 23 | 0.023840 | 23 violation(s) |
| WARN | SEM-060 | products | product_category_name | 13 | 0.040197 | ['pc_gamer', 'portateis_cozinha_e_preparadores_de_alimentos'] |
| WARN | SEM-061 | geolocation | geolocation_zip_code_prefix | 8 | 0.042072 | 8 prefixes associated with more than one state |

## 12. DQ Gate 03 — perfil estatístico numérico

| table | variable | missing_pct | min | median | mean | p99 | max | skewness | iqr_candidate_pct | mad_candidate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| order_items | price | 0.000000 | 0.850000 | 74.990000 | 120.653739 | 890.000000 | 6735.000000 | 7.923208 | 7.480692 | 6.847759 |
| order_items | freight_value | 0.000000 | 0.000000 | 16.260000 | 19.990320 | 84.520000 | 409.680000 | 5.639870 | 10.771416 | 9.129161 |
| payments | payment_value | 0.000000 | 0.000000 | 100.000000 | 154.100380 | 1039.916500 | 13664.080000 | 9.254010 | 7.682460 | 6.822864 |
| payments | payment_installments | 0.000000 | 0.000000 | 1.000000 | 2.853349 | 10.000000 | 24.000000 | 1.655074 | 6.076853 | 0.000000 |
| products | product_weight_g | 0.006070 | 0.000000 | 700.000000 | 2276.472488 | 22538.000000 | 40425.000000 | 3.604860 | 13.812255 | 16.598379 |
| products | product_length_cm | 0.006070 | 7.000000 | 25.000000 | 30.815078 | 100.000000 | 105.000000 | 1.750460 | 4.188291 | 4.561595 |
| products | product_height_cm | 0.006070 | 2.000000 | 13.000000 | 16.937661 | 69.000000 | 105.000000 | 2.140061 | 5.742208 | 4.852955 |
| products | product_width_cm | 0.006070 | 6.000000 | 20.000000 | 23.196728 | 63.000000 | 118.000000 | 1.670971 | 2.767914 | 3.016784 |

Extremos detectados por IQR/MAD permanecem **diagnósticos**, não regras automáticas de exclusão.

## 13. DQ Gate 03B — Conditional & Task Completeness

- Source orders: **99,441**
- Task candidate orders: **96,470**
- Task/source: **97.0123%**
- Critical failures: **0**
- Warnings: **8**

### Feature readiness da coorte

| readiness_dimension | task_orders | ready_orders | not_ready_orders | ready_pct |
| --- | --- | --- | --- | --- |
| has_items | 96470 | 96470 | 0 | 100.000000 |
| all_item_core_complete | 96470 | 96470 | 0 | 100.000000 |
| customer_resolved | 96470 | 96470 | 0 | 100.000000 |
| core_purchase_context_complete | 96470 | 96470 | 0 | 100.000000 |
| product_metadata_complete | 96470 | 95077 | 1393 | 98.556028 |
| seller_reference_complete | 96470 | 96470 | 0 | 100.000000 |
| payment_data_complete | 96470 | 96469 | 1 | 99.998963 |
| customer_geo_reference_complete | 96470 | 96206 | 264 | 99.726340 |
| seller_geo_reference_complete | 96470 | 96253 | 217 | 99.775060 |
| geo_reference_complete | 96470 | 95990 | 480 | 99.502436 |

### Taxonomia de ausência

| component | scope | state | count | denominator | pct | condition | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| order_items | TASK_ORDER | RELATION_MISSING | 0 | 96470 | 0.000000 | task order exists | No child order_item rows. |
| payments | TASK_ORDER | RELATION_MISSING | 1 | 96470 | 0.001037 | task order exists | No child payment rows. |
| customer | TASK_ORDER | RELATION_MISSING | 0 | 96470 | 0.000000 | task order exists | customer_id does not resolve. |
| customer_geolocation | TASK_ORDER | RELATION_MISSING | 264 | 96470 | 0.273660 | customer_id resolves | Customer exists, but ZIP has no geolocation reference. |
| product | TASK_ITEM | RELATION_MISSING | 0 | 110189 | 0.000000 | order item exists | product_id does not resolve. |
| product_category | TASK_ITEM | ATTRIBUTE_MISSING | 1537 | 110189 | 1.394876 | product_id resolves | Entity exists; category attribute is NULL. |
| product_physical | TASK_ITEM | ATTRIBUTE_MISSING | 18 | 110189 | 0.016336 | product_id resolves | Entity exists; one or more physical attributes are NULL. |
| seller | TASK_ITEM | RELATION_MISSING | 0 | 110189 | 0.000000 | order item exists | seller_id does not resolve. |
| seller_geolocation | TASK_ITEM | RELATION_MISSING | 249 | 110189 | 0.225975 | seller_id resolves | Seller exists, but ZIP has no geolocation reference. |
| order_approved_at | TASK_ORDER | ATTRIBUTE_MISSING | 14 | 96470 | 0.014512 | task order exists | Post-purchase field; forbidden as baseline t0 feature. |
| order_delivered_carrier_date | TASK_ORDER | ATTRIBUTE_MISSING | 1 | 96470 | 0.001037 | task order exists | Future process field; forbidden as baseline t0 feature. |

### Source quality vs task quality

| metric | source_denominator | source_affected | source_affected_pct | task_denominator | task_affected | task_affected_pct | difference_task_minus_source_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| orders_without_items | 99441 | 775 | 0.779357 | 96470 | 0 | 0.000000 | -0.779357 |
| orders_without_payments | 99441 | 1 | 0.001006 | 96470 | 1 | 0.001037 | 0.000031 |
| orders_without_customer_resolution | 99441 | 0 | 0.000000 | 96470 | 0 | 0.000000 | 0.000000 |
| item_product_orphans | 112650 | 0 | 0.000000 | 110189 | 0 | 0.000000 | 0.000000 |
| item_seller_orphans | 112650 | 0 | 0.000000 | 110189 | 0 | 0.000000 | 0.000000 |

### Estrutura multi-entidade

| metric | orders | pct | maximum |
| --- | --- | --- | --- |
| orders_multi_item | 9635 | 9.987561 | 21.000000 |
| orders_multi_product | 3197 | 3.313984 | 8.000000 |
| orders_multi_seller | 1275 | 1.321654 | 5.000000 |
| orders_multi_payment_record | 2875 | 2.980201 | 26.000000 |
| orders_multi_payment_type | 2182 | 2.261843 | 2.000000 |

## 14. DQ Gate 04 — Label / Construct Validity

- Status: **PASS**
- Observation end: **2018-10-17 13:22:46**
- Estimated timestamps at midnight: **100.000000%**
- Label disagreements: **1,292** (**1.339276%**)
- Same-day disagreements: **1,292** (**100.000000% dos desacordos**)
- Construct recommendation: **CALENDAR_DAY_RECOMMENDED_PENDING_BUSINESS_CONFIRMATION**
- Label finalizado automaticamente: **False**

### Comparação das definições

| label_definition | role | orders | late_orders | on_time_orders | late_rate_pct | late_rate_ci95_low_pct | late_rate_ci95_high_pct | no_skill_pr_auc | majority_class_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TIMESTAMP_STRICT | candidate | 96470 | 7826 | 88644 | 8.112367 | 7.941742 | 8.286327 | 0.081124 | 0.918876 |
| CALENDAR_DAY | candidate | 96470 | 6534 | 89936 | 6.773090 | 6.616237 | 6.933386 | 0.067731 | 0.932269 |
| CALENDAR_DAY_PLUS_1_GRACE | sensitivity_only | 96470 | 5709 | 90761 | 5.917902 | 5.770752 | 6.068563 | 0.059179 | 0.940821 |

### Matriz de concordância

| timestamp_strict_label | calendar_day_label | count | pct_task |
| --- | --- | --- | --- |
| 0.000000 | 0.000000 | 88644.000000 | 91.887633 |
| 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| 1.000000 | 0.000000 | 1292.000000 | 1.339276 |
| 1.000000 | 1.000000 | 6534.000000 | 6.773090 |

### Distribuição da margem

| variable | metric | value |
| --- | --- | --- |
| lateness_hours | min | -3504.386944 |
| lateness_hours | p01 | -843.468008 |
| lateness_hours | p05 | -622.551222 |
| lateness_hours | p25 | -389.857569 |
| lateness_hours | median | -286.754444 |
| lateness_hours | mean | -268.275017 |
| lateness_hours | p75 | -153.355556 |
| lateness_hours | p95 | 91.607833 |
| lateness_hours | p99 | 454.549267 |
| lateness_hours | max | 4535.401944 |
| lateness_hours | std | 244.424494 |
| lateness_calendar_days | min | -147.000000 |
| lateness_calendar_days | p01 | -36.000000 |
| lateness_calendar_days | p05 | -26.000000 |
| lateness_calendar_days | p25 | -17.000000 |
| lateness_calendar_days | median | -12.000000 |
| lateness_calendar_days | mean | -11.875889 |
| lateness_calendar_days | p75 | -7.000000 |
| lateness_calendar_days | p95 | 3.000000 |
| lateness_calendar_days | p99 | 18.000000 |
| lateness_calendar_days | max | 188.000000 |
| lateness_calendar_days | std | 10.182105 |

### Faixas de adiantamento/atraso

| lateness_bucket | orders | pct_task |
| --- | --- | --- |
| <= -8 days | 71303 | 73.912097 |
| -7 to -2 days | 15879 | 16.460039 |
| -1 day | 1462 | 1.515497 |
| 0 days / same promised day | 1292 | 1.339276 |
| +1 day | 825 | 0.855188 |
| +2 to +3 days | 1045 | 1.083238 |
| +4 to +7 days | 1802 | 1.867938 |
| +8 to +14 days | 1478 | 1.532083 |
| >= +15 days | 1384 | 1.434643 |

### Janela de observação

| observation_class | orders | pct_source | observation_end |
| --- | --- | --- | --- |
| OBSERVED_DELIVERED | 96470 | 97.012299 | 2018-10-17 13:22:46 |
| OUTCOME_UNOBSERVED_BY_DATASET_END | 1737 | 1.746764 | 2018-10-17 13:22:46 |
| NOT_APPLICABLE_TERMINAL_NON_DELIVERY | 1228 | 1.234903 | 2018-10-17 13:22:46 |
| OUTCOME_PRESENT_STATUS_MISMATCH | 6 | 0.006034 | 2018-10-17 13:22:46 |

### Label Contract — candidatos

| candidate | formula | recommended_role | construct_status |
| --- | --- | --- | --- |
| TIMESTAMP_STRICT | 1[actual_timestamp > estimated_timestamp] | sensitivity / comparison | NOT_PRIMARY_RECOMMENDATION |
| CALENDAR_DAY | 1[date(actual) > date(estimated)] | primary candidate | CALENDAR_DAY_RECOMMENDED_PENDING_BUSINESS_CONFIRMATION |
| CALENDAR_DAY_PLUS_1_GRACE | 1[date(actual) > date(estimated) + 1 day] | sensitivity only | NO BUSINESS JUSTIFICATION FOR PRIMARY USE |

## 15. Campos proibidos no baseline em t0

`order_status` final, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date` e reviews não podem ser utilizados como features do modelo preventivo em `order_purchase_timestamp`.

`shipping_limit_date` permanece em HOLD até confirmação de sua disponibilidade operacional em t0.

Os campos de pagamento permanecem candidatos sujeitos à confirmação de point-in-time availability.

## 16. Questões canônicas abertas

| canonical_issue_id | canonical_topic | priority | source_issue_count | source_issue_ids | question_summary | required_actions |
| --- | --- | --- | --- | --- | --- | --- |
| ISSUE-FINANCIAL-001 | financial_reconciliation | MEDIUM | 2 | OPEN-008;AUD03-017 | Qual a semântica dos pedidos com diferença relevante entre pagamentos e preço+frete? \| Revisar a distribuição de payment_delta antes de qualquer correção ou exclusão. | Inspecionar sem alterar valores RAW. \| Review before final treatment. |
| ISSUE-GEO-001 | geolocation_consolidation | MEDIUM | 2 | OPEN-007;AUD03-018 | Como consolidar CEPs com observações em múltiplos estados ou grande dispersão? \| Inspecionar CEPs com vários estados/cidades ou grande dispersão antes da consolidação definitiva. | Inspecionar exceções antes da consolidação definitiva. \| Review before final treatment. |
| ISSUE-GROUND-TRUTH-001 | external_ground_truth | HIGH | 1 | OPEN-004 | É possível validar datas/valores contra sistemas operacionais originais da Olist? | Documentar limite: internal consistency != external ground truth. |
| ISSUE-IMPUTATION-001 | numeric_imputation_policy | MEDIUM | 1 | AUD03-014 | Método e estatísticas somente depois do split. Fit apenas no conjunto de treino. | Review before final treatment. |
| ISSUE-MISSINGNESS-001 | conditional_and_task_missingness | HIGH | 1 | OPEN-003 | Qual a completude dos atributos condicionada à existência da entidade relacionada? | Executar DQ Gate 03B Conditional & Task Completeness. |
| ISSUE-MULTI-SELLER-001 | multi_seller_representation | MEDIUM | 2 | OPEN-006;AUD03-016 | Como representar pedidos contendo múltiplos sellers? \| Definir regra definitiva para sellers e distâncias quando um pedido possui vários sellers. | Comparar agregações min/mean/max/count e regra operacional. \| Review before final treatment. |
| ISSUE-NUMERIC-TRANSFORMATION-001 | numeric_transformation_policy | MEDIUM | 1 | AUD03-013 | Log, clipping ou winsorization somente após EDA. Se adotados, parâmetros serão ajustados somente no treino. | Review before final treatment. |
| ISSUE-OBSERVATION-WINDOW-001 | label_observation_window | HIGH | 1 | OPEN-009 | Como representar setembro/outubro de 2018 na população supervisionada? | Tratar como janela não madura/censura potencial; não converter ausência de label em on-time. |
| ISSUE-OUTLIERS-001 | extreme_value_policy | MEDIUM | 1 | AUD03-012 | Investigar extremos plausíveis e suspeitos. Nenhuma exclusão automática. | Review before final treatment. |
| ISSUE-PAYMENT-AVAILABILITY-001 | payment_point_in_time_availability | MEDIUM | 1 | OPEN-010 | payment_type/value/installments são operacionalmente disponíveis no exato t0 definido como purchase_timestamp? | Manter como candidate e documentar hipótese. |
| ISSUE-SELLER-HISTORY-001 | seller_point_in_time_history | HIGH | 1 | OPEN-005 | Como construir seller history sem conhecer outcomes futuros? | Usar somente eventos cujo outcome estava observável antes do pedido atual; definir smoothing. |
| ISSUE-SHIPPING-LIMIT-001 | shipping_limit_date_provenance | HIGH | 2 | OPEN-002;AUD03-015 | shipping_limit_date era conhecido exatamente no instante da compra? \| Confirmar se a informação existia no instante t0 antes de liberar como feature. | Manter fora do baseline até confirmação. \| Review before final treatment. |
| ISSUE-TARGET-001 | label_and_target_semantics | CRITICAL | 2 | OPEN-001;AUD03-011 | A promessa deve ser avaliada por timestamp exato ou por dia-calendário? \| Comparar timestamp_strict versus calendar_date e decidir a semântica correta da promessa. | Formalizar Label Contract e análise de sensibilidade. \| Review before final treatment. |

## 17. Decisões metodológicas atualmente estabelecidas

1. A unidade estatística do modelo é `order_id`.
2. O instante de previsão base é `order_purchase_timestamp`.
3. Child tables (`order_items`, `payments`) precisam ser agregadas antes da modelagem em nível de pedido.
4. Missing de relação e missing de atributo são fenômenos diferentes.
5. Source Quality e Task Quality são medidas separadamente.
6. Informação futura não entra no baseline mesmo que seja 100% completa.
7. Resultado não observado não é convertido automaticamente em `on-time`.
8. O label final requer contrato explícito de construct validity.

## 18. Proveniência e limites da verdade observável

Os valores existentes nos CSVs constituem a **source truth da distribuição pública**, mas não temos acesso aos sistemas transacionais privados originais para confirmação independente.

\[
\boxed{\text{internal consistency}\neq\text{externally verified ground truth}}
\]

## 19. Referências metodológicas verificadas

- **Olist — Brazilian E-Commerce Public Dataset:** dataset comercial anonimizado; a data estimada é informada ao cliente no momento da compra; pedidos podem possuir múltiplos itens e sellers.
- **NIST AI Risk Management Framework / Playbook, Measure 2.5:** proxies e indicadores devem demonstrar construct validity, isto é, medir o conceito que afirmam representar.
- **Survival Analysis / time-to-event methodology:** quando o evento não pode ser observado até o final da janela de acompanhamento, a observação não deve ser simplesmente transformada em evento negativo.
- **Croissant / MLCommons:** metadata de datasets deve tornar estrutura, recursos e semântica legíveis por máquina e reutilizáveis.
- **Conditional data validation:** regras de qualidade devem ser avaliadas sobre a população em que são semanticamente aplicáveis.

## 20. Artefatos centrais do projeto

- `metadata/column_catalog.csv`
- `metadata/truth_provenance_registry.csv`
- `metadata/temporal_availability_registry.csv`
- `metadata/registry_v1_1/canonical_issue_registry.csv`
- `metadata/registry_v1_1/ml_grain_policy_registry.csv`
- `metadata/registry_v1_1/croissant_1_1_FIXED.json`
- `metadata/registry_v1_1/croissant_1_1_MLCROISSANT_COMPAT.json`
- `reports/data_quality/gate_03b_conditional_task/DQ_GATE_03B_CONDITIONAL_TASK_REPORT.txt`
- `reports/data_quality/gate_04_label_construct/DQ_GATE_04_LABEL_CONSTRUCT_REPORT.txt`
- `reports/data_quality/gate_04_label_construct/dq_gate_04_summary.json`
- `docs/PROJECT_MATHEMATICAL_SPEC_AND_RESULTS.md`

## 21. Próximas decisões

A sequência metodológica após o Gate 04 é:

```text
Label Contract final
        ↓
Point-in-Time / Leakage Gate
        ↓
Representativeness / Shift
        ↓
Data Treatment Plan
        ↓
Silver order-level
        ↓
Feature Engineering
        ↓
Temporal Train / Validation / Test
        ↓
Baseline + ML Models
        ↓
Calibration + Business Threshold
        ↓
Deploy / Monitoring / MLOps
```
