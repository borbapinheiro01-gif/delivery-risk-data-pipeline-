# Functional Representation Decision — D2

## 1. Estado metodológico

**Decision ID:** `FUNCTIONAL_REPRESENTATION_D2_V1`

**Status:** `PASS`

**Representação primária para a próxima etapa:** `RAW-30D`

**Smoothing challenger:** `SMOOTH-30D-EDF0.75` — `REJECTED_AS_PRIMARY_UNSUPERVISED_PREPROCESSING`

Esta decisão é restrita à representação funcional não supervisionada e à fidelidade temporal da curva original. Ela não constitui prova de superioridade preditiva para a classificação de atraso.

## 2. Contrato point-in-time

Para cada pedido i, o instante de previsão é:

\[
t_{0,i}=T_i^{purchase}.
\]

Um evento histórico j somente pode participar da representação se:

\[
T_j^{purchase}<T_i^{purchase}.
\]

Assim, cada curva representa exclusivamente informação disponível antes do instante de previsão do pedido corrente.

## 3. Representação funcional RAW-30D

Para uma janela de 30 dias, cada pedido é associado a uma curva discreta:

\[
H_i=\left(h_i(1),\ldots,h_i(30)\right).
\]

A matriz centrada no conjunto de treino é:

\[
X_R=H-\mathbf{1}\mu_R^T.
\]

Sua decomposição SVD é:

\[
X_R=U_R\Sigma_RV_R^T.
\]

Com K componentes, a reconstrução é:

\[
\widehat H_R
=
\mu_R+(H-\mu_R)V_{R,K}V_{R,K}^{T}.
\]

O erro temporal utilizado é:

\[
RE_R
=
\frac{\left\|H_{test}-\widehat H_R\right\|_F}{\left\|H_{test}-\mu_R\right\|_F}.
\]

## 4. Representação suavizada avaliada

O challenger foi o suavizador de segunda diferença:

\[
S_\lambda
=
\left(I+\lambda D_2^TD_2\right)^{-1}.
\]

A curva suavizada é:

\[
H^S=HS_\lambda.
\]

A configuração avaliada foi:

\[
EDF=22.5,\qquad EDF/30=0.75,
\]

com

\[
\lambda=0.0840662845049379.
\]

## 5. Comparação temporal justa

A PCA/SVD foi ajustada somente com pedidos anteriores ao mês de teste. O número K90 também foi aprendido somente no passado.

Além do erro interno no espaço suavizado, foi calculado o erro end-to-end contra a curva RAW futura:

\[
RE_{S\rightarrow R}
=
\frac{\left\|H_{test}-\widehat H_S\right\|_F}{\left\|H_{test}-\mu_R\right\|_F}.
\]

A diferença pareada por fold temporal é:

\[
\Delta_m
=
RE_{S\rightarrow R,m}-RE_{R,m}.
\]

Quando Delta_m > 0, o smoothing possui maior erro end-to-end que a representação RAW naquele fold.

## 6. Resultados empíricos

Foram avaliadas 34 comparações temporais pareadas:

\[
34=17\times2,
\]

correspondentes a 17 períodos futuros para cada um dos dois canais.

### 6.1 Purchase freight

Testes temporais: 17

RAW wins: 17/17

SMOOTH wins: 0/17

Erro RAW futuro médio: 0.3100039582

Erro SMOOTH->RAW futuro médio: 0.3752674090

Delta médio: 0.0652634508

Razão média SMOOTH/RAW: 1.2396055420

K90 mediano: RAW 13 -> SMOOTH 7

### 6.2 Purchase volume

Testes temporais: 17

RAW wins: 17/17

SMOOTH wins: 0/17

Erro RAW futuro médio: 0.3055432296

Erro SMOOTH->RAW futuro médio: 0.3650761651

Delta médio: 0.0595329355

Razão média SMOOTH/RAW: 1.2439387775

K90 mediano: RAW 10 -> SMOOTH 6

## 7. Resultado conjunto

Nos 34 pares:

\[
\Delta_m>0
\qquad
\text{para }34/34\text{ comparações}.
\]

Ao mesmo tempo:

\[
K_{90}^{Smooth}<K_{90}^{RAW}
\qquad
\text{em }34/34\text{ comparações}.
\]

Portanto, o smoothing produziu maior compressão dimensional, mas menor fidelidade end-to-end em relação às curvas RAW futuras.

## 8. Decisão

\[
\boxed{\text{RAW-30D = PRIMARY FOR NEXT FUNCTIONAL STAGE}}
\]

`SMOOTH-30D-EDF0.75` é rejeitado como pré-processamento funcional não supervisionado primário.

A decisão não exclui a possibilidade de alguma transformação suavizada ser útil futuramente em um experimento supervisionado.

## 9. Correção metodológica do D2 original

O D2 original foi preservado com status FAIL por provenance. A única falha era a exigência de que os subconjuntos temporais futuros apresentassem valores negativos após smoothing.

D1.1 demonstrou negativos na população global auditada, enquanto os folds futuros do D2 apresentaram zero valores negativos. Essas observações não são contraditórias.

A validação metodológica V2 corrigiu a interpretação sem alterar os resultados originais do D2.

## 10. Fundamentação teórica

O smoothing por diferenças penalizadas é fundamentado como um problema de mínimos quadrados penalizados, no qual o parâmetro de regularização controla explicitamente o compromisso entre fidelidade e suavidade.

A literatura de Functional Data Analysis trata a análise de componentes principais funcionais como ferramenta de redução dimensional das curvas e distingue situações de observação densa, regular, esparsa e ruidosa.

No presente problema, as curvas são construídas deterministicamente em uma grade diária regular a partir dos eventos históricos PIT. Não foi estabelecido um modelo explícito de erro de medição que permitisse interpretar automaticamente a estrutura removida pelo smoothing como ruído.

A escolha entre RAW e smoothing foi, portanto, submetida a uma comparação temporal out-of-sample com múltiplas origens/períodos.

## 11. Referências bibliográficas

1. EILERS, P. H. C. A Perfect Smoother. Analytical Chemistry, v. 75, n. 14, p. 3631-3636, 2003. DOI: 10.1021/ac034173t.

2. RAMSAY, J. O.; SILVERMAN, B. W. Functional Data Analysis. 2. ed. New York: Springer, 2005. DOI: 10.1007/b98888.

3. HALL, P.; MÜLLER, H.-G.; WANG, J.-L. Properties of Principal Component Methods for Functional and Longitudinal Data Analysis. The Annals of Statistics, v. 34, n. 3, p. 1493-1517, 2006. DOI: 10.1214/009053606000000272.

4. WANG, J.-L.; CHIOU, J.-M.; MÜLLER, H.-G. Functional Data Analysis. Annual Review of Statistics and Its Application, v. 3, p. 257-295, 2016. DOI: 10.1146/annurev-statistics-041715-033624.

5. TASHMAN, L. J. Out-of-sample Tests of Forecasting Accuracy: An Analysis and Review. International Journal of Forecasting, v. 16, n. 4, p. 437-450, 2000. DOI: 10.1016/S0169-2070(00)00065-0.

6. BERGMEIR, C.; HYNDMAN, R. J.; KOO, B. A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction. Computational Statistics & Data Analysis, v. 120, p. 70-83, 2018. DOI: 10.1016/j.csda.2017.11.003.

## 12. Limites da conclusão

- Target utilizado nesta decisão: NÃO.
- Superioridade preditiva afirmada: NÃO.
- K final selecionado: NÃO.
- Folds finais congelados: NÃO.
- Classificador treinado: NÃO.
- Silver criada: NÃO.
- RAW modificado: NÃO.

## 13. Próxima etapa

A representação que avança é `RAW-30D`. A próxima questão metodológica é definir uma política temporal para o número de componentes K, aprendida somente com informação disponível no passado.

