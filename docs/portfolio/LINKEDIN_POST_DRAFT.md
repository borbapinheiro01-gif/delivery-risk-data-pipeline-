# LinkedIn Post Draft

Transformei uma base pública de e-commerce em uma pipeline de dados auditável
para análise de risco logístico.

O projeto **Delivery Risk Intelligence** foi construído com foco em algo que
costuma ser subestimado em projetos de ML: a confiabilidade da base antes do
modelo.

A pipeline cobre:

- arquitetura RAW → Bronze → Silver → Gold;
- quality gates estruturais, semânticos e estatísticos;
- validação do target de atraso;
- governança temporal e prevenção de leakage;
- feature contract no momento da compra;
- Expected Freight reproduzido out-of-time;
- Shipping Friction como diagnóstico logístico.

A coorte Gold congelada possui **96,470 pedidos entregues**, com
**6,534 atrasos (6.77%)**, e o núcleo de features foi mantido
pequeno e auditável.

Um dos pontos que mais gostei do projeto foi separar claramente:

**o que é conhecido no momento da compra**
vs.
**o que só pode ser observado depois da entrega**.

Isso evita transformar informação futura em “boa performance” artificial.

O repositório está organizado para mostrar tanto a visão executiva quanto o
aprofundamento técnico.

GitHub:
https://github.com/borbapinheiro01-gif/delivery-risk-data-pipeline-

#DataEngineering #DataQuality #Python #SQL #DuckDB #MachineLearning
#Logistics #Analytics #Portfolio
