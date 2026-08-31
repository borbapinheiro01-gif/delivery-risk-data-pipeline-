# Delivery Risk Intelligence — Platform Migration Handoff

## 1. Objetivo

Este documento define a fronteira da árvore executável que deve ser usada
para reconstruir o projeto em outra plataforma.

## 2. Princípio

A migração separa cinco classes:

1. código-fonte executável;
2. configurações e contratos;
3. metadata / Registry;
4. dados originais e externos;
5. artefatos científicos derivados.

Dados RAW e artefatos grandes não devem ser confundidos com o código-fonte.

## 3. Código legado

Cinco arquivos Python são mantidos exclusivamente por proveniência histórica.

Eles não pertencem ao pipeline executável atual.

O monólito:

`src/spatiotemporal_logistics/01_build_fast_spatiotemporal_core.py`

foi substituído pelo pipeline checkpointizado.

As quatro cópias do monólito preservadas no projeto correspondem ao mesmo
conteúdo histórico.

## 4. Pipeline espaço-temporal atual

A construção foi modularizada em:

1. `00_environment_and_registry.py`
2. `01_build_route_checkpoint.py`
3. `02_build_external_context_checkpoint.py`
4. `03_build_master_checkpoint.py`
5. `04_run_four_analyses.py`
6. `05_integrated_validation_report.py`

A partir dos artefatos resultantes, seguem os módulos científicos,
GN/GN-ADMM, governança preditiva, seleção final e Shipping Friction.

## 5. Itens que devem acompanhar a migração

Código:

- `src/`
- `scripts/`

Governança:

- `configs/`
- `contracts/`
- `metadata/`

Documentação:

- `docs/`

Auditoria e resultados pequenos:

- subconjunto selecionado de `reports/`

Dados:

- `data/raw/` transferido separadamente ou readquirido da fonte;
- `data/external/` transferido/reconstruído com provenance.

Artefatos:

- grandes arquivos de `artifacts/` devem ser transferidos separadamente
  ou reproduzidos pela sequência de execução.

## 6. Segurança

Secrets não são exportados.

Valores de variáveis de ambiente não fazem parte do manifesto de migração.

## 7. Validação

A árvore publicável deve apresentar:

`PUBLISHABLE_FAIL = 0`

antes da migração operacional.

## 8. Arquivos de auditoria

Consultar:

- `reports/migration_audit/26_legacy_source_exclusions.csv`
- `reports/migration_audit/27_publishable_python_manifest.csv`
- `reports/migration_audit/28_pipeline_entrypoints.csv`
- `reports/migration_audit/29_publishable_source_sha256.txt`
- `reports/migration_audit/30_migration_readiness_final.txt`

