# Delivery Risk Intelligence — Migration Infrastructure

Generated:

`20260831T071618Z`

## Project root

`/home/runner/workspace/Delivery_Risk_Intelligence`

## Migration principle

The scientific RAW layer and generated artifacts are distinct from executable
source code.

Migration should preserve:

1. source code;
2. frozen configurations and contracts;
3. metadata / Registry;
4. documentation;
5. dependency specification;
6. acquisition instructions for RAW and external data;
7. hashes and provenance;
8. execution order.

Large datasets and derived artifacts should be transferred separately or
reproduced from their source when possible.

## Audit directory

`reports/migration_audit/`

## Generated infrastructure manifests

- 01_system_environment.txt
- 02_python_environment.txt
- 03_pip_freeze.txt
- 04_pip_list.txt
- 05_critical_python_versions.txt
- 06_system_tools.txt
- 07_directory_tree.txt
- 08_file_manifest.csv
- 09_disk_usage.txt
- 10_raw_data_inventory.csv
- 11_external_data_inventory.csv
- 12_source_code_inventory.txt
- 13_python_compile_audit.csv
- 14_python_import_inventory.txt
- 15_configs_contracts_inventory.txt
- 16_metadata_registry_inventory.txt
- 17_reports_inventory.txt
- 18_artifacts_inventory.txt
- 19_environment_variable_names_ONLY.txt
- 20_environment_references_in_code.txt
- 21_build_environment_files.txt
- 22_git_state.txt
- 23_symlinks.txt
- 24_pipeline_inventory.txt

## Security

Environment-variable values and credentials are intentionally NOT included.
Only variable names and code references are recorded.

## Scientific-state protection

RAW modified: NO

Models retrained: NO

Scientific artifacts recalculated: NO

Data deleted: NO
