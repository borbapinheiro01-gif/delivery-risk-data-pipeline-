# Historical source archive

This directory preserves historical source states for scientific provenance.

These files are **not part of the executable pipeline**.

They include:

- one explicitly broken historical Model 01 implementation;
- the superseded monolithic spatiotemporal implementation;
- historical snapshots of that monolith.

The active and reproducible execution path is stored under `src/`.

The historical `.py` files are stored with a `.txt` suffix deliberately so
automated Python compilation, test discovery and packaging do not mistake them
for executable source.

See:

`reports/migration_audit/26_legacy_source_exclusions.csv`

for the formal classification.
