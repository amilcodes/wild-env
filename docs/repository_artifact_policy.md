# Repository artifact policy

The repository versions source, tests, experiment contracts, compact
machine-readable summaries, figures needed by the research notes, and small
deterministic reference assets. This keeps a clone sufficient for review and
local smoke tests without turning Git into an incident-data or checkpoint
store.

## Versioned

- Python and C++ source, tests, build metadata, and deployment manifests.
- YAML/JSON configuration and evidence registries.
- Research methods, limitations, execution plans, and frozen partition
  definitions.
- Compact JSON/CSV/NPZ summaries and publication figures that support a stated
  result.
- Checksums and source citations that allow an external dataset to be obtained
  and verified.

## External or regenerated

- Raw NIROPS, GOFER, LANDFIRE, HRRR, FEDS, and terrain payloads.
- Per-incident perimeter and active-line GeoJSON exports.
- GeoTIFF, NetCDF, Zarr, and Parquet materializations.
- Physics caches, replay stores, videos, training checkpoints, and run logs.
- Failed or active local-run state containing workstation paths.
- Package builds, virtual environments, test caches, and native binaries.

Large immutable research releases should use a versioned object store or a
GitHub release with a checksum manifest. Trained weights require a model card,
training configuration, data-partition identifier, code commit, and evaluation
summary before publication.

## Reproducibility rule

An excluded artifact may support a result only when the repository retains the
command, configuration, input identity, checksum contract, and compact output
needed to audit or regenerate it. Historical test observations remain outside
calibration paths regardless of storage location.
