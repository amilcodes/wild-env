# Incident and replay data contracts

There is no broadly adopted, simulator-complete wildfire episode standard.
Version 0.3 therefore composes existing geospatial conventions instead of
inventing new encodings for each asset.

## IncidentBundle v2

An incident is a relocatable directory with a STAC 1.1 Item at `item.json`.
The Item has a WGS84 geometry/bounding box, start and end timestamps, source
provenance, and these assets:

| Asset key | Encoding | Required | Semantics |
|---|---|---:|---|
| `simulator-landscape` | compressed NPZ | yes | aligned arrays used by the fast kernel |
| `observed-perimeters` | GeoJSON | yes | cumulative, timestamped perimeter observations |
| `landscape` | GeoTIFF | no | source terrain/fuel bands with CRS/affine transform |
| `weather` | CF-NetCDF | no | time-indexed incident weather forcing |

Paths are resolved below the incident root and traversal outside the bundle is
rejected. Loading validates the STAC/schema versions, asset existence, raster
shape, cell size and perimeter collection.

### Simulator landscape

Arrays use `(y, x)` order:

- `elevation_m`: float32 metres;
- `fuel_load_kg_m2`: float32 non-negative surface-fuel proxy;
- `fuel_model_number`: int16 Anderson/Scott–Burgan model code;
- `canopy_cover`: float32 fraction;
- `canopy_height_m` and `canopy_base_height_m`: float32 metres;
- `canopy_bulk_density_kg_m3`: float32 kg/m³;
- `barrier`: boolean non-burnable/control cells;
- `asset_value`: float32 non-negative objective weight.

Metadata records schema version, CRS, six-value affine transform, bounds, cell
size, source services, transformations and immutable train/development/
evaluation split.

The public importer preserves LANDFIRE FBFM40 codes and converts LANDFIRE
canopy integer scaling to SI. The fuel-load proxy remains available for
conditioning and legacy bundles. `aeolus-incident enrich-scenario` upgrades a
legacy NPZ from its retained six-band source GeoTIFF without re-downloading.

### Perimeter observations

Every feature requires:

- WGS84 polygon or multipolygon geometry;
- normalized ISO-8601 `observed_at`;
- observation source and source time semantics.

The NASA FEDS importer retains original properties, sorts frames by source
time and records its 375 m nominal resolution and 12-hour time-bin semantics.
Duplicate timestamps are retained because upstream products may publish
multiple records with the same nominal bin.

### Weather forcing

The CF-NetCDF reader requires a monotonic `time` coordinate and:

- `wind_speed` in m/s;
- `wind_from_direction` in degrees;
- `air_temperature` in K;
- `relative_humidity` in percent.
- optional `precipitation_rate` in mm/h.

Direction interpolation unwraps angles, so interpolation between 350° and 10°
passes through north.

## ReplayBundle v1

`states.zarr` stores minute-indexed phase, fire type, intensity, spread rate,
flame length, fuel remaining, dead-fuel moisture, belief, treatments and
resources. Static fields include elevation, fuel model/load and canopy layers.
`events.parquet` stores typed event records. `metadata.json` fixes the schema,
scenario, episode result, policy identity and checkpoint digest.

Replay bundles are products of a particular simulator version and experiment
manifest. They are suitable for deterministic analysis and rendering; they are
not treated as new incident observations.

## Storage at scale

Local bundles use NPZ/GeoTIFF/NetCDF for portability. A larger archive should
retain the STAC catalog and move aligned multidimensional arrays to chunked
Zarr/object storage, tabular episode indices to GeoParquet, and immutable
source assets to content-addressed storage. Dataset splits must be geographic
and incident-level before tiling to prevent adjacent pixels or later timestamps
from the same fire leaking across research splits.
