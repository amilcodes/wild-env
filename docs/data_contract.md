# Incident and replay data contracts

There is no broadly adopted, simulator-complete wildfire episode standard.
Version 0.5 therefore composes existing geospatial conventions instead of
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

Each fuel or vegetation source in `metadata.sources` should include:

- `product_year` or exact product edition;
- `data_cutoff`;
- `disturbance_through_year` or exact disturbance cutoff;
- source event/treatment identifiers and dates when the landscape is
  reconstructed;
- native resolution, resampling, class mapping, and checksum; and
- an explicit historical-use assessment.

The public LANDFIRE 2025 importer now records product year 2025 and disturbance
coverage through 2024. A historical run must pass
`assess_historical_fuel_provenance` and a pixel-level dated-disturbance review
before its landscape is treated as time-admissible evidence.

### Perimeter observations

Every feature requires:

- WGS84 polygon or multipolygon geometry;
- normalized ISO-8601 `observed_at`;
- observation source and source time semantics.

Historical scoring should additionally provide these source-specific fields
when the source publishes them:

- `acquisition_start` and `acquisition_end`: ISO-8601 sensing or compositing
  interval;
- `available_at`: earliest time the observation could enter the incident
  information state;
- `detection_probability` and `false_alarm_probability`: calibrated source
  likelihood terms or a referenced calibration record;
- `obscured_probability`: scalar or aligned raster probability for cloud,
  smoke, missing scan, or another observability loss;
- `geolocation_sigma_m`: declared one-sigma spatial error; and
- footprint/scan geometry plus retained source quality flags.

`observed_at` remains a compatibility field. It must not silently substitute
for both sensing and availability time in a causal replay. When exact scan time
is unavailable, evaluation uses a declared acquisition-time prior over the
source interval and records that choice.

The NASA FEDS importer retains original properties, sorts frames by source
time and records its 375 m nominal resolution and 12-hour time-bin semantics.
Duplicate timestamps are retained because upstream products may publish
multiple records with the same nominal bin. `PerimeterSeries` converts that
source collection to an evaluation sequence by unioning all nonempty features
with an identical timestamp. Each derived frame records
`source_feature_count`, `coalesced_duplicate_features`, and the retained source
properties. Evaluation time must therefore be strictly increasing even when
the source feature collection contains scene fragments.

### Tactical performance and airspace

The reference vehicle catalog is a separate JSON contract. A profile contains:

- a stable profile identifier, display name, and operator reference;
- resource kind, current operational roles, and operational control/autonomy;
- identity-source URLs and a current-use status statement;
- a simulation evidence grade;
- an optional reviewed performance-surface path;
- a separately graded optional delivery-surface path; and
- simulator parameters with value, SI unit, evidence basis, source value,
  source URL, and explanatory note.

Evidence basis is one of `published`, `unit_conversion`, `role_mapping`, or
`modeling_assumption`. The profile-level grade is one of
`scenario_assumption`, `public_specification`, `flight_manual`, or
`engineering_validated`. A `flight_manual` or `engineering_validated` profile
without a performance surface is invalid. Resource records retain the profile
identifier and evidence grade when materialized, so a replay or checkpoint can
be traced back to the catalog.

An engineering-grade delivery claim likewise requires a delivery surface. The
current S-2T record stores controlled-test coverage levels, flow rates,
controller settings, measured longest-line lengths, airspeed/drop-height test
domain, source digest, configuration applicability, and limitations. The
volume-equivalent width used by the raster simulator is labeled as an inferred
transform rather than a measured quantity.

Retardant mass coverage and effective subcell coverage are separate state
fields. The former is a cell-area average whose integral reproduces applied
volume. The latter preserves the source-table target coverage along a line that
is narrower than the simulation cell. This prevents grid dilution from being
mistaken for a weak delivery while retaining an auditable mass field.

An optional tactical-performance JSON record contains:

- schema version and source/configuration provenance;
- strictly increasing density-altitude and payload-fraction axes;
- true-airspeed and endurance-multiplier tables on those axes; and
- maximum payload fraction by density altitude.

Values outside the source axes are infeasible; the simulator does not
extrapolate them. The bundled generic surface is marked synthetic and is
suitable only for interface tests.

Scenario airspace volumes use grid-coordinate polygons, an MSL altitude band,
an active elapsed-time interval, a `prohibited` or `reserved` kind, and
optional allowed resource IDs. They are simulator constraints, not a source
aviation product. A real scenario must retain the source TFR, NOTAM, incident-
airspace, or authorization record and the transformation into scenario
coordinates.

### Weather forcing

The CF-NetCDF reader requires a monotonic `time` coordinate and:

- `wind_speed` in m/s;
- `wind_from_direction` in degrees;
- `air_temperature` in K;
- `relative_humidity` in percent.
- optional `precipitation_rate` in mm/h.
- optional dead/live fuel-moisture fields in kg/kg;
- optional `wind_u_correction` and `wind_v_correction` in m/s.

Variables may use `(time,)` for incident-wide forcing or `(time, y, x)` for
fields aligned to the fire grid. Direction interpolation uses sine/cosine
components, so interpolation between 350° and 10° passes through north.
Coupled wind corrections are added in Cartesian components. The
`analyze_incident_forcing` path produces these fields from station innovations
and a gridded background by Gaussian optimum interpolation.

`aeolus_metadata_json` stores the complete forcing provenance as a NetCDF
global attribute. The loader restores it while treating the CF time units,
source, and history attributes as authoritative. HRRR forcing records requested
and available analysis-hour counts, coverage fraction, missing timestamps,
native source-index bounds, and interpolation semantics.

The historical preparation hierarchy uses NASA POWER as a long spin-up
background and overlays NOAA HRRR 10-m wind, 2-m temperature/humidity, and
surface precipitation during the incident. A missing HRRR hour is accepted
only when total coverage exceeds the configured threshold; interpolation over
that gap is explicit in metadata. The current thermodynamic topographic
projection adjusts temperature by elevation and recomputes humidity while
conserving vapor pressure. It leaves the source wind unchanged and does not
claim sub-3-km terrain-flow reconstruction.

## ReplayBundle v2

`states.zarr` stores one-minute time-indexed state:

- truth phase, signed level set, fire type, intensity, spread rate, flame
  length, fuel remaining and dead-fuel moisture;
- belief intensity mean/standard deviation, burn probability, known burned
  area and arrival-time mean/standard deviation;
- raw/effective water and retardant coverage, constructed line and line
  engagement status;
- full raster wind speed/direction, air temperature, relative humidity and
  precipitation forcing;
- resource position, status, payload fraction, remaining endurance, ETA,
  task kind/heading, target, current site and requested service site;
- remaining finite suppressant volume at each service node.

Static fields include elevation, fuel model/load, canopy, barriers and asset
value. Arrays use `(time, y, x)` or `(time, resource/site)`. Raster time chunks
are one frame deep for deterministic seeks.

`events.parquet` stores typed event records. `metadata.json` fixes the schema,
complete scenario, scenario/incident identity, civil-time origin, spatial
reference, episode result, policy identity and checkpoint digest.

ReplayBundle v1 remains readable. Its absent weather and detailed
resource/logistics arrays are treated as unavailable; viewers may use scalar
scenario weather as an explicit compatibility fallback.

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
