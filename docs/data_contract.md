# Scenario bundle contract

`ScenarioBundle` is the immutable hand-off between geospatial preprocessing and
simulation. It avoids binding the environment to LANDFIRE, GDAL, cloud object
storage, or a particular feature-store implementation.

Each compressed NPZ bundle includes rasters in `(y, x)` order:

- `elevation_m`: float32, metres.
- `fuel_load_kg_m2`: float32, non-negative.
- `barrier`: boolean non-burnable/control features.
- `asset_value`: float32 incident objective/value layer.

The metadata object must include schema version, CRS, cell size, source dataset
versions/acquisition dates, spatial/temporal transformations, scenario split,
and any ignition/initial-perimeter reference. Scenario split is stored before
training, so geographic patches from a source landscape do not leak across
train/development/test.

The current simulator accepts a bundle through `scenario.landscape_bundle`. Its
configured dimensions and cell size must match the bundle. Weather, resource
schedule, suppression parameters, objective profile, and seed remain in the
experiment manifest and are logged with each episode.
