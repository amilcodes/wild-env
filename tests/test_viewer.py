from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aeolus.config import load_config
from aeolus.viewer.config import load_viewer_config


def test_viewer_config_applies_preset_and_resolves_imagery(tmp_path: Path) -> None:
    imagery = tmp_path / "orthophoto.tif"
    manifest = tmp_path / "viewer.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "preset": "imagery",
                "camera": {"mode": "follow", "follow_resource": "heli_07"},
                "imagery": {
                    "path": imagery.name,
                    "bands": [3, 2, 1],
                    "attribution": "Test imagery",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_viewer_config(manifest)

    assert config.layers.imagery
    assert not config.layers.contours
    assert config.camera.follow_resource == "heli_07"
    assert config.imagery.path == str(imagery.resolve())
    assert config.imagery.bands == (3, 2, 1)


def test_viewer_config_rejects_unknown_preset(tmp_path: Path) -> None:
    manifest = tmp_path / "viewer.yaml"
    manifest.write_text("preset: dashboard\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown viewer preset"):
        load_viewer_config(manifest)


def test_shipped_viewer_and_reference_manifests_validate() -> None:
    root = Path(__file__).parents[1]
    for path in sorted((root / "configs" / "viewer").glob("*.yaml")):
        load_viewer_config(path)
    experiment = load_config(root / "configs" / "replay_reference.yaml")
    assert experiment.scenario.scenario_id == "foothill-operations-reference"
    assert len(experiment.scenario.resources) == 8
    assert len(experiment.scenario.service_sites) == 4
