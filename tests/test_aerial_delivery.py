from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeolus.data.aerial_delivery import (
    delivery_geometry,
    load_aerial_delivery_surface,
)

SURFACE_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "aviation"
    / "delivery_surfaces"
    / "calfire_s2t_mtdc_2006_gum_v1.json"
)


def test_s2t_surface_reproduces_usfs_table_three() -> None:
    surface = load_aerial_delivery_surface(SURFACE_PATH)
    geometry = delivery_geometry(
        surface,
        requested_coverage_gpc=3.0,
        payload_l=surface.nominal_payload_l,
    )

    assert surface.surface_id == "calfire-s2t-mtdc-2006-gum-v1"
    assert geometry.line_length_m == pytest.approx(595.0 * 0.3048)
    assert geometry.flow_rate_l_s == pytest.approx(430.0 * 3.785411784)
    assert geometry.controller_setting == "3"
    assert geometry.effective_width_m == pytest.approx(20.4907563)
    assert not geometry.extrapolated


def test_delivery_transform_scales_line_with_partial_load_and_conserves_area() -> None:
    surface = load_aerial_delivery_surface(SURFACE_PATH)
    full = delivery_geometry(
        surface,
        requested_coverage_gpc=2.0,
        payload_l=surface.nominal_payload_l,
    )
    half = delivery_geometry(
        surface,
        requested_coverage_gpc=2.0,
        payload_l=0.5 * surface.nominal_payload_l,
    )

    assert half.line_length_m == pytest.approx(0.5 * full.line_length_m)
    assert half.effective_width_m == pytest.approx(full.effective_width_m)


def test_delivery_transform_does_not_treat_short_high_coverage_contours_as_widths() -> None:
    surface = load_aerial_delivery_surface(SURFACE_PATH)
    geometry = delivery_geometry(
        surface,
        requested_coverage_gpc=8.0,
        payload_l=surface.nominal_payload_l,
    )

    assert geometry.requested_coverage_gpc == 6.0
    assert geometry.line_length_m == pytest.approx(285.0 * 0.3048)
    assert geometry.extrapolated


def test_delivery_surface_rejects_unproven_provenance(tmp_path: Path) -> None:
    payload = json.loads(SURFACE_PATH.read_text(encoding="utf-8"))
    del payload["metadata"]["configuration_applicability"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="applicability"):
        load_aerial_delivery_surface(path)
