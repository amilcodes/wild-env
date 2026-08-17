"""Attribute audit for USDA Forest Service RDS-2025-0011."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdb", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import pyogrio

    columns = [
        "IncidentName",
        "FeatureCategory",
        "LineDateTime",
        "CreateDate",
        "LineLengthGeodesicKM",
        "FirelineEngagement",
        "Year",
        "IRWINID",
    ]
    _, table = pyogrio.read_arrow(
        args.gdb,
        layer="Firelines_Engagement_17_24",
        columns=columns,
        read_geometry=False,
    )
    frame = table.to_pandas()

    def grouped(names: list[str]) -> dict[str, dict[str, float | int]]:
        aggregate = frame.groupby(names, dropna=False).agg(
            features=("IRWINID", "size"),
            length_km=("LineLengthGeodesicKM", "sum"),
            line_datetime_present=("LineDateTime", "count"),
        )
        result: dict[str, dict[str, float | int]] = {}
        for key, row in aggregate.iterrows():
            label = " | ".join(str(value) for value in key) if isinstance(key, tuple) else str(key)
            result[label] = {
                "features": int(row["features"]),
                "length_km": float(row["length_km"]),
                "line_datetime_present": int(row["line_datetime_present"]),
            }
        return result

    crockets = frame[
        frame["IncidentName"].str.contains("Crocket", case=False, na=False)
    ]
    result = {
        "schema_version": 1,
        "source": {
            "title": (
                "Fireline engagement from the National Interagency Fire Center "
                "Feature Service from 2017-2024"
            ),
            "doi": "https://doi.org/10.2737/RDS-2025-0011",
            "archive_sha256": (
                "7698cccb39a07369b1dcc3f1bf83bfa12f8a6ee7afdd2c2f"
                "473599487b4bc64d"
            ),
        },
        "features": int(len(frame)),
        "incidents_by_irwin_id": int(frame["IRWINID"].nunique()),
        "incident_names": int(frame["IncidentName"].nunique()),
        "line_datetime_present_fraction": float(frame["LineDateTime"].notna().mean()),
        "create_date_present_fraction": float(frame["CreateDate"].notna().mean()),
        "engagement": grouped(["FirelineEngagement"]),
        "feature_category": grouped(["FeatureCategory"]),
        "year": grouped(["Year"]),
        "crockets_knob": {
            "features": int(len(crockets)),
            "line_datetime_present_fraction": float(
                crockets["LineDateTime"].notna().mean()
            ),
            "engagement": {
                str(key): {
                    "features": int(row["features"]),
                    "length_km": float(row["length_km"]),
                    "line_datetime_present": int(row["line_datetime_present"]),
                }
                for key, row in crockets.groupby(
                    "FirelineEngagement", dropna=False
                )
                .agg(
                    features=("IRWINID", "size"),
                    length_km=("LineLengthGeodesicKM", "sum"),
                    line_datetime_present=("LineDateTime", "count"),
                )
                .iterrows()
            },
        },
        "interpretation": [
            "LineDateTime is the only candidate construction-time field.",
            "CreateDate is a geodatabase creation field and is not treated as construction time.",
            "Engagement is a final-perimeter overlay outcome, not a causal treatment effect.",
            (
                "Summed length attributes are retained for audit only and are not "
                "treated as unique constructed-line length because the archive "
                "contains fragmented and potentially repeated features."
            ),
        ],
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
