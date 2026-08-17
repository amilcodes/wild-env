"""Build the Aeolus-IA historical validation research note as a PDF."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#182129")
SLATE = colors.HexColor("#485867")
MUTED = colors.HexColor("#6D7785")
BLUE = colors.HexColor("#2C718E")
PALE_BLUE = colors.HexColor("#EAF2F5")
ORANGE = colors.HexColor("#C96732")
PALE_ORANGE = colors.HexColor("#F7EEE8")
GREEN = colors.HexColor("#31836B")
PALE_GREEN = colors.HexColor("#E9F2EE")
RED = colors.HexColor("#B6473A")
PALE_RED = colors.HexColor("#F8E9E6")
SAND = colors.HexColor("#F3F0E8")
LINE = colors.HexColor("#CFD5DB")
WHITE = colors.white

METHODS = ("persistence", "raw_physics", "calibrated_physics")
METHOD_LABELS = {
    "persistence": "Persistence",
    "raw_physics": "Raw physics",
    "calibrated_physics": "Calibrated physics",
}
INCIDENT_LABELS = {
    "CA-AEU-017769_Electra": "Electra, CA",
    "OR-MAF-022199_CrocketsKnob": "Crockets Knob, OR",
    "AZ-SCA-001418_DryLake": "Dry Lake, AZ",
    "ID-IPF-000447_RidgeCreek": "Ridge Creek, ID",
    "NM-GNF-000382_Davis": "Davis, NM",
    "UT-VLD-000127_Bear": "Bear, UT",
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=30,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=SLATE,
            spaceAfter=10,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=10,
            textColor=BLUE,
            tracking=1.0,
            spaceAfter=9,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10.2,
            textColor=SLATE,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.5,
            textColor=SLATE,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=20,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.5,
            textColor=SLATE,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.2,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.8,
            textColor=WHITE,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=INK,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=3,
        ),
        "ref": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.5,
            textColor=INK,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=4,
        ),
    }


def _p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def _bullet(text: str, styles: dict[str, ParagraphStyle], *, small: bool = False) -> Paragraph:
    style = styles["small" if small else "body"]
    return Paragraph(f"<bullet>&bull;</bullet>{text}", style)


def _header_footer(canvas: Any, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    width, height = letter
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(0.65 * inch, height - 0.50 * inch, width - 0.65 * inch, height - 0.50 * inch)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(SLATE)
        canvas.drawString(0.65 * inch, height - 0.39 * inch, "AEOLUS-IA RESEARCH NOTE 2026-01")
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(
            width - 0.65 * inch,
            height - 0.39 * inch,
            "HISTORICAL SPREAD VALIDATION",
        )
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(0.65 * inch, 0.45 * inch, width - 0.65 * inch, 0.45 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, 0.27 * inch, "28 July 2026")
    canvas.drawRightString(width - 0.65 * inch, 0.27 * inch, str(doc.page))
    canvas.restoreState()


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    *,
    header: bool = True,
    font_size: float = 7.3,
) -> Table:
    converted: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        converted.append(
            [
                value
                if hasattr(value, "wrap")
                else Paragraph(
                    str(value),
                    styles["table_head" if header and row_index == 0 else "table"],
                )
                for value in row
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, LINE),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ]
        )
        first_body = 1
    else:
        first_body = 0
    for row_index in range(first_body, len(rows)):
        if (row_index - first_body) % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F6F8F9")))
    table.setStyle(TableStyle(commands))
    return table


def _callout(
    title: str,
    body: str,
    styles: dict[str, ParagraphStyle],
    *,
    background: colors.Color = PALE_BLUE,
    accent: colors.Color = BLUE,
) -> Table:
    content = [
        _p(title, styles, "h2"),
        _p(body, styles, "body"),
    ]
    table = Table([[content]], colWidths=[6.82 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _image(path: Path, width: float, height: float) -> Image:
    image = Image(str(path), width=width, height=height)
    image.hAlign = "CENTER"
    return image


def _metric_summary(results: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    calibrated = results["summaries"]["calibrated_physics"]
    growth = results["active_growth_summaries"]["calibrated_physics"]
    persistence = results["summaries"]["persistence"]
    cells = [
        [
            [
                _p(f"{calibrated['metrics.iou']['mean']:.3f}", styles, "metric"),
                _p("calibrated cumulative IoU", styles, "metric_label"),
            ],
            [
                _p(f"{persistence['metrics.iou']['mean']:.3f}", styles, "metric"),
                _p("persistence cumulative IoU", styles, "metric_label"),
            ],
            [
                _p(f"{growth['growth_tolerance_1_cell.f1']['mean']:.3f}", styles, "metric"),
                _p("active-growth tolerance F1", styles, "metric_label"),
            ],
            [_p("24", styles, "metric"), _p("held-out forecast intervals", styles, "metric_label")],
        ]
    ]
    table = Table(cells, colWidths=[1.705 * inch] * 4)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SAND),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _method_table(results: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[Any]] = [
        [
            "Method",
            "Cumulative IoU",
            "Active-growth IoU",
            "Active-growth F1",
            "Boundary mean",
            "Symmetric diff.",
        ]
    ]
    for method in METHODS:
        summary = results["summaries"][method]
        active = results["active_growth_summaries"][method]
        rows.append(
            [
                METHOD_LABELS[method],
                f"{summary['metrics.iou']['mean']:.3f}",
                f"{active['growth_metrics.iou']['mean']:.3f}",
                f"{active['growth_tolerance_1_cell.f1']['mean']:.3f}",
                f"{summary['boundary.mean_symmetric_distance_m']['mean']:.0f} m",
                f"{summary['metrics.symmetric_difference_km2']['mean']:.1f} km2",
            ]
        )
    return _table(
        rows,
        [1.22 * inch, 1.02 * inch, 1.08 * inch, 1.05 * inch, 1.12 * inch, 1.10 * inch],
        styles,
    )


def _incident_means(results: dict[str, Any], code: str, method: str) -> dict[str, float]:
    records = [
        item["forecast"]
        for item in results["forecasts"]
        if item["incident_code"] == code and item["method"] == method
    ]
    active = [item for item in records if item["growth_metrics"]["observed_area_km2"] > 0]
    return {
        "iou": sum(item["metrics"]["iou"] for item in records) / len(records),
        "growth_f1": (
            sum(item["growth_tolerance_1_cell"]["f1"] for item in active) / len(active) if active else 0.0
        ),
        "boundary": sum(item["boundary"]["mean_symmetric_distance_m"] for item in records) / len(records),
        "symmetric_difference": sum(item["metrics"]["symmetric_difference_km2"] for item in records)
        / len(records),
    }


def _incident_table(results: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    calibration_by_code = {
        item["incident_code"]: item["selected_spread_adjustment"] for item in results["calibrations"]
    }
    rows: list[list[Any]] = [
        [
            "Incident",
            "Multiplier",
            "Persistence IoU",
            "Raw IoU",
            "Calibrated IoU",
            "Cal. growth F1",
            "Cal. boundary",
        ]
    ]
    for code in calibration_by_code:
        persistence = _incident_means(results, code, "persistence")
        raw = _incident_means(results, code, "raw_physics")
        calibrated = _incident_means(results, code, "calibrated_physics")
        rows.append(
            [
                INCIDENT_LABELS[code],
                f"{calibration_by_code[code]:g}",
                f"{persistence['iou']:.3f}",
                f"{raw['iou']:.3f}",
                f"{calibrated['iou']:.3f}",
                f"{calibrated['growth_f1']:.3f}",
                f"{calibrated['boundary']:.0f} m",
            ]
        )
    return _table(
        rows,
        [
            1.31 * inch,
            0.70 * inch,
            0.88 * inch,
            0.72 * inch,
            0.92 * inch,
            0.90 * inch,
            0.88 * inch,
        ],
        styles,
    )


def _build_story(
    results: dict[str, Any],
    audit: dict[str, Any],
    figures: Path,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = []

    # Cover
    story.extend(
        [
            Spacer(1, 0.15 * inch),
            _p("AEOLUS-IA RESEARCH NOTE 2026-01", styles, "kicker"),
            _p("Historical Validation of Aeolus-IA v0.3", styles, "title"),
            _p(
                "Airborne-infrared spread hindcasts and a suppression-evidence audit",
                styles,
                "subtitle",
            ),
            HRFlowable(width="100%", thickness=2, color=BLUE, spaceBefore=4, spaceAfter=14),
            _callout(
                "Result",
                "The current simulator is <b>not historically accurate enough for a spread-forecast claim</b>. "
                "It shows weak advancing-front signal, but a no-growth persistence baseline wins on cumulative "
                "extent and boundary location. Historical suppression accuracy is not causally scoreable from "
                "the public action records available for this study.",
                styles,
                background=PALE_RED,
                accent=RED,
            ),
            Spacer(1, 0.16 * inch),
            _metric_summary(results, styles),
            Spacer(1, 0.18 * inch),
            _image(figures / "incident_locations.png", 6.6 * inch, 3.68 * inch),
            _p(
                "<b>Study design.</b> Six western U.S. fires, one calibration transition per fire, and four later "
                "held-out transitions per fire. Every forecast starts from an observed NIROPS perimeter.",
                styles,
                "caption",
            ),
            Spacer(1, 0.05 * inch),
            _p(
                "Prepared 28 July 2026  |  Version 1.0  |  Source and frozen results accompany this note",
                styles,
                "small",
            ),
        ]
    )

    # Page 2 - abstract and decision
    story.extend(
        [
            PageBreak(),
            _p("Abstract and decision", styles, "h1"),
            _p(
                "Aeolus-IA v0.3 was evaluated against analyst-interpreted airborne infrared perimeter "
                "progressions from six western U.S. fires. One earlier observation interval per incident "
                "selected a scalar spread adjustment; four subsequent intervals were held out. The benchmark "
                "contains 24 forecasts, 21 with observed growth. Persistence, raw physics, and calibrated "
                "physics were compared using burned-extent overlap, newly burned-area localization, boundary "
                "distance, and area disagreement. Confidence intervals were bootstrapped by incident.",
                styles,
            ),
            _p(
                "Persistence achieved mean cumulative IoU 0.873 (95% cluster interval 0.830-0.923), mean "
                "symmetric boundary distance 156 m, and mean symmetric-difference area 4.4 km2. Raw physics "
                "reached 0.534 IoU, 1,223 m, and 50.1 km2. Single-interval calibrated physics reached 0.611 "
                "IoU, 938 m, and 50.3 km2. On active-growth intervals, persistence had zero advancing-front "
                "overlap; raw and calibrated physics had one-cell-tolerance F1 of 0.145 and 0.151. The model "
                "therefore contains weak propagation information while failing the more important historical "
                "forecast baseline.",
                styles,
            ),
            _p(
                "The suppression audit covers 2,236,584 public fireline features. Only 6.61% carry the candidate "
                "construction timestamp. Crockets Knob, which overlaps the spread sample, has 1,742 classified "
                "line features and only three timestamps. No matched event-level aerial drop sequence was found. "
                "The data support spatial line-outcome analysis, but not chronological replay or causal policy "
                "validation.",
                styles,
            ),
            Spacer(1, 0.08 * inch),
            _callout(
                "Decision",
                "Keep the benchmark as a release gate. Treat v0.3 replays as simulator experiments. Before "
                "training policies for historical claims, add perimeter assimilation, incident-grade weather "
                "and fuel moisture, probabilistic ensembles, explicit suppression actions, and action logs.",
                styles,
                background=PALE_ORANGE,
                accent=ORANGE,
            ),
            Spacer(1, 0.12 * inch),
            _p("Claims supported by this study", styles, "h2"),
            _table(
                [
                    ["Question", "Status", "Evidence"],
                    [
                        "Can real fires be imported and replayed?",
                        "Yes",
                        "NIROPS perimeters, USGS terrain, LANDFIRE fuels/canopy, and hourly POWER forcing are frozen in incident bundles.",
                    ],
                    [
                        "Does v0.3 locate some observed growth?",
                        "Weakly",
                        "Active-growth tolerance F1 is 0.151 after scalar calibration, versus 0 for persistence.",
                    ],
                    [
                        "Is v0.3 historically accurate as a spread forecast?",
                        "No",
                        "Persistence wins cumulative IoU, boundary distance, and symmetric-difference area.",
                    ],
                    [
                        "Can historical suppression-policy accuracy be scored?",
                        "No",
                        "Action timing and matched aerial-drop histories are missing; the current action model is incomplete.",
                    ],
                ],
                [1.72 * inch, 0.72 * inch, 4.36 * inch],
                styles,
            ),
        ]
    )

    # Page 3 - data
    story.extend(
        [
            PageBreak(),
            _p("1. Reference data and incident assembly", styles, "h1"),
            _p(
                "<b>Observed progression.</b> The Magstadt et al. NIROPS release contains 12,705 perimeter "
                "observations for 737 incidents from 2020-2024. Trained interpreters manually digitized active "
                "wildfire extents from U.S. Forest Service airborne thermal imagery. Timestamps are normalized "
                "to UTC, geometries to EPSG:4326, and included fires have at least three collection days plus "
                "one consecutive-day pair. Operational deployment favors moderate and large fires and frequently "
                "begins after ignition.",
                styles,
            ),
            _p(
                "<b>Landscape.</b> Each fire is rasterized to 128 x 128 cells. Elevation and slope are derived "
                "from USGS 3DEP. Surface fuel model, canopy cover, canopy height, canopy base height, and canopy "
                "bulk density are sampled from LANDFIRE 2025. Cell size ranges from 134 m to 188 m across the "
                "six incident windows.",
                styles,
            ),
            _p(
                "<b>Weather.</b> Hourly wind speed and direction, temperature, relative humidity, and precipitation "
                "are requested at the incident center from NASA POWER in UTC. POWER reports hourly values at "
                "native source resolution. This study uses it as a reproducible reanalysis forcing, not as a "
                "claim of fire-line meteorological fidelity.",
                styles,
            ),
            Spacer(1, 0.06 * inch),
            _table(
                [
                    [
                        "Incident",
                        "State",
                        "NIROPS frames",
                        "Cell size",
                        "Calibration pair",
                        "Validation pairs",
                    ],
                    ["Electra", "CA", "8", "143 m", "1 -> 2", "2->3, 3->4, 4->5, 5->6"],
                    ["Crockets Knob", "OR", "16", "134 m", "0 -> 1", "1->2, 2->3, 3->4, 4->5"],
                    ["Dry Lake", "AZ", "12", "134 m", "0 -> 1", "1->2, 2->3, 3->4, 4->5"],
                    ["Ridge Creek", "ID", "17", "142 m", "3 -> 4", "4->5, 5->6, 6->7, 7->8"],
                    ["Davis", "NM", "9", "135 m", "0 -> 1", "1->2, 2->3, 3->4, 4->5"],
                    ["Bear", "UT", "14", "188 m", "0 -> 1", "1->2, 2->3, 3->4, 4->5"],
                ],
                [1.24 * inch, 0.48 * inch, 0.82 * inch, 0.68 * inch, 0.92 * inch, 2.66 * inch],
                styles,
            ),
            Spacer(1, 0.11 * inch),
            _p("Frozen source checksums", styles, "h2"),
            _p(
                "<font name='Courier' size='7'>NIROPS SHA-256  "
                "b19fb16ce2792d9a9c01f1768d09962566b0f3cada8d1f23a9851ab3fce75615</font>",
                styles,
                "small",
            ),
            _p(
                "<font name='Courier' size='7'>FLE SHA-256     "
                "7698cccb39a07369b1dcc3f1bf83bfa12f8a6ee7afdd2c2f473599487b4bc64d</font>",
                styles,
                "small",
            ),
            _callout(
                "Observation target",
                "NIROPS is the reference for spatial progression at the acquisition times. It is not continuous "
                "ground truth between flights. Undetected islands, perimeter interpretation, flight timing, and "
                "historical suppression are folded into the target.",
                styles,
            ),
        ]
    )

    # Page 4 - protocol
    story.extend(
        [
            PageBreak(),
            _p("2. Experimental protocol", styles, "h1"),
            _p(
                "The protocol isolates one-step propagation. Each held-out forecast is reinitialized from its own "
                "observed start perimeter, so errors do not compound across the entire incident. This answers a "
                "narrow question: given the observed fire state at time t, does the simulator reproduce the next "
                "observed extent at t + delta?",
                styles,
            ),
            _p("Forecast pipeline", styles, "h2"),
            _table(
                [
                    ["Stage", "Operation", "Leakage control"],
                    [
                        "1. Align",
                        "Project and rasterize observed perimeters on the simulator landscape.",
                        "One common grid per incident.",
                    ],
                    [
                        "2. Calibrate",
                        "Select one shared surface/crown spread multiplier on one earlier interval.",
                        "Validation intervals are excluded.",
                    ],
                    [
                        "3. Initialize",
                        "Set burned state from the observed start perimeter.",
                        "No ignition-location reconstruction.",
                    ],
                    [
                        "4. Integrate",
                        "Apply hourly weather and run for the exact observation interval.",
                        "No simulated suppression.",
                    ],
                    [
                        "5. Score",
                        "Compare cumulative extent, new growth, and perimeter boundary.",
                        "Same target and tolerance for all methods.",
                    ],
                    [
                        "6. Quantify",
                        "Bootstrap 2,000 times by incident.",
                        "Four intervals from one fire remain clustered.",
                    ],
                ],
                [0.82 * inch, 3.34 * inch, 2.64 * inch],
                styles,
            ),
            _p("Forecasts", styles, "h2"),
            _bullet(
                "<b>Persistence.</b> The start perimeter remains fixed. This is the required conservative baseline.",
                styles,
            ),
            _bullet(
                "<b>Raw physics.</b> Aeolus-IA v0.3 with surface and crown spread adjustment 1.0.", styles
            ),
            _bullet(
                "<b>Calibrated physics.</b> The same model with one incident-specific multiplier selected by "
                "new-growth IoU and an area-ratio penalty.",
                styles,
            ),
            _p("Metrics", styles, "h2"),
            _table(
                [
                    ["Metric", "Definition", "Interpretation"],
                    [
                        "Cumulative IoU",
                        "|predicted intersection observed| / |predicted union observed|",
                        "Rewards overlap of the full burned interior.",
                    ],
                    [
                        "Growth IoU",
                        "IoU after subtracting the observed start mask",
                        "Tests localization of newly burned cells.",
                    ],
                    [
                        "Tolerance F1",
                        "Growth precision/recall after one-cell radial dilation",
                        "Allows 134-188 m raster localization error.",
                    ],
                    [
                        "Boundary mean",
                        "Mean of bidirectional nearest-boundary distances",
                        "Physical perimeter displacement in meters.",
                    ],
                    [
                        "Hausdorff 95",
                        "95th percentile of bidirectional boundary distances",
                        "Robust near-worst boundary error.",
                    ],
                    [
                        "Symmetric difference",
                        "Area predicted or observed, but not both",
                        "Absolute spatial disagreement in km2.",
                    ],
                ],
                [1.08 * inch, 3.02 * inch, 2.70 * inch],
                styles,
            ),
            Spacer(1, 0.07 * inch),
            _callout(
                "Interpretation rule",
                "Cumulative IoU alone is insufficient. Daily fire growth is often small relative to the existing "
                "burned interior, allowing persistence to score highly. Growth metrics and boundary distances are "
                "reported beside it.",
                styles,
                background=PALE_ORANGE,
                accent=ORANGE,
            ),
        ]
    )

    # Page 5 - aggregate results
    story.extend(
        [
            PageBreak(),
            _p("3. Aggregate held-out results", styles, "h1"),
            _method_table(results, styles),
            Spacer(1, 0.08 * inch),
            _image(figures / "aggregate_metrics.png", 6.78 * inch, 4.34 * inch),
            _p(
                "<b>Figure 1.</b> Means and incident-cluster 95% bootstrap intervals. The advancing-front panel "
                "uses only the 21 intervals with observed growth. The other panels use all 24 intervals.",
                styles,
                "caption",
            ),
            _callout(
                "Finding",
                "The simulator does not beat persistence on cumulative extent, boundary displacement, or area "
                "disagreement. Its active-front scores are nonzero, which indicates directional signal, but the "
                "localization is weak: calibrated one-cell-tolerance F1 is 0.151.",
                styles,
                background=PALE_RED,
                accent=RED,
            ),
        ]
    )

    # Page 6 - incident results
    story.extend(
        [
            PageBreak(),
            _p("4. Incident-level transfer", styles, "h1"),
            _incident_table(results, styles),
            Spacer(1, 0.10 * inch),
            _p(
                "Calibration improves cumulative IoU strongly for Electra, Dry Lake, and Ridge Creek. It degrades "
                "Crockets Knob and Davis. Bear is the dominant transfer failure: its multiplier reaches the upper "
                "candidate 3.5 and later forecasts continue expanding while observed daily growth decelerates.",
                styles,
            ),
            _image(figures / "calibration_transfer.png", 6.75 * inch, 3.24 * inch),
            _p(
                "<b>Figure 2.</b> One scalar is fitted on an earlier transition, then frozen for four later "
                "transitions. Positive bars indicate improvement over raw physics.",
                styles,
                "caption",
            ),
            _callout(
                "Bear regime failure",
                "Calibrated Bear forecasts average cumulative IoU 0.243, boundary displacement 3.11 km, and "
                "symmetric-difference area 235 km2. This is the expected failure mode of a fixed effective spread "
                "factor under changing winds, moisture, fire-atmosphere coupling, and suppression.",
                styles,
                background=PALE_ORANGE,
                accent=ORANGE,
            ),
        ]
    )

    # Page 7 - atlas
    story.extend(
        [
            PageBreak(),
            _p("5. Spatial error atlas", styles, "h1"),
            _image(figures / "incident_atlas.png", 6.95 * inch, 5.19 * inch),
            _p(
                "<b>Figure 3.</b> Final held-out transition for each fire. Dark cells are observed at initialization. "
                "Blue is observed growth the model misses, green is observed growth it predicts, and red is "
                "predicted growth outside the next observation. Terrain is hillshaded.",
                styles,
                "caption",
            ),
            _p(
                "The atlas clarifies why cumulative scores can be misleading. Electra and Ridge Creek retain high "
                "cumulative overlap because their initial perimeters dominate the total area, while the advancing "
                "front is localized only in small segments. Crockets Knob and Dry Lake show broad overexpansion. "
                "Bear is a near-domain-scale overprediction. Davis has limited observed growth during the displayed "
                "interval and modest fringe error.",
                styles,
            ),
        ]
    )

    # Page 8 - suppression audit
    case = audit["crockets_knob"]
    story.extend(
        [
            PageBreak(),
            _p("6. Suppression-evidence audit", styles, "h1"),
            _p(
                "The USDA Fireline Effectiveness archive provides quality-assured spatial line geometry and final "
                "engagement outcomes for fires larger than 1,000 acres. The three classes are Held, Burned Over, "
                "and Not Engaged. This is valuable for spatial outcome research and for constructing suppression "
                "scenario priors.",
                styles,
            ),
            _table(
                [
                    ["Archive quantity", "Value", "Use in Aeolus-IA"],
                    ["Line features", f"{audit['features']:,}", "Spatial action/outcome distribution"],
                    [
                        "IRWIN incident identifiers",
                        f"{audit['incidents_by_irwin_id']:,}",
                        "Cross-incident sampling",
                    ],
                    ["Held", f"{audit['engagement']['Held']['features']:,}", "Line outcome label"],
                    [
                        "Burned Over",
                        f"{audit['engagement']['Burned Over']['features']:,}",
                        "Line outcome label",
                    ],
                    [
                        "Not Engaged",
                        f"{audit['engagement']['Not Engaged']['features']:,}",
                        "Line outcome label",
                    ],
                    [
                        "LineDateTime present",
                        f"{audit['line_datetime_present_fraction'] * 100:.2f}%",
                        "Chronology coverage is insufficient",
                    ],
                ],
                [2.13 * inch, 1.23 * inch, 3.44 * inch],
                styles,
            ),
            Spacer(1, 0.08 * inch),
            _image(figures / "crockets_knob_firelines.png", 5.5 * inch, 3.78 * inch),
            _p(
                f"<b>Figure 4.</b> Crockets Knob overlap case. The archive contains {case['features']:,} line "
                f"features: {case['engagement']['Held']['features']:,} held, "
                f"{case['engagement']['Burned Over']['features']:,} burned over, and "
                f"{case['engagement']['Not Engaged']['features']:,} not engaged. Only three features "
                f"({case['line_datetime_present_fraction'] * 100:.2f}%) carry LineDateTime.",
                styles,
                "caption",
            ),
        ]
    )

    # Page 9 - why fight is not scoreable
    story.extend(
        [
            PageBreak(),
            _p("7. Why historical fight accuracy is not yet scoreable", styles, "h1"),
            _p(
                "A final engagement label answers where reported line held relative to the final fire perimeter. "
                "It does not recover when line was constructed, what resources were present, whether the line was "
                "completed before fire arrival, or what the fire would have done without the line. The label is an "
                "outcome, not a randomized treatment effect.",
                styles,
            ),
            _p(
                "In the national archive, LineDateTime is present on 6.61% of features. CreateDate appears on "
                "94.98%, but represents database record creation and is not treated as construction time. For "
                "Crockets Knob, the only overlapping study case, timestamp coverage is 0.17%. No public, event-level "
                "water or retardant drop sequence matched to the six study incidents was identified.",
                styles,
            ),
            _p("Model-action mismatch", styles, "h2"),
            _p(
                "Aeolus-IA v0.3 exposes aerial placement actions and an implicit ground-hold behavior where treated "
                "cells intersect ground arrival. It does not yet expose time- and resource-constrained line "
                "construction, firing operations, or explicit crew actions. Even a complete historical ground-line "
                "trace could not be replayed faithfully until the action model represents those decisions.",
                styles,
            ),
            _p("Minimum evidence package for a causal replay", styles, "h2"),
            _table(
                [
                    ["Evidence", "Required fields", "Reason"],
                    [
                        "Aerial drops",
                        "Time, polygon, material, coverage level, platform",
                        "Reconstruct treatment placement and decay",
                    ],
                    [
                        "Aircraft state",
                        "Dispatch, arrival, reload, turnaround, availability",
                        "Constrain action feasibility",
                    ],
                    [
                        "Ground line",
                        "Segment geometry, construction start/end, method, production rate",
                        "Determine line state at fire arrival",
                    ],
                    [
                        "Firing operations",
                        "Ignition geometry, time, method",
                        "Represent intentional fire spread",
                    ],
                    [
                        "Assignments",
                        "Resource-to-division/task history and handoffs",
                        "Recover coordination constraints",
                    ],
                    [
                        "Observations",
                        "Perimeter acquisition time and uncertainty",
                        "Separate action timing from observation delay",
                    ],
                    [
                        "Counterfactual design",
                        "Matched controls, causal model, or calibrated ensemble",
                        "Estimate effect rather than association",
                    ],
                ],
                [1.20 * inch, 3.12 * inch, 2.48 * inch],
                styles,
            ),
            Spacer(1, 0.10 * inch),
            _callout(
                "Current status",
                "Historical incident replays can compare alternative policies inside the same simulator. They are "
                "counterfactual scenario studies. They should not be labeled historical suppression accuracy until "
                "action chronology, resource state, and causal assumptions are explicit.",
                styles,
                background=PALE_RED,
                accent=RED,
            ),
        ]
    )

    # Page 10 - research position
    story.extend(
        [
            PageBreak(),
            _p("8. Position relative to current wildfire modeling", styles, "h1"),
            _p(
                "The comparison below separates free forward forecasting from observation reconstruction. Metrics "
                "reported by mapping or assimilation systems cannot be used as direct leaderboards against a free "
                "forecast because those systems condition on observations from the evaluation period.",
                styles,
            ),
            _table(
                [
                    ["System / approach", "State and forcing", "Observation use", "Validation relevance"],
                    [
                        "Aeolus-IA v0.3",
                        "Raster surface/crown spread, fuel moisture, weather, spotting; suppression agents",
                        "Observed perimeter only at forecast initialization",
                        "This study: 24 held-out daily-scale transitions; cumulative IoU 0.611 calibrated",
                    ],
                    [
                        "ELMFIRE",
                        "Level-set spread with ensembles over uncertain inputs",
                        "Historical and real-time perimeter overlays; operational forecast workflow",
                        "Documents validation against observed perimeters and 2022 qualitative forecast review",
                    ],
                    [
                        "WRF-SFIRE",
                        "Coupled atmosphere-fire model with Rothermel-type level set",
                        "Research methods periodically assimilate observed perimeters",
                        "Captures fire-atmosphere feedback absent from v0.3; materially greater compute cost",
                    ],
                    [
                        "GOFER / FEDS",
                        "Satellite-derived progression mapping",
                        "Current and retrospective active-fire observations",
                        "Published final-perimeter mapping IoU 0.77 GOFER, 0.83 FEDS; reconstruction, not free forecast",
                    ],
                    [
                        "Conditional generative reconstruction",
                        "WRF-SFIRE-trained arrival-time generator",
                        "Conditioned on VIIRS, GOES-derived times, and terrain",
                        "Mean Dice 0.81 on five fires; not comparable to an observation-free forward step",
                    ],
                    [
                        "GPU RL fire environments",
                        "Vectorized cellular or raster spread designed for throughput",
                        "Usually synthetic training distributions",
                        "Useful for policy optimization; empirical incident fidelity remains a separate requirement",
                    ],
                ],
                [1.23 * inch, 2.12 * inch, 1.72 * inch, 1.73 * inch],
                styles,
            ),
            Spacer(1, 0.10 * inch),
            _callout(
                "Research position",
                "Aeolus-IA now has the data contract and scoring harness needed for empirical model development. "
                "Its present spread skill is below operational forecast practice. Its differentiating research "
                "question remains resource-constrained multi-agent suppression under uncertainty, which requires "
                "a calibrated ensemble environment rather than a single deterministic fitted fire.",
                styles,
                background=PALE_BLUE,
                accent=BLUE,
            ),
            _p("What this benchmark adds", styles, "h2"),
            _bullet(
                "A reproducible importer for a current, high-resolution NIROPS progression archive.", styles
            ),
            _bullet(
                "Frozen incident bundles with aligned terrain, fuels, canopy, weather, and observation times.",
                styles,
            ),
            _bullet("Held-out interval selection and persistence as a mandatory baseline.", styles),
            _bullet("Growth, boundary, and area metrics with incident-cluster uncertainty.", styles),
            _bullet(
                "A concrete evidence audit separating spread validation from suppression validation.", styles
            ),
        ]
    )

    # Page 11 - limitations
    story.extend(
        [
            PageBreak(),
            _p("9. Limitations and threats to validity", styles, "h1"),
            _table(
                [
                    ["Threat", "Effect on result", "Mitigation / next test"],
                    [
                        "Six-fire convenience sample",
                        "Uncertainty intervals are wide; incident composition matters.",
                        "Run a stratified benchmark across region, fuel, size, and fire regime.",
                    ],
                    [
                        "Daily/sub-daily observation gaps",
                        "Sub-interval spread and suppression timing are hidden.",
                        "Add GOFER/FEDS for temporal context while retaining NIROPS as spatial reference.",
                    ],
                    [
                        "Coarse POWER weather",
                        "Local wind and humidity near complex terrain are unresolved.",
                        "Use RAWS plus HRRR/WRF forcing; quantify meteorological uncertainty.",
                    ],
                    [
                        "Static fuel moisture approximation",
                        "Rate of spread and extinction state may be biased.",
                        "Initialize dead/live classes and update from weather and observations.",
                    ],
                    [
                        "Observed target includes suppression",
                        "A no-suppression forecast can overgrow even with correct free-spread physics.",
                        "Mask known treatments or jointly model time-indexed actions.",
                    ],
                    [
                        "128 x 128, 134-188 m cells",
                        "Narrow features, line barriers, and spotting localization are under-resolved.",
                        "Run adaptive or nested 20-60 m active-front grids and test convergence.",
                    ],
                    [
                        "Scalar calibration",
                        "A fitted effective factor absorbs multiple mechanisms and transfers poorly.",
                        "Infer uncertain parameters sequentially with ensembles and regularization.",
                    ],
                    [
                        "Single deterministic trajectory",
                        "No calibrated forecast probability or coverage.",
                        "Score ensembles using reliability, CRPS, Brier, and containment coverage.",
                    ],
                    [
                        "No action chronology",
                        "Suppression effects are not identifiable.",
                        "Obtain incident action logs or design prospective instrumented exercises.",
                    ],
                ],
                [1.42 * inch, 2.40 * inch, 2.98 * inch],
                styles,
            ),
            Spacer(1, 0.12 * inch),
            _p("Metric caveats", styles, "h2"),
            _p(
                "IoU is grid- and extent-dependent. Boundary metrics weight every boundary sample equally and can "
                "overrepresent highly crenulated perimeters. The one-cell tolerance varies physically with cell "
                "size. Symmetric-difference area is dominated by large fires. The report therefore uses all metric "
                "families and retains incident-level outputs in the accompanying CSV and JSON.",
                styles,
            ),
        ]
    )

    # Page 12 - work program
    story.extend(
        [
            PageBreak(),
            _p("10. Engineering and research work program", styles, "h1"),
            _p(
                "The next development cycle should be organized around measurable release gates. Simulator "
                "throughput matters for MARL, but it is downstream of observation fit and uncertainty calibration.",
                styles,
            ),
            _table(
                [
                    ["Priority", "Work package", "Acceptance evidence"],
                    [
                        "P0",
                        "Automate NIROPS benchmark expansion and incident stratification.",
                        "At least 100 fires; frozen train/calibration/test splits; documented exclusions.",
                    ],
                    [
                        "P0",
                        "Replace point reanalysis with RAWS plus HRRR/WRF spatial forcing.",
                        "Weather provenance, gaps, uncertainty, and ablation against POWER baseline.",
                    ],
                    [
                        "P0",
                        "Add perimeter state assimilation and ensemble parameter updates.",
                        "Forward forecast after assimilation beats persistence on held-out boundary and growth metrics.",
                    ],
                    [
                        "P1",
                        "Implement dynamic dead/live fuel moisture and precipitation response.",
                        "Laboratory/reference cases plus historical ablation improvement.",
                    ],
                    [
                        "P1",
                        "Add stochastic spotting with transport, landing, ignition delay, and censoring.",
                        "Distributional verification against observed spot-fire distances where available.",
                    ],
                    [
                        "P1",
                        "Represent explicit crew/dozer line construction and firing operations.",
                        "Action feasibility tests; replay on complete synthetic logs; schema aligned to incident data.",
                    ],
                    [
                        "P1",
                        "Add nested active-front resolution and convergence tests.",
                        "20-60 m front grids; metric stability across refinement levels.",
                    ],
                    [
                        "P2",
                        "Train robust MARL policies over posterior simulator ensembles.",
                        "Policy ranking stable under parameter, weather, sensing, and resource perturbations.",
                    ],
                    [
                        "P2",
                        "Run prospective suppression-data collection.",
                        "Complete time-stamped action/resource trace and independent observation stream.",
                    ],
                ],
                [0.55 * inch, 3.08 * inch, 3.17 * inch],
                styles,
            ),
            Spacer(1, 0.11 * inch),
            _callout(
                "Release gate",
                "A spread model should beat persistence on held-out cumulative and boundary metrics and show "
                "material advancing-front skill across incident clusters. A policy model should additionally "
                "demonstrate robustness over the calibrated uncertainty ensemble. Historical suppression claims "
                "require matched chronological actions.",
                styles,
                background=PALE_GREEN,
                accent=GREEN,
            ),
        ]
    )

    # Page 13 - reproducibility
    story.extend(
        [
            PageBreak(),
            _p("11. Reproducibility package", styles, "h1"),
            _p(
                "The study implementation, manifest, frozen result tables, examples, and figure builders are "
                "versioned with the repository. The large source archives and prepared incident assets remain "
                "external because of their size and licensing/provenance role.",
                styles,
            ),
            _p("Commands", styles, "h2"),
            _p(
                "<font name='Courier' size='7.2'>"
                "aeolus-study prepare --manifest configs/historical_validation.yaml<br/>"
                "&nbsp;&nbsp;--source-shapefile /path/to/NIROPS_2020_2024_R1_R6.shp<br/>"
                "&nbsp;&nbsp;--out outputs/historical-validation/incidents<br/><br/>"
                "aeolus-study run --manifest configs/historical_validation.yaml<br/>"
                "&nbsp;&nbsp;--prepared-root outputs/historical-validation/incidents<br/>"
                "&nbsp;&nbsp;--out outputs/historical-validation/results"
                "</font>",
                styles,
                "body",
            ),
            _p("Frozen result artifacts", styles, "h2"),
            _table(
                [
                    ["Artifact", "Contents"],
                    [
                        "historical_validation_results.json",
                        "Manifest, calibrations, 72 method-interval forecasts, cluster summaries, interpretation constraints",
                    ],
                    ["forecast_metrics.csv", "Flat interval-level metrics for independent analysis"],
                    [
                        "historical_validation_examples.npz",
                        "Six final-interval raster examples used in the error atlas",
                    ],
                    ["fireline_archive_audit.json", "National and Crockets Knob feature/timestamp counts"],
                    [
                        "figures/*.png",
                        "Publication-ready aggregate, transfer, atlas, location, and fireline figures",
                    ],
                ],
                [2.30 * inch, 4.50 * inch],
                styles,
            ),
            _p("Verification", styles, "h2"),
            _bullet(
                "30 Python tests pass, including importer, metric, fire-behavior, environment, training, bundle, and replay coverage.",
                styles,
            ),
            _bullet("Ruff static checks pass across source, tests, and tools.", styles),
            _bullet(
                "The source and binary distributions build, and the installed console entry point exposes prepare/run workflows.",
                styles,
            ),
            _bullet("The PDF is rendered to page images and visually inspected before delivery.", styles),
            Spacer(1, 0.10 * inch),
            _callout(
                "Determinism",
                "Study seed 20260728 controls simulator initialization and incident-cluster bootstrap draws. "
                "External service products may change; prepared incident bundles and source checksums are the "
                "reproduction boundary for this result.",
                styles,
            ),
        ]
    )

    # Pages 14-15 references
    story.extend(
        [
            PageBreak(),
            _p("References", styles, "h1"),
            _p(
                "1. Magstadt, S. et al. (2026). <i>A high spatial resolution daily fire perimeter progression "
                "dataset for wildfires in the Western United States: 2020-2024.</i> Mendeley Data, v1. "
                "<link href='https://doi.org/10.17632/95rj5d379g.1' color='#2C718E'>"
                "doi:10.17632/95rj5d379g.1</link>.",
                styles,
                "ref",
            ),
            _p(
                "2. Arkowitz, A. et al. (2025). <i>Quality assured spatial dataset of wildfire containment "
                "firelines and engagement outcomes 2017 to 2024.</i> Scientific Data 12, 897. "
                "<link href='https://doi.org/10.1038/s41597-025-05208-0' color='#2C718E'>"
                "doi:10.1038/s41597-025-05208-0</link>.",
                styles,
                "ref",
            ),
            _p(
                "3. Liu, T. et al. (2024). <i>Systematically tracking the hourly progression of large wildfires "
                "using GOES satellite observations.</i> Earth System Science Data 16, 1395-1424. "
                "<link href='https://doi.org/10.5194/essd-16-1395-2024' color='#2C718E'>"
                "doi:10.5194/essd-16-1395-2024</link>.",
                styles,
                "ref",
            ),
            _p(
                "4. Kochanski, A. K. et al. (2023). <i>Analysis of methods for assimilating fire perimeters into "
                "a coupled fire-atmosphere model.</i> Frontiers in Forests and Global Change 6. "
                "<link href='https://doi.org/10.3389/ffgc.2023.1203578' color='#2C718E'>"
                "doi:10.3389/ffgc.2023.1203578</link>.",
                styles,
                "ref",
            ),
            _p(
                "5. Shaddy, B. et al. (2025). <i>Generative Algorithms for Wildfire Progression Reconstruction "
                "from Multi-Modal Satellite Active Fire Measurements and Terrain Height.</i> "
                "<link href='https://arxiv.org/abs/2506.10404' color='#2C718E'>arXiv:2506.10404</link>.",
                styles,
                "ref",
            ),
            _p(
                "6. Mandel, J. et al. (2012). <i>Assimilation of Perimeter Data and Coupling with Fuel Moisture "
                "in a Wildland Fire - Atmosphere DDDAS.</i> "
                "<link href='https://arxiv.org/abs/1203.2230' color='#2C718E'>arXiv:1203.2230</link>.",
                styles,
                "ref",
            ),
            _p(
                "7. Rochoux, M. C. et al. (2014). <i>Towards predictive simulation of wildfire spread at regional "
                "scale using ensemble-based data assimilation to correct the fire front position.</i> Fire Safety "
                "Science 11, 1443-1456. "
                "<link href='https://doi.org/10.3801/IAFSS.FSS.11-1443' color='#2C718E'>"
                "doi:10.3801/IAFSS.FSS.11-1443</link>.",
                styles,
                "ref",
            ),
            _p(
                "8. NASA POWER. <i>Hourly API documentation.</i> "
                "<link href='https://power.larc.nasa.gov/docs/services/api/temporal/hourly/' color='#2C718E'>"
                "power.larc.nasa.gov/docs/services/api/temporal/hourly/</link>.",
                styles,
                "ref",
            ),
            _p(
                "9. ELMFIRE. <i>Validation documentation.</i> "
                "<link href='https://elmfire.io/validation.html' color='#2C718E'>"
                "elmfire.io/validation.html</link>.",
                styles,
                "ref",
            ),
            _p(
                "10. USDA Forest Service Rocky Mountain Research Station. <i>Fireline Effectiveness Dashboard.</i> "
                "<link href='https://research.fs.usda.gov/rmrs/products/dataandtools/fireline-effectiveness-fle-dashboard' "
                "color='#2C718E'>research.fs.usda.gov/.../fireline-effectiveness-fle-dashboard</link>.",
                styles,
                "ref",
            ),
            _p(
                "11. Gannon, B. M. et al. (2020). <i>A geospatial framework to assess fireline effectiveness for "
                "large wildfires in the western USA.</i> Fire 3(3), 43. "
                "<link href='https://research.fs.usda.gov/treesearch/60734' color='#2C718E'>"
                "USFS Treesearch 60734</link>.",
                styles,
                "ref",
            ),
            _p(
                "12. Cakir, U. et al. (2025). <i>JaxWildfire: A GPU-Accelerated Wildfire Simulator for "
                "Reinforcement Learning.</i> "
                "<link href='https://arxiv.org/abs/2512.06102' color='#2C718E'>arXiv:2512.06102</link>.",
                styles,
                "ref",
            ),
            Spacer(1, 0.16 * inch),
            _callout(
                "Citation boundary",
                "GOFER/FEDS and conditional-generative scores cited above are mapping or reconstruction results. "
                "They are included to locate the study in current practice, not as directly comparable forecast "
                "leaderboard entries.",
                styles,
                background=PALE_ORANGE,
                accent=ORANGE,
            ),
            Spacer(1, 0.18 * inch),
            _p("End of research note", styles, "kicker"),
        ]
    )
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    styles = _styles()
    doc = BaseDocTemplate(
        str(args.out),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.55 * inch,
        title="Historical Validation of Aeolus-IA v0.3",
        author="Aeolus-IA research",
        subject="Airborne-infrared spread hindcasts and suppression-evidence audit",
        creator="Aeolus-IA reproducible study pipeline",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_header_footer)])
    doc.build(_build_story(results, audit, args.figures, styles))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
