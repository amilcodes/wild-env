"""Build the corrected historical-accuracy research paper."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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

INK = colors.HexColor("#17232D")
SLATE = colors.HexColor("#526273")
MUTED = colors.HexColor("#758190")
BLUE = colors.HexColor("#32768F")
PALE_BLUE = colors.HexColor("#E9F2F5")
ORANGE = colors.HexColor("#CB683A")
PALE_ORANGE = colors.HexColor("#F8EDE7")
GREEN = colors.HexColor("#3D8068")
PALE_GREEN = colors.HexColor("#E9F2EE")
RED = colors.HexColor("#B84C42")
PALE_RED = colors.HexColor("#F7EAE8")
PURPLE = colors.HexColor("#66599A")
LINE = colors.HexColor("#CDD5DC")
WHITE = colors.white

METHODS = (
    "persistence",
    "raw_physics",
    "history_raw_physics",
    "calibrated_physics",
    "history_calibrated_physics",
    "calibrated_ensemble",
    "history_calibrated_ensemble",
)
LABELS = {
    "persistence": "Persistence",
    "raw_physics": "Raw",
    "history_raw_physics": "Raw + history",
    "calibrated_physics": "Calibrated",
    "history_calibrated_physics": "Calibrated + history",
    "calibrated_ensemble": "Ensemble",
    "history_calibrated_ensemble": "Ensemble + history",
}


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=29,
            textColor=INK,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11.5,
            leading=15,
            textColor=SLATE,
            spaceAfter=9,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=BLUE,
            tracking=1.1,
            spaceAfter=9,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2",
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
            leading=13.1,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.8,
            textColor=SLATE,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10.2,
            textColor=SLATE,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=19,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=SLATE,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=8.6,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.2,
            textColor=WHITE,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.3,
            leading=14,
            textColor=INK,
            leftIndent=7,
            rightIndent=7,
            spaceAfter=2,
        ),
        "ref": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.4,
            textColor=INK,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=4,
        ),
    }


def para(text: str, sheet: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, sheet[name])


def bullet(text: str, sheet: dict[str, ParagraphStyle], *, small: bool = False) -> Paragraph:
    return Paragraph(
        f"<bullet>&bull;</bullet>{text}",
        sheet["small" if small else "body"],
    )


def page_chrome(canvas: Any, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    width, height = letter
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.4)
        canvas.line(0.65 * inch, height - 0.50 * inch, width - 0.65 * inch, height - 0.50 * inch)
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(SLATE)
        canvas.drawString(0.65 * inch, height - 0.39 * inch, "AEOLUS-IA HISTORICAL ACCURACY V3")
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(width - 0.65 * inch, 0.36 * inch, f"{doc.page}")
    canvas.restoreState()


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def styled_table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    repeat_rows: int = 1,
    highlight_last: bool = False,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F5F7F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_last:
        commands.append(("BACKGROUND", (0, -1), (-1, -1), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def metric(summary: dict[str, Any], method: str, key: str) -> float:
    return float(summary[method][key]["mean"])


def build_story(
    analysis: dict[str, Any],
    figure_dir: Path,
    sheet: dict[str, ParagraphStyle],
) -> list[Any]:
    aggregate = analysis["aggregate_summaries"]
    active = analysis["active_growth_summaries"]
    ablation = analysis["paired_arrival_history_ablation"]
    probability = analysis["probabilistic_skill_against_persistence"]
    calibration = analysis["calibration_diagnostics"]
    protocol = analysis["protocol_checks"]
    story: list[Any] = []

    story.extend(
        [
            Spacer(1, 0.25 * inch),
            para("HISTORICAL ACCURACY STUDY  |  CORRECTED V3", sheet, "kicker"),
            para("Coupled-state wildfire hindcasts", sheet, "title"),
            para(
                "Held-out evaluation of fire spread, arrival-history reconstruction, "
                "fuel-moisture spin-up, and ensemble uncertainty against six NIROPS incidents.",
                sheet,
                "subtitle",
            ),
            HRFlowable(width="100%", thickness=2.0, color=BLUE, spaceBefore=3, spaceAfter=15),
        ]
    )
    metric_cards = Table(
        [
            [
                para("6", sheet, "metric"),
                para("24", sheet, "metric"),
                para("168", sheet, "metric"),
                para("0.7458", sheet, "metric"),
            ],
            [
                para("incidents", sheet, "metric_label"),
                para("held-out transitions", sheet, "metric_label"),
                para("scored forecasts", sheet, "metric_label"),
                para("best physics IoU", sheet, "metric_label"),
            ],
        ],
        colWidths=[1.68 * inch] * 4,
    )
    metric_cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ]
        )
    )
    story.extend(
        [
            metric_cards,
            Spacer(1, 14),
            para("Result", sheet, "h1"),
            para(
                "Causal two-perimeter arrival history improves all three physics families. "
                "The history ensemble is the strongest physics method, with cumulative IoU "
                "0.7458, active-growth one-cell-tolerance F1 0.1776, and mean symmetric "
                "boundary displacement 396.3 m.",
                sheet,
            ),
        ]
    )
    callout = Table(
        [
            [
                para(
                    "Persistence remains substantially better on cumulative extent "
                    "(IoU 0.8732; boundary displacement 156.5 m). The ensemble has "
                    "positive active-front probability skill but overpredicts growth "
                    "over the full domain. This is research evidence, not an operational "
                    "forecasting claim.",
                    sheet,
                    "callout",
                )
            ]
        ],
        colWidths=[6.72 * inch],
    )
    callout.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_ORANGE),
                ("BOX", (0, 0), (-1, -1), 0.7, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend(
        [
            callout,
            Spacer(1, 12),
            para("What changed", sheet, "h2"),
            bullet(
                "<b>Forcing:</b> every forecast now samples weather at its absolute observation timestamp.",
                sheet,
            ),
            bullet(
                "<b>Moisture:</b> 1 h, 10 h, and 100 h dead fuels receive a 14-day weather spin-up with hysteretic drying/wetting equilibria.",
                sheet,
            ),
            bullet(
                "<b>Fuels:</b> exact FBFM40 loading and bed depth replace a coarse family proxy; fuel loading no longer double-scales spread.",
                sheet,
            ),
            bullet(
                "<b>Numerics:</b> subgrid front support is preserved until the level-set interface crosses the next cell or is suppressed.",
                sheet,
            ),
            bullet(
                "<b>Initialization:</b> causal two-perimeter replay reconstructs arrival time, fire age, fuel consumption, and current front state.",
                sheet,
            ),
            Spacer(1, 7),
            para("Study boundary", sheet, "h2"),
            para(
                "No historical suppression-effectiveness or learned-policy claim is made. "
                "The perimeter source does not encode a complete action and resource trace, "
                "and the weather remains coarse point forcing.",
                sheet,
            ),
            Spacer(1, 10),
            para("Aeolus-IA research program | 29 July 2026", sheet, "small"),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("1. Protocol and implementation audit", sheet, "h1"),
            para(
                "Each incident contributes one calibration transition followed by four later "
                "held-out transitions. All transitions from one incident remain a statistical "
                "cluster. Bootstrap intervals therefore resample six incident trajectories, "
                "not 24 nominally independent days.",
                sheet,
            ),
        ]
    )
    source_rows = [
        [
            para("Source", sheet, "table_head"),
            para("Variables", sheet, "table_head"),
            para("Role", sheet, "table_head"),
            para("Limitation", sheet, "table_head"),
        ],
        [
            para("NIROPS", sheet, "table"),
            para("Interpreted perimeters", sheet, "table"),
            para("State and target", sheet, "table"),
            para("Daily; uncertainty not modeled", sheet, "table"),
        ],
        [
            para("USGS 3DEP", sheet, "table"),
            para("Elevation", sheet, "table"),
            para("Terrain and slope", sheet, "table"),
            para("Resampled to 128 cells", sheet, "table"),
        ],
        [
            para("LANDFIRE 2025", sheet, "table"),
            para("FBFM40 and canopy", sheet, "table"),
            para("Fuel state", sheet, "table"),
            para("Not contemporaneous; fixed curing", sheet, "table"),
        ],
        [
            para("NASA POWER", sheet, "table"),
            para("Hourly wind, T, RH, rain", sheet, "table"),
            para("Weather and moisture", sheet, "table"),
            para("Coarse incident point", sheet, "table"),
        ],
    ]
    story.extend(
        [
            styled_table(source_rows, [0.95 * inch, 1.45 * inch, 1.35 * inch, 2.97 * inch]),
            Spacer(1, 8),
            para("Seven paired forecast methods", sheet, "h2"),
            para(
                "Persistence, raw physics, raw physics with history, calibrated physics, "
                "calibrated physics with history, a 12-particle calibrated ensemble, and the "
                "same ensemble with history all receive identical terrain, fuel, weather, "
                "start perimeter, and duration. The ensemble varies spread, wind exposure, "
                "direction bias, and dead-fuel moisture.",
                sheet,
            ),
            para("Integrity checks", sheet, "h2"),
            bullet(f"Exactly {analysis['held_out_intervals']} held-out forecasts per method.", sheet),
            bullet(
                f"All physics forecasts reached their requested timestamp: {str(protocol['all_physics_forecasts_reached_requested_time']).lower()}.",
                sheet,
            ),
            bullet(
                f"Maximum time integration overshoot: {protocol['maximum_time_overshoot_min']} min.", sheet
            ),
            bullet(
                f"All history forecasts use only prior and current perimeters: {str(protocol['all_history_forecasts_are_causal']).lower()}.",
                sheet,
            ),
            para("Scoring", sheet, "h2"),
            para(
                "Cumulative IoU measures final burned extent. New-growth IoU and one-cell-"
                "tolerance F1 isolate the advancing front. Boundary displacement, 95th-"
                "percentile Hausdorff distance, symmetric-difference area, area bias, and "
                "Brier variants expose spatial and probabilistic failure modes that cumulative "
                "overlap alone conceals.",
                sheet,
            ),
            PageBreak(),
            para("2. Held-out performance", sheet, "h1"),
            fit_image(figure_dir / "historical_accuracy_summary.png", 6.85 * inch, 5.05 * inch),
            para(
                "<b>Figure 1.</b> Mean held-out performance. Bars show incident-cluster "
                "bootstrap 95% intervals. Arrival history improves every physics-family "
                "cumulative IoU and boundary mean. Persistence benefits from the large "
                "overlap between successive daily cumulative perimeters.",
                sheet,
                "caption",
            ),
        ]
    )

    result_rows: list[list[Any]] = [
        [
            para("Method", sheet, "table_head"),
            para("Extent IoU", sheet, "table_head"),
            para("Growth F1", sheet, "table_head"),
            para("Active F1", sheet, "table_head"),
            para("Boundary m", sheet, "table_head"),
            para("Sym. diff. km2", sheet, "table_head"),
        ]
    ]
    for method in METHODS:
        result_rows.append(
            [
                para(LABELS[method], sheet, "table"),
                para(f"{metric(aggregate, method, 'metrics.iou'):.4f}", sheet, "table"),
                para(f"{metric(aggregate, method, 'growth_tolerance_1_cell.f1'):.4f}", sheet, "table"),
                para(f"{metric(active, method, 'growth_tolerance_1_cell.f1'):.4f}", sheet, "table"),
                para(
                    f"{metric(aggregate, method, 'boundary.mean_symmetric_distance_m'):.1f}", sheet, "table"
                ),
                para(f"{metric(aggregate, method, 'metrics.symmetric_difference_km2'):.3f}", sheet, "table"),
            ]
        )
    story.extend(
        [
            styled_table(
                result_rows,
                [1.50 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 1.05 * inch, 1.15 * inch],
                highlight_last=True,
            ),
            para(
                "The calibrated deterministic forecast is worse than raw physics. A scalar "
                "fit on one early day absorbs unresolved weather, fuel, suppression, and "
                "observation effects, then fails to transfer. Bear selects the upper search "
                "boundary (5.0), which is a calibration warning rather than evidence for a "
                "physical fivefold spread correction.",
                sheet,
                "caption",
            ),
            PageBreak(),
            para("3. Arrival-history reconstruction", sheet, "h1"),
            fit_image(figure_dir / "arrival_history_ablation.png", 6.85 * inch, 4.2 * inch),
            para(
                "<b>Figure 2.</b> Paired effect of using the previous perimeter to reconstruct "
                "arrival history and coupled fire state. Intervals resample incidents. For "
                "distance and area disagreement, values below zero are improvements.",
                sheet,
                "caption",
            ),
        ]
    )
    pairs = [
        ("Raw", "raw_physics__to__history_raw_physics"),
        ("Calibrated", "calibrated_physics__to__history_calibrated_physics"),
        ("Ensemble", "calibrated_ensemble__to__history_calibrated_ensemble"),
    ]
    ablation_rows = [
        [
            para("Pair", sheet, "table_head"),
            para("Delta IoU", sheet, "table_head"),
            para("Delta growth F1", sheet, "table_head"),
            para("Delta boundary m", sheet, "table_head"),
            para("Delta sym. diff. km2", sheet, "table_head"),
        ]
    ]
    for label, key in pairs:
        entry = ablation[key]
        ablation_rows.append(
            [
                para(label, sheet, "table"),
                para(f"{entry['metrics.iou']['mean_delta']:+.4f}", sheet, "table"),
                para(f"{entry['growth_tolerance_1_cell.f1']['mean_delta']:+.4f}", sheet, "table"),
                para(f"{entry['boundary.mean_symmetric_distance_m']['mean_delta']:+.1f}", sheet, "table"),
                para(f"{entry['metrics.symmetric_difference_km2']['mean_delta']:+.3f}", sheet, "table"),
            ]
        )
    calibrated_iou = ablation["calibrated_physics__to__history_calibrated_physics"]["metrics.iou"]
    calibrated_boundary = ablation["calibrated_physics__to__history_calibrated_physics"][
        "boundary.mean_symmetric_distance_m"
    ]
    story.extend(
        [
            styled_table(ablation_rows, [1.30 * inch, 1.05 * inch, 1.35 * inch, 1.35 * inch, 1.55 * inch]),
            Spacer(1, 7),
            para(
                f"The clearest effect is calibrated physics: IoU rises {calibrated_iou['mean_delta']:.4f} "
                f"(95% interval {calibrated_iou['ci95_low']:.4f} to {calibrated_iou['ci95_high']:.4f}) "
                f"and boundary displacement changes {calibrated_boundary['mean_delta']:.1f} m "
                f"(95% interval {calibrated_boundary['ci95_low']:.1f} to "
                f"{calibrated_boundary['ci95_high']:.1f} m). The raw and ensemble effects "
                "have the same aggregate direction, but some intervals overlap zero.",
                sheet,
            ),
            para(
                "Interpretation: a perimeter is not a complete dynamical state. Reconstructing "
                "where the fire arrived recently improves fuel age, residual burning, interface "
                "orientation, and the immediate direction of propagation. It does not correct "
                "the atmospheric forcing or reveal hidden suppression.",
                sheet,
            ),
            PageBreak(),
            para("4. Spatial failure modes", sheet, "h1"),
            fit_image(figure_dir / "history_ensemble_atlas.png", 6.85 * inch, 5.1 * inch),
            para(
                "<b>Figure 3.</b> Final held-out transition for each incident, history ensemble "
                "thresholded at p &gt;= 0.5. Dark: initial perimeter. Blue: observed growth "
                "only. Green: matched growth. Red: predicted growth only. The atlas shows "
                "systematic false growth on Electra and Dry Lake, localized useful advance on "
                "Crockets Knob, and near-static cumulative overlap on Ridge Creek.",
                sheet,
                "caption",
            ),
            para("Incident heterogeneity", sheet, "h2"),
            para(
                "Mean history-ensemble IoU ranges from 0.628 to 0.932 and growth F1 from "
                "0.036 to 0.306. Electra has +11.1 km2 mean area bias, while Bear has "
                "-8.1 km2. One global correction cannot resolve this sign-changing error. "
                "Local weather, suppression, observation timing, and fuel condition must be "
                "represented or inferred.",
                sheet,
            ),
            PageBreak(),
            para("5. Moisture state and probabilistic skill", sheet, "h1"),
            fit_image(figure_dir / "fuel_moisture_spinup.png", 6.85 * inch, 3.75 * inch),
            para(
                "<b>Figure 4.</b> Fourteen-day pre-incident dead-fuel moisture spin-up. "
                "One-hour fuels respond fastest; 100-hour fuels preserve multi-day memory. "
                "All curves remain incident-point estimates because the source forcing is not "
                "spatially resolved.",
                sheet,
                "caption",
            ),
        ]
    )
    probability_rows = [
        [
            para("Ensemble", sheet, "table_head"),
            para("Whole ordinary", sheet, "table_head"),
            para("Whole balanced", sheet, "table_head"),
            para("Active ordinary", sheet, "table_head"),
            para("Active balanced", sheet, "table_head"),
        ]
    ]
    for method in ("calibrated_ensemble", "history_calibrated_ensemble"):
        p = probability[method]
        probability_rows.append(
            [
                para(LABELS[method], sheet, "table"),
                para(f"{100 * p['whole_domain_brier_skill']:+.1f}%", sheet, "table"),
                para(f"{100 * p['whole_domain_balanced_brier_skill']:+.1f}%", sheet, "table"),
                para(f"{100 * p['active_domain_brier_skill']:+.1f}%", sheet, "table"),
                para(f"{100 * p['active_domain_balanced_brier_skill']:+.1f}%", sheet, "table"),
            ]
        )
    story.extend(
        [
            styled_table(probability_rows, [1.55 * inch, 1.20 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch]),
            Spacer(1, 7),
            para(
                "Ordinary whole-domain Brier skill is negative because a large grid contains "
                "many easy non-growth cells and the ensemble spreads probability too broadly. "
                "Balanced and active-domain scores are positive, showing useful discrimination "
                "near the evolving front. Before any probability product is used externally, "
                "it requires out-of-incident calibration, reliability diagrams by distance to "
                "front, and coverage tests for arrival time.",
                sheet,
            ),
            para("Calibration diagnostics", sheet, "h2"),
        ]
    )
    calibration_rows = [
        [
            para("Incident", sheet, "table_head"),
            para("Spread", sheet, "table_head"),
            para("Raw ESS", sheet, "table_head"),
            para("Tempered ESS", sheet, "table_head"),
            para("Beta", sheet, "table_head"),
            para("Boundary", sheet, "table_head"),
        ]
    ]
    for row in calibration:
        incident = row["incident_code"].split("_")[-1]
        calibration_rows.append(
            [
                para(incident, sheet, "table"),
                para(f"{row['selected_spread_adjustment']:.3f}", sheet, "table"),
                para(f"{row['raw_effective_sample_size']:.2f}", sheet, "table"),
                para(f"{row['tempered_effective_sample_size']:.2f}", sheet, "table"),
                para(f"{row['likelihood_tempering_beta']:.3f}", sheet, "table"),
                para("upper" if row["at_upper_search_boundary"] else "-", sheet, "table"),
            ]
        )
    story.extend(
        [
            styled_table(
                calibration_rows,
                [1.45 * inch, 0.85 * inch, 0.85 * inch, 1.05 * inch, 0.85 * inch, 0.85 * inch],
            ),
            PageBreak(),
            para("6. Limits and closure program", sheet, "h1"),
            para("2.1 Historical forcing and observations", sheet, "h2"),
            bullet(
                "<b>Incident wind:</b> replace point POWER winds with HRRR analysis fields; assimilate quality-controlled RAWS innovations; score wind before spread.",
                sheet,
            ),
            bullet(
                "<b>Spatial moisture:</b> drive fuel classes with gridded T, RH, rain, radiation, and canopy exposure; add live moisture and herbaceous curing; validate at fuel-stick sites.",
                sheet,
            ),
            bullet(
                "<b>Sub-daily observations:</b> add GOFER/FEDS progressions with sensor footprints, clouds, geolocation, and acquisition intervals represented as likelihoods.",
                sheet,
            ),
            bullet(
                "<b>Suppression confounding:</b> ingest time-indexed fireline, aerial-drop, firing, resource, and engagement records; mask or jointly infer affected front segments.",
                sheet,
            ),
            para(
                "<b>Gate:</b> 30-50 incidents; incident, geography, and year held out; "
                "1/3/6/12/24-hour targets; positive advancing-front skill over persistence "
                "with incident-cluster intervals excluding zero.",
                sheet,
            ),
            para("2.2 Coupled fire state and behavior", sheet, "h2"),
            bullet(
                "<b>Sequential assimilation:</b> localized ensemble filtering over arrival time, spread correction, wind correction, moisture, and spotting; assess the next observation.",
                sheet,
            ),
            bullet(
                "<b>Coupled-model hierarchy:</b> WRF-SFIRE and QUIC-Fire teachers for bounded terrain-wind, fire-flow, spread, and uncertainty corrections.",
                sheet,
            ),
            bullet(
                "<b>Dynamic fuels:</b> curing, live woody and foliar moisture, dated disturbance, and crown transition; keep exact FBFM40 loading.",
                sheet,
            ),
            bullet(
                "<b>Spotting:</b> independently validate lofting, transport, landing, ignition delay, and spot-distance distributions before enabling historical scores.",
                sheet,
            ),
            para(
                "<b>Gate:</b> independent error reductions for wind, moisture, spread, and "
                "spotting, with stable held-out skill across fuel and weather regimes.",
                sheet,
            ),
            para("2.3 Statistics and RL implications", sheet, "h2"),
            bullet(
                "<b>Generalization:</b> stratify incidents by geography, year, fuels, wind, topography, size, and observation source; never split one incident across train and test.",
                sheet,
            ),
            bullet(
                "<b>Calibration:</b> replace the scalar multiplier with hierarchical physical priors, posterior uncertainty, identifiability checks, and transfer tests.",
                sheet,
            ),
            bullet(
                "<b>Probability:</b> add reliability, expected calibration error, log score, Brier decomposition, CRPS, and arrival-time coverage by horizon and front distance.",
                sheet,
            ),
            bullet(
                "<b>Policy training:</b> sample the empirically supported posterior rather than one fitted fire; freeze region/year/incident-held-out policy evaluation.",
                sheet,
            ),
            para(
                "<b>Gate:</b> preregistered immutable manifest, disjoint calibration and "
                "evaluation incidents, at least five training seeds, and paired policy tests "
                "against doctrine, greedy, optimization, and no-action baselines.",
                sheet,
            ),
            PageBreak(),
            para("7. Conclusions", sheet, "h1"),
        ]
    )
    conclusion = Table(
        [
            [
                para(
                    "The most defensible current claim is narrow: the model has measurable "
                    "held-out active-front signal, and reconstructing arrival history improves "
                    "the next forecast. The most important negative result is equally clear: "
                    "the model still loses to persistence on cumulative perimeter and boundary "
                    "location.",
                    sheet,
                    "callout",
                )
            ]
        ],
        colWidths=[6.72 * inch],
    )
    conclusion.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
                ("BOX", (0, 0), (-1, -1), 0.7, GREEN),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend(
        [
            conclusion,
            Spacer(1, 10),
            para(
                "The next accuracy gain should come from better evidence, not a broader spread "
                "search: incident-scale wind, spatial fuel moisture, explicit observation "
                "uncertainty, and suppression-aware targets. The simulation should then use "
                "sequential coupled-state assimilation and a hierarchy of high-fidelity "
                "teachers. RL training should begin only after this uncertainty set is encoded "
                "as the environment distribution and the historical evaluation suite is frozen.",
                sheet,
            ),
            para("Artifacts", sheet, "h2"),
            bullet(
                "historical_validation_results.json - 168 full forecast records and provenance.",
                sheet,
                small=True,
            ),
            bullet(
                "historical_validation_examples.npz - observed, predicted, and ensemble raster examples.",
                sheet,
                small=True,
            ),
            bullet(
                "historical_accuracy_analysis.json - cluster intervals, ablations, incident summaries, and protocol checks.",
                sheet,
                small=True,
            ),
            bullet(
                "historical_accuracy_intervals.csv - flat, inspectable forecast metric table.",
                sheet,
                small=True,
            ),
            para("References", sheet, "h2"),
            para(
                "1. Magstadt et al. (2026). High spatial resolution daily fire perimeter progression dataset, western U.S., 2020-2024. doi:10.17632/95rj5d379g.1",
                sheet,
                "ref",
            ),
            para(
                "2. Kochanski et al. (2023). Analysis of methods for assimilating fire perimeters into a coupled fire-atmosphere model. doi:10.3389/ffgc.2023.1203578",
                sheet,
                "ref",
            ),
            para(
                "3. Mandel et al. (2011). Coupled atmosphere-wildland fire modeling with WRF-Fire version 3.3. arXiv:1208.1059.",
                sheet,
                "ref",
            ),
            para(
                "4. Liu et al. (2024). Systematically tracking hourly progression of large wildfires using GOES. Earth System Science Data 16, 1395-1414.",
                sheet,
                "ref",
            ),
            para("5. NASA POWER. Hourly API documentation. power.larc.nasa.gov.", sheet, "ref"),
            para("6. NOAA. High-Resolution Rapid Refresh. rapidrefresh.noaa.gov/hrrr.", sheet, "ref"),
            para(
                "7. LANDFIRE. Landscape fire and resource management planning tools. landfire.gov.",
                sheet,
                "ref",
            ),
            Spacer(1, 7),
            para(
                "Machine-readable analysis is authoritative for reported precision. "
                "Study code, configuration, and tests are retained with the repository.",
                sheet,
                "small",
            ),
        ]
    )
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet = styles()
    doc = BaseDocTemplate(
        str(args.out),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.63 * inch,
        bottomMargin=0.58 * inch,
        title="Aeolus-IA Historical Accuracy Study - Corrected V3",
        author="Aeolus-IA research program",
        subject="Held-out wildfire spread hindcasts and coupled-state initialization",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=page_chrome)])
    doc.build(build_story(analysis, args.figures, sheet))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
