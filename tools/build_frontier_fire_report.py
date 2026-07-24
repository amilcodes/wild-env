"""Build the fire-state and behavior research report from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#17242B")
MUTED = colors.HexColor("#52636D")
ORANGE = colors.HexColor("#D96324")
CYAN = colors.HexColor("#247D91")
PURPLE = colors.HexColor("#67578E")
PAPER = colors.HexColor("#F7F4EE")
LINE = colors.HexColor("#CAD2D6")


def _image(path: Path, width: float, height: float) -> Image:
    image = Image(str(path), width=width, height=height)
    image.hAlign = "LEFT"
    return image


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=28,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=CYAN,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.2,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=14,
            textColor=INK,
            borderColor=ORANGE,
            borderWidth=0,
            borderPadding=(8, 10, 8, 10),
            backColor=colors.HexColor("#F2E8DE"),
            spaceBefore=7,
            spaceAfter=10,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.5,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.5,
            textColor=colors.white,
        ),
    }


def _page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setStrokeColor(LINE)
    canvas.line(0.62 * inch, 0.48 * inch, width - 0.62 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.62 * inch, 0.29 * inch, "Fire-state and behavior research increment")
    canvas.drawRightString(
        width - 0.62 * inch,
        0.29 * inch,
        f"{doc.page}",
    )
    canvas.restoreState()


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
) -> Table:
    normalized = [
        [
            value
            if hasattr(value, "wrap")
            else Paragraph(
                str(value),
                styles["table_head" if row_index == 0 else "table"],
            )
            for value in row
        ]
        for row_index, row in enumerate(rows)
    ]
    table = Table(normalized, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _metric(
    results: dict[str, Any],
    method: str,
    key: str,
    *,
    active: bool = False,
) -> float:
    collection = results["active_growth_summaries"] if active else results["summaries"]
    return float(collection[method][key]["mean"])


def _formatted_metric(
    results: dict[str, Any],
    method: str,
    key: str,
    *,
    active: bool = False,
) -> str:
    return f"{_metric(results, method, key, active=active):.3f}"


def build_report(args: argparse.Namespace) -> Path:
    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    front = json.loads(args.front.read_text(encoding="utf-8"))
    behavior = json.loads(args.behavior.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    styles = _styles()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(args.out),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.62 * inch,
        title="Fire State and Behavior Research Increment",
        author="Aeolus-IA research",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="body",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates([PageTemplate(id="research", frames=[frame], onPage=_page)])
    story: list[Any] = []

    story.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Fire State and Behavior Research Increment", styles["title"]),
            Paragraph(
                f"Version 0.4 technical report | {date.today().isoformat()}",
                styles["subtitle"],
            ),
            Paragraph(
                "A signed level-set fireline, continuous probabilistic incident "
                "state, spatial forcing, posterior parameter ensembles, and a "
                "held-out historical evaluation.",
                styles["callout"],
            ),
            Paragraph("Research outcome", styles["h1"]),
            Paragraph(
                "The fireline now uses the numerical structure of current "
                "WRF-Fire/CFBM-class spread solvers: a signed level set, WENO5 "
                "spatial derivatives, SSP-RK3 integration and periodic "
                "reinitialization. NumPy and PyTorch paths implement the same "
                "front equation. This advances numerical fidelity and state "
                "quality; it does not add coupled atmosphere or plume physics.",
                styles["body"],
            ),
            Paragraph("Evidence at a glance", styles["h2"]),
        ]
    )
    weno_15 = next(
        item
        for item in front["radial_grid_refinement"]
        if item["solver"] == "weno5" and item["cell_size_m"] == 15.0
    )
    ensemble_iou = _metric(historical, "calibrated_ensemble", "metrics.iou")
    ensemble_growth = _metric(
        historical,
        "calibrated_ensemble",
        "growth_tolerance_1_cell.f1",
        active=True,
    )
    posterior_brier = float(
        historical["probabilistic_active_growth_summaries"][
            "active_domain_probabilistic_metrics.balanced_brier_score"
        ]["mean"]
    )
    probability_skill = float(
        historical["probabilistic_skill_against_persistence"]["active_domain_balanced_brier_skill"]
    )
    story.append(
        _table(
            [
                ["Dimension", "Measured result", "Interpretation"],
                [
                    "Front accuracy",
                    f"{abs(weno_15['equivalent_radius_error_m']):.2f} m error",
                    "15 m circular manufactured solution after 30 min",
                ],
                [
                    "Rotation",
                    (f"{100 * front['rotation_invariance']['area_coefficient_of_variation']:.2f}% area CV"),
                    "Eight rotated anisotropic headings",
                ],
                [
                    "Held-out extent",
                    f"{ensemble_iou:.3f} IoU",
                    "Posterior ensemble, 24 intervals / six incidents",
                ],
                [
                    "Held-out front",
                    f"{ensemble_growth:.3f} tolerance F1",
                    "Intervals with observed growth",
                ],
                [
                    "Probability",
                    (f"{posterior_brier:.3f} balanced Brier; {100 * probability_skill:+.1f}% skill"),
                    "Active-growth domain; persistence reference",
                ],
                [
                    "Local MPS throughput",
                    (f"{benchmark['million_cell_steps_s']:.2f} M cell-steps/s"),
                    (
                        f"{benchmark['batch']} x {benchmark['grid'][0]} x "
                        f"{benchmark['grid'][1]} batch, "
                        f"{benchmark['steps']} fire minutes"
                    ),
                ],
            ],
            [1.25 * inch, 1.55 * inch, 4.0 * inch],
            styles,
        )
    )
    persistence_iou = _metric(historical, "persistence", "metrics.iou")
    claim = (
        "The ensemble beats persistence on cumulative IoU in this study."
        if ensemble_iou > persistence_iou
        else (
            "Persistence remains stronger on cumulative IoU. The result is "
            "retained as a model-development constraint, not reframed as an "
            "accuracy claim."
        )
    )
    story.extend(
        [
            Spacer(1, 0.12 * inch),
            Paragraph(claim, styles["callout"]),
            Paragraph("Scope", styles["h2"]),
            Paragraph(
                "Use this simulator for MARL method development, paired policy "
                "experiments, numerical ablations and probabilistic stress "
                "testing. It is not an operational spread forecast or dispatch "
                "system.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("1. Model formulation and state", styles["h1"]),
            Paragraph(
                "<b>Front equation.</b> The zero contour of phi is propagated by "
                "phi_t + R(n)|grad(phi)| = 0. Directional rate R(n) is recovered "
                "from the local heading rate, ellipse eccentricity and front "
                "normal. A connected-support mask prevents disconnected contour "
                "nucleation across barriers.",
                styles["body"],
            ),
            Paragraph(
                "<b>Local behavior.</b> Surface behavior is interpolated from a "
                "packaged Pyretechnics-derived Anderson/Scott-Burgan table. Wind "
                "and slope combine as vectors. Crown transition uses Van Wagner "
                "initiation and Cruz potential spread. Spotting uses stochastic "
                "downwind and crosswind transport.",
                styles["body"],
            ),
            Paragraph(
                "<b>Forcing.</b> CF-NetCDF fields can be incident-wide time "
                "series or aligned (time, y, x) rasters. Wind direction is "
                "interpolated on the unit circle. Dead 1/10/100-hour fuels move "
                "toward equilibrium moisture with precipitation wetting.",
                styles["body"],
            ),
            Paragraph("State contract", styles["h2"]),
            _table(
                [
                    ["Plane", "Fields retained", "Use"],
                    [
                        "Truth",
                        "phase, fire type, phi, arrival time, intensity, ROS, flame, fuel, moisture",
                        "Propagation, critic input and deterministic replay",
                    ],
                    [
                        "Belief",
                        "burn probability, arrival mean/std, intensity mean/std, observation time",
                        "Actor tasks and delayed information boundary",
                    ],
                    [
                        "Ensemble",
                        "spread, wind exposure/direction, moisture bias, posterior weight",
                        "Burn probability and conditional arrival-time moments",
                    ],
                ],
                [0.9 * inch, 3.65 * inch, 2.25 * inch],
                styles,
            ),
            Paragraph("Execution paths", styles["h2"]),
            Paragraph(
                "The canonical NumPy simulator integrates GIS, resources, "
                "suppression and PettingZoo semantics. TensorFireKernel retains "
                "complete batches on CUDA, ROCm, MPS or CPU through behavior "
                "lookup, level-set propagation, reinitialization, crown and "
                "spotting. Historical members run through a persistent, "
                "spawn-safe process pool.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("2. Numerical verification", styles["h1"]),
            Paragraph(
                "Manufactured circular fronts test grid refinement against an "
                "analytic radius. An anisotropic case is rotated through eight "
                "headings to expose orientation bias. These tests isolate the "
                "front discretization from uncertainty in physical rate inputs.",
                styles["body"],
            ),
            _image(args.front_figure, 7.1 * inch, 4.82 * inch),
            Spacer(1, 0.08 * inch),
            _table(
                [
                    ["Solver / cell", "Equivalent radius error", "Maximum radius error"],
                    *[
                        [
                            f"{item['solver'].upper()} / {item['cell_size_m']:.0f} m",
                            f"{item['equivalent_radius_error_m']:+.2f} m",
                            f"{item['maximum_radius_error_m']:+.2f} m",
                        ]
                        for item in front["radial_grid_refinement"]
                    ],
                ],
                [2.2 * inch, 2.3 * inch, 2.3 * inch],
                styles,
            ),
            Paragraph(
                "Automated tests additionally require exact WENO derivatives "
                "for a linear field, NumPy/PyTorch one-step agreement, correct "
                "signed-distance sign and zero crossing beyond a spanning "
                "barrier.",
                styles["small"],
            ),
            PageBreak(),
            Paragraph("3. Integrated behavior cases", styles["h1"]),
            Paragraph(
                "The atlas exercises calm surface spread, wind-driven spread, "
                "wind plus slope, and active crown fire with spotting. It also "
                "plots the underlying reference-table response across fuel "
                "models. The cases are verification fixtures, not field "
                "validation.",
                styles["body"],
            ),
            _image(args.behavior_atlas, 7.2 * inch, 4.53 * inch),
            Spacer(1, 0.08 * inch),
            _table(
                [
                    ["Case", "Reached cells", "Peak ROS", "Peak intensity"],
                    *[
                        [
                            item["name"].replace("_", " "),
                            f"{item['reached_cells']:,}",
                            f"{item['max_spread_rate_m_min']:.1f} m/min",
                            f"{item['max_intensity_kw_m']:,.0f} kW/m",
                        ]
                        for item in behavior["cases"]
                    ],
                ],
                [2.45 * inch, 1.25 * inch, 1.35 * inch, 1.75 * inch],
                styles,
            ),
            PageBreak(),
            Paragraph("4. Held-out historical evaluation", styles["h1"]),
            Paragraph(
                "Six NIROPS incidents contribute one earlier calibration "
                "interval and four later forecast intervals each. Every "
                "forecast starts from its observed initial perimeter and uses "
                "hourly NASA POWER weather. The held-out target is not "
                "assimilated. Suppression is disabled because a matched "
                "time-stamped action history is unavailable.",
                styles["body"],
            ),
            _image(args.aggregate_figure, 7.1 * inch, 4.66 * inch),
            Spacer(1, 0.08 * inch),
            _table(
                [
                    ["Method", "Extent IoU", "Active growth F1", "Boundary m"],
                    *[
                        [
                            label,
                            f"{_metric(historical, method, 'metrics.iou'):.3f}",
                            _formatted_metric(
                                historical,
                                method,
                                "growth_tolerance_1_cell.f1",
                                active=True,
                            ),
                            (f"{_metric(historical, method, 'boundary.mean_symmetric_distance_m'):.0f}"),
                        ]
                        for method, label in (
                            ("persistence", "Persistence"),
                            ("raw_physics", "Raw physics"),
                            ("calibrated_physics", "Scalar calibrated"),
                            ("calibrated_ensemble", "Posterior ensemble"),
                        )
                    ],
                ],
                [2.4 * inch, 1.4 * inch, 1.65 * inch, 1.35 * inch],
                styles,
            ),
            PageBreak(),
            Paragraph("5. Posterior fire-reach state", styles["h1"]),
            Paragraph(
                "Each map is the posterior-weighted fraction of parameter "
                "particles reaching a cell. The earlier calibration updates "
                "joint particles over spread, wind exposure, wind direction and "
                "dead-fuel moisture. Final-perimeter distance, growth area and "
                "one-cell growth localization define the update. Cyan outlines "
                "are held-out perimeters.",
                styles["body"],
            ),
            _image(args.probability_atlas, 7.15 * inch, 4.96 * inch),
            Paragraph(
                "Whole-raster Brier scores are reported for comparability. A "
                "forecast-independent, observation-defined buffered "
                "active-domain balanced Brier "
                "score is also required "
                "because the whole raster is dominated by easy unburned "
                "background. Reliability bins, entropy, effective sample size "
                "and member area ranges are retained in JSON.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("6. Validity boundary and research gates", styles["h1"]),
            Paragraph("What changed", styles["h2"]),
            Paragraph(
                "High-order continuous front geometry, sub-minute arrival time, "
                "spatial weather, probabilistic belief, posterior parameter "
                "ensembles, proper scores, process-parallel historical "
                "orchestration and replay of the new state are implemented and "
                "tested.",
                styles["body"],
            ),
            Paragraph("What remains unresolved", styles["h2"]),
            _table(
                [
                    ["Gap", "North-star capability", "Next falsifiable test"],
                    [
                        "Fire-weather feedback",
                        "WRF-Fire / UFS-CFBM two-way coupling",
                        "Distill correction fields from matched coupled runs",
                    ],
                    [
                        "3-D fuel / flow",
                        "QUIC-Fire or FIRETEC",
                        "RxCADRE/Spring Hill cross-model behavior metrics",
                    ],
                    [
                        "Initialization",
                        "Arrival-history coupled spin-up",
                        "Two-perimeter arrival reconstruction and free forecast",
                    ],
                    [
                        "Fuel moisture",
                        "Incident RAWS / gridded conditioning",
                        "Held-out ablation against coarse POWER forcing",
                    ],
                    [
                        "Spotting",
                        "Observed ember / spot-fire distributions",
                        "Distance, direction and ignition-delay calibration",
                    ],
                    [
                        "Suppression",
                        "Time-stamped drop and fireline actions",
                        "Independent treatment-response validation",
                    ],
                ],
                [1.45 * inch, 2.45 * inch, 2.9 * inch],
                styles,
            ),
            Paragraph("Release gate", styles["h2"]),
            Paragraph(
                "A future fire-model release should beat persistence on "
                "held-out advancing-front localization, improve calibrated "
                "probability rather than only a 0.5 contour, and preserve the "
                "numerical verification thresholds. MARL policies should train "
                "against the posterior uncertainty set and be evaluated with "
                "paired seeds.",
                styles["callout"],
            ),
            Paragraph("References", styles["h2"]),
            Paragraph(
                "WRF-Fire user guide: "
                "https://www2.mmm.ucar.edu/wrf/site/documentation/users_guide/fire.html<br/>"
                "Jimenez y Munoz et al. 2026, CFBM: "
                "https://doi.org/10.5194/gmd-19-3035-2026<br/>"
                "Kochanski et al. 2023, perimeter assimilation: "
                "https://doi.org/10.3389/ffgc.2023.1203578<br/>"
                "Linn et al. 2020, QUIC-Fire: "
                "https://doi.org/10.1016/j.envsoft.2019.104616<br/>"
                "Magstadt et al. 2026, NIROPS: "
                "https://doi.org/10.17632/95rj5d379g.1",
                styles["small"],
            ),
        ]
    )
    document.build(story)
    return args.out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--front-figure", required=True, type=Path)
    parser.add_argument("--behavior-atlas", required=True, type=Path)
    parser.add_argument("--aggregate-figure", required=True, type=Path)
    parser.add_argument("--probability-atlas", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(build_report(args).resolve())


if __name__ == "__main__":
    main()
