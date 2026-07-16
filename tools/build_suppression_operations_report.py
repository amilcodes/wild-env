"""Build the v0.5 suppression and coupled-state technical report."""

from __future__ import annotations

import argparse
import json
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
MUTED = colors.HexColor("#53636C")
GREEN = colors.HexColor("#3F8C62")
RED = colors.HexColor("#B5473E")
CYAN = colors.HexColor("#247D91")
MAGENTA = colors.HexColor("#A83C71")
PAPER = colors.HexColor("#F7F4EE")
LINE = colors.HexColor("#CAD2D6")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=27,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=INK,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.8,
            leading=13,
            textColor=CYAN,
            spaceBefore=5,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.0,
            leading=12.4,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.6,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13.6,
            textColor=INK,
            backColor=colors.HexColor("#E7EFE9"),
            borderColor=GREEN,
            borderWidth=0,
            borderPadding=(8, 10, 8, 10),
            spaceBefore=4,
            spaceAfter=9,
        ),
        "warning": ParagraphStyle(
            "warning",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=12.4,
            textColor=INK,
            backColor=colors.HexColor("#F3E6E2"),
            borderPadding=(7, 9, 7, 9),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.1,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.1,
            textColor=colors.white,
        ),
    }


def _page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setStrokeColor(LINE)
    canvas.line(0.58 * inch, 0.45 * inch, width - 0.58 * inch, 0.45 * inch)
    canvas.setFont("Helvetica", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        0.58 * inch,
        0.27 * inch,
        "Suppression, operations, and coupled-state initialization | v0.5",
    )
    canvas.drawRightString(width - 0.58 * inch, 0.27 * inch, str(doc.page))
    canvas.restoreState()


def _table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    *,
    header_color: colors.Color = INK,
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
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _image(path: Path, width: float, height: float) -> Image:
    value = Image(str(path), width=width, height=height)
    value.hAlign = "LEFT"
    return value


def _href(url: str, label: str) -> str:
    return f'<link href="{url}" color="#247D91">{label}</link>'


def build(args: argparse.Namespace) -> Path:
    result = json.loads(args.results.read_text(encoding="utf-8"))
    arrival = result["arrival_history"]["paired_summary"]
    styles = _styles()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(args.out),
        pagesize=letter,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.54 * inch,
        bottomMargin=0.60 * inch,
        title="Suppression, Operations, and Coupled-State Initialization",
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
    document.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=_page)])
    story: list[Any] = []

    # Page 1
    story.extend(
        [
            Spacer(1, 0.08 * inch),
            Paragraph(
                "Suppression, Operations, and Coupled-State Initialization",
                styles["title"],
            ),
            Paragraph(
                "Version 0.5 technical research report | 2026-07-28",
                styles["subtitle"],
            ),
            Paragraph(
                "Explicit air/ground suppression mechanics, two-perimeter fire "
                "arrival reconstruction, incident wind/fuel-moisture analysis, "
                "and advancing-front localization.",
                styles["callout"],
            ),
            Paragraph("Research outcome", styles["h1"]),
            Paragraph(
                "The simulator now conserves liquid payload volume, constructs "
                "crew/dozer line over time, records held and breached "
                "engagement, queues shared reload capacity, masks unsafe "
                "missions, and initializes a free forecast from two observed "
                "perimeters with burn age, residual heat, and recent front "
                "velocity. Station innovations can condition gridded wind and "
                "dead/live fuel moisture.",
                styles["body"],
            ),
            _table(
                [
                    ["Evidence set", "Design", "Principal result"],
                    [
                        "Historical state",
                        "6 NIROPS fires; 24 held-out transitions",
                        "IoU 0.807 -> 0.832; mean boundary 299 -> 224 m",
                    ],
                    [
                        "Suppression",
                        "8 seeds x 3 winds x 3 strategies; 72 trials",
                        "Integrated loss -28.0%; burned fraction -43.1%",
                    ],
                    [
                        "Verification",
                        "Unit, integration, API, numerics and replay",
                        "51 tests and static checks pass",
                    ],
                ],
                [1.15 * inch, 2.25 * inch, 3.45 * inch],
                styles,
                header_color=GREEN,
            ),
            Spacer(1, 0.12 * inch),
            _image(args.operations_frame, 6.85 * inch, 4.00 * inch),
            Paragraph(
                "Figure 1. Deterministic replay at T+120 min. Raw treatment, "
                "line engagement, resource mission state, fire truth, delayed "
                "belief edge, terrain and protected assets are rendered from "
                "the stored episode.",
                styles["small"],
            ),
        ]
    )

    # Page 2
    story.extend(
        [
            PageBreak(),
            Paragraph("1. Reference problem and model boundary", styles["h1"]),
            Paragraph(
                "Wildfire suppression is a joint fire-behavior, logistics, "
                "safety and incomplete-information problem. A fast RL "
                "environment cannot inherit operational validity from a "
                "visually plausible fire front. This increment therefore "
                "defines measurable state variables and keeps each "
                "unvalidated response coefficient explicit.",
                styles["body"],
            ),
            _table(
                [
                    ["Reference class", "What is adopted", "What remains outside"],
                    [
                        "USFS containment/production",
                        "anchor/flank/indirect task structure; variable line rate; held/burned-over outcome",
                        "incident command organization, full safety/duty doctrine",
                    ],
                    [
                        "Aerial retardant guidance",
                        "GPC, fuel/intensity coverage requirement, wind-shaped footprint",
                        "field-calibrated coverage/effectiveness and drop telemetry",
                    ],
                    [
                        "WRF-SFIRE replay",
                        "two perimeter arrival history; burn age; heat-flux memory; free forecast",
                        "resolved atmosphere and plume feedback",
                    ],
                    [
                        "RAWS/NFDRS/FlamMap",
                        "time-lag fuel moisture and station-conditioned spatial forcing",
                        "archived incident RAWS ingest in the frozen NIROPS bundles",
                    ],
                    [
                        "Level-set DA",
                        "signed-distance innovation and advancing-front localization",
                        "full ensemble atmosphere/fire state covariance",
                    ],
                ],
                [1.25 * inch, 2.75 * inch, 2.85 * inch],
                styles,
            ),
            Paragraph("System state added in v0.5", styles["h2"]),
            _table(
                [
                    ["Plane", "State"],
                    [
                        "Suppression truth",
                        "water/retardant GPC and efficacy; line geometry, strength, status",
                    ],
                    [
                        "Coupled fire state",
                        "arrival time, burn age, fuel remaining, heat memory, localized speed/head",
                    ],
                    [
                        "Operations",
                        "working and queued status; line progress; task orientation; reload bays",
                    ],
                    [
                        "Incident forcing",
                        "dead/live moisture and u/v correction fields with uncertainty",
                    ],
                    [
                        "Evidence",
                        "engagement, queue, productivity, drop-volume and correction diagnostics",
                    ],
                ],
                [1.55 * inch, 5.30 * inch],
                styles,
                header_color=CYAN,
            ),
            Paragraph(
                "Actors continue to receive delivered belief and observable "
                "fleet/task state. Hidden fire truth, line outcome before "
                "engagement, and undelivered measurements remain outside the "
                "actor observation.",
                styles["warning"],
            ),
            Paragraph("Primary implementation units", styles["h2"]),
            Paragraph(
                "<b>core/suppression.py</b>: drop, line and engagement physics. "
                "<b>core/initialization.py</b>: causal arrival reconstruction. "
                "<b>core/localization.py</b>: signed-distance correction. "
                "<b>data/forcing.py</b>: station/background optimum "
                "interpolation. <b>core/simulator.py</b>: mission/logistics "
                "integration.",
                styles["body"],
            ),
        ]
    )

    # Page 3
    story.extend(
        [
            PageBreak(),
            Paragraph("2. Suppression mechanics", styles["h1"]),
            Paragraph("Volume-conserving drops", styles["h2"]),
            Paragraph(
                "A payload V is distributed by a normalized finite Gaussian "
                "ground pattern p. Cell coverage in GPC is "
                "<b>C = V p / A_cell / 0.407436</b>. Wind displaces the "
                "centroid and broadens the pattern without changing volume. "
                "The required coverage class follows the USFS fuel guide with "
                "an explicit fireline-intensity increment. Effective treatment "
                "is <b>E = 1 - exp(-C/C_required)</b>.",
                styles["body"],
            ),
            _table(
                [
                    ["State/process", "Default", "Testable consequence"],
                    ["Water half-life", "8 min", "short-duration intensity/moisture conditioning"],
                    ["Retardant half-life", "720 min", "persistent spread conditioning"],
                    ["Rain wash", "0.08 fraction/mm", "time-integrated precipitation loss"],
                    ["Coverage conversion", "0.407436 L/m2/GPC", "payload reconstructed to rtol 1e-5"],
                    ["Footprint", "finite oriented Gaussian", "drift/dispersion without volume gain"],
                ],
                [1.75 * inch, 1.55 * inch, 3.55 * inch],
                styles,
                header_color=MAGENTA,
            ),
            Paragraph("Constructed line", styles["h2"]),
            Paragraph(
                "A crew/dozer mission accrues length each minute using a "
                "mean-one lognormal production multiplier (CV 0.38). Every "
                "line cell retains physical width/strength. When adjacent fire "
                "arrives, a logistic comparison of intensity demand with base, "
                "width and retardant capacity produces held or breached state. "
                "Held cells stop local normal advance; line ends and breached "
                "cells remain pathways.",
                styles["body"],
            ),
            _image(args.suppression_figure, 6.85 * inch, 4.24 * inch),
            Paragraph(
                "Figure 2. Matched-seed mechanism experiment. Integrated "
                "operations have nonzero held and breached line, reload queues "
                "and finite resource workload.",
                styles["small"],
            ),
        ]
    )

    # Page 4
    story.extend(
        [
            PageBreak(),
            Paragraph("3. Mission and logistics model", styles["h1"]),
            Paragraph(
                "<b>AVAILABLE -> OUTBOUND -> WORKING/DROP/OBSERVE -> "
                "RETURNING -> QUEUED -> RELOADING -> AVAILABLE</b>. "
                "WITHDRAWN is terminal within an episode.",
                styles["callout"],
            ),
            _table(
                [
                    ["Mechanism", "Implementation"],
                    ["Travel", "dispatch latency plus continuous cruise interpolation"],
                    ["Ground work", "oriented line segment increment each minute"],
                    ["Drop", "payload-specific water or retardant ground pattern"],
                    ["Return/service", "travel to base, shared reload-bay queue, reload"],
                    ["Endurance", "minute duty/flight accumulation and withdrawal"],
                    ["Safety gate", "aviation wind and crew direct-intensity action mask"],
                    ["Audit", "attempt, acceptance, block reason, event, cost, exposure"],
                ],
                [1.45 * inch, 5.40 * inch],
                styles,
            ),
            Paragraph("Doctrine comparator", styles["h2"]),
            Paragraph(
                "Observe, water, retardant, reinforce and line tasks are "
                "generated from the delivered belief. Lines are placed ahead "
                "of a threatened front, normal to the approach toward the "
                "value-weighted asset centroid; otherwise the wind-driven head "
                "is used. Retardant reinforcement shares line geometry. "
                "Mission orientation is fixed at assignment. The exact joint "
                "assignment comparator honors the same compatibility and "
                "operational mask as a learned policy.",
                styles["body"],
            ),
            _image(args.operations_3d, 6.85 * inch, 4.13 * inch),
            Paragraph(
                "Figure 3. Terrain-aware replay generated from the same stored "
                "state as Figure 1. Rendering remains downstream of simulation.",
                styles["small"],
            ),
        ]
    )

    # Page 5
    story.extend(
        [
            PageBreak(),
            Paragraph("4. Two-perimeter coupled-state initialization", styles["h1"]),
            Paragraph(
                "A single perimeter supplies geometry but no fire history. "
                "Following the WRF-SFIRE replay problem, an earlier and a "
                "forecast-start perimeter constrain a causal arrival-time "
                "field. The implementation solves <b>Laplacian(T)=0</b> in the "
                "growth band with T=-Delta and T=0 Dirichlet fronts. It differs "
                "from the cited biharmonic spline while reconstructing the same "
                "state variables with explicit non-nesting diagnostics.",
                styles["body"],
            ),
            _table(
                [
                    ["Derived state", "Definition/use"],
                    ["Arrival/burn age", "T and max(-T,0); causal replay clock"],
                    ["Fuel/phase", "age-conditioned fuel; active recent band; burned interior"],
                    ["Heat memory", "age-decayed flux state retained for coupled correction"],
                    ["Recent velocity", "1/|grad T| and outward normalized grad T"],
                    ["Localization", "8-cell Gaussian front band; 180 min temporal half-life"],
                    ["Regularization", "rate ratio in [0.55,3.0]; direction blend <=0.45"],
                ],
                [1.55 * inch, 5.30 * inch],
                styles,
                header_color=CYAN,
            ),
            Spacer(1, 0.05 * inch),
            _image(args.arrival_figure, 6.70 * inch, 6.43 * inch),
        ]
    )

    # Page 6
    story.extend(
        [
            PageBreak(),
            Paragraph("5. Held-out historical result", styles["h1"]),
            Paragraph(
                "Six NIROPS fires contribute 24 held-out transitions. Every "
                "history forecast uses only the immediately preceding perimeter "
                "and its own start perimeter. The v0.4 per-incident scalar "
                "spread calibration and forecast windows are reused. Confidence "
                "intervals resample incidents, preserving within-fire "
                "dependence.",
                styles["body"],
            ),
            _table(
                [
                    ["Metric", "Single perimeter", "Two perimeter", "Paired delta (95% CI)"],
                    [
                        "Perimeter IoU",
                        f"{arrival['perimeter_iou']['baseline_mean']:.3f}",
                        f"{arrival['perimeter_iou']['history_mean']:.3f}",
                        "+0.024 (+0.004, +0.051)",
                    ],
                    [
                        "Mean boundary distance",
                        f"{arrival['mean_boundary_distance_m']['baseline_mean']:.0f} m",
                        f"{arrival['mean_boundary_distance_m']['history_mean']:.0f} m",
                        "-75 m (-201, -4)",
                    ],
                    [
                        "Hausdorff 95",
                        f"{arrival['hausdorff_95_m']['baseline_mean']:.0f} m",
                        f"{arrival['hausdorff_95_m']['history_mean']:.0f} m",
                        "-283 m (-730, -5)",
                    ],
                    [
                        "Exact growth IoU",
                        f"{arrival['growth_iou']['baseline_mean']:.3f}",
                        f"{arrival['growth_iou']['history_mean']:.3f}",
                        "-0.011 (-0.018, -0.005)",
                    ],
                    [
                        "One-cell growth F1",
                        f"{arrival['growth_tolerance_f1']['baseline_mean']:.3f}",
                        f"{arrival['growth_tolerance_f1']['history_mean']:.3f}",
                        "+0.014 (-0.044, +0.112)",
                    ],
                ],
                [1.65 * inch, 1.30 * inch, 1.30 * inch, 2.60 * inch],
                styles,
                header_color=GREEN,
            ),
            Paragraph("Interpretation", styles["h2"]),
            Paragraph(
                "The coupled-state initializer improves cumulative location and "
                "boundary displacement. It is conservative on exact new-growth "
                "overlap. The tolerance-based growth result is positive but "
                "uncertain. This is evidence for state reconstruction and "
                "evidence that incident wind/moisture and posterior correction "
                "are still required; it is not a general spread-accuracy claim.",
                styles["warning"],
            ),
            Paragraph("6. Matched suppression result", styles["h1"]),
            _table(
                [
                    ["Strategy", "Loss", "Burned", "Escape", "Relative loss vs no action"],
                    [
                        "Uncontrolled",
                        "196.1",
                        "8.30%",
                        "66.7%",
                        "-",
                    ],
                    [
                        "Aerial only",
                        "166.5",
                        "6.46%",
                        "45.8%",
                        "-15.1%",
                    ],
                    [
                        "Integrated",
                        "141.2",
                        "4.72%",
                        "33.3%",
                        "-28.0%",
                    ],
                ],
                [1.30 * inch, 1.00 * inch, 1.10 * inch, 1.10 * inch, 2.35 * inch],
                styles,
                header_color=GREEN,
            ),
            Paragraph(
                "Integrated minus uncontrolled loss is -54.8 (seed-cluster 95% "
                "CI -74.7 to -36.7); burned fraction changes -43.1%. These are "
                "controlled simulator mechanism effects. No historical drop or "
                "crew log is used to estimate field effectiveness.",
                styles["body"],
            ),
            Paragraph(
                "Mean integrated workload: 7.3 water drops, 8.0 retardant "
                "drops, 2.9 completed lines, 6.6 held line cells, 9.3 breached "
                "line cells and 12.5 reload queue entries per 180 min trial.",
                styles["callout"],
            ),
        ]
    )

    # Page 7
    story.extend(
        [
            PageBreak(),
            Paragraph("7. Incident forcing and correction fields", styles["h1"]),
            Paragraph(
                "The CF-NetCDF contract now carries spatial dead/live fuel "
                "moisture and coupled-model u/v wind corrections. "
                "analyze_incident_forcing applies Gaussian optimum interpolation "
                "<b>x_a = x_b + B H^T(HBH^T+R)^-1(y-Hx_b)</b>. Wind is "
                "analyzed in Cartesian components; temperature, humidity and "
                "moisture use the same spatial covariance. Posterior normalized "
                "standard deviation is returned for ensembles.",
                styles["body"],
            ),
            _image(args.forcing_figure, 6.85 * inch, 2.23 * inch),
            Paragraph(
                "Figure 5. Synthetic contract test with two station "
                "innovations. The frozen NIROPS study still uses coarse NASA "
                "POWER forcing and is not relabeled incident-grade.",
                styles["small"],
            ),
            Paragraph("Advancing-front correction", styles["h2"]),
            Paragraph(
                "Sequential perimeter assimilation works in signed-distance "
                "coordinates: <b>delta_phi = gain L clip(phi_obs-phi_f)</b>. "
                "The Gaussian localization L is zero beyond three radii. Only "
                "cells whose corrected level-set sign changes are advanced or "
                "retracted; distant fuels, burned interior and weather are not "
                "rewritten.",
                styles["body"],
            ),
            _table(
                [
                    ["Input", "Accepted representation", "Failure check"],
                    ["Weather background", "(time,) or aligned (time,y,x)", "shape/time/finite/range"],
                    ["Station data", "projected x/y, time, optional variables", "explicit time window"],
                    ["Wind correction", "u/v m/s fields", "component addition before direction"],
                    ["Fuel moisture", "dead 1/10/100 h; live herb/woody", "kg/kg range [0,3]"],
                    ["Perimeter correction", "2-D cumulative mask", "common grid and finite radius"],
                ],
                [1.35 * inch, 3.05 * inch, 2.45 * inch],
                styles,
            ),
            Paragraph(
                "Coupled correction state is represented and replayed by the "
                "fast kernel. A resolved atmospheric response still requires "
                "WRF-SFIRE, QUIC-Fire or another coupled model.",
                styles["warning"],
            ),
        ]
    )

    # Page 8
    story.extend(
        [
            PageBreak(),
            Paragraph("8. Verification and validity envelope", styles["h1"]),
            _table(
                [
                    ["Gate", "Frozen outcome"],
                    ["Automated tests", "51 passed"],
                    ["Static analysis", "ruff check src tests tools passed"],
                    ["Arrival analytic", "radial speed/direction, causality, heat and localization"],
                    ["Suppression analytic", "payload conservation, half-life and engagement state"],
                    ["Operations integration", "multi-minute crew line and masked assignment"],
                    ["Forcing analytic", "station localization and posterior uncertainty"],
                    ["Replay", "241 frames; Zarr/Parquet/JSON; 2-D, 3-D and MP4"],
                    ["Evidence artifacts", "JSON, CSV, NPZ, figures, report and source"],
                ],
                [1.55 * inch, 5.30 * inch],
                styles,
                header_color=GREEN,
            ),
            Paragraph("What remains unvalidated", styles["h2"]),
            Paragraph(
                "Coverage-response and line-capacity coefficients are mechanism "
                "priors. Historical bundles lack complete drops, line, crew, "
                "objective and decision logs. Line width below raster "
                "resolution is subgrid strength. The model does not contain "
                "safety zones/escape routes, full shift/fatigue rules, aircraft "
                "deconfliction, maintenance, dispatch governance, plume flow or "
                "two-way atmosphere feedback.",
                styles["warning"],
            ),
            Paragraph("Reproduction", styles["h2"]),
            Paragraph(
                "<b>Configuration:</b> configs/frontier_suppression.yaml<br/>"
                "<b>Study:</b> tools/run_suppression_operations_study.py<br/>"
                "<b>Figures:</b> tools/build_frontier_operations_figures.py<br/>"
                "<b>Frozen results:</b> results/frontier_operations_final<br/>"
                "<b>Methods:</b> docs/suppression_operations_research.md",
                styles["body"],
            ),
            Paragraph("Primary references", styles["h2"]),
            Paragraph(
                _href(
                    "https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2023.1203578/full",
                    "Kochanski et al.: WRF-SFIRE perimeter replay",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/download/treesearch/69196.pdf",
                    "USFS generalized wildfire containment algorithm",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/treesearch/44803",
                    "USFS operational fireline production estimates",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/treesearch/47358",
                    "USFS stochastic line production",
                )
                + "<br/>"
                + _href(
                    "https://www.fs.usda.gov/t-d/programs/wfcs/pubs/htmlpubs/htm01572808/index.htm",
                    "USFS aerial retardant coverage guidance",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/treesearch/80803",
                    "USFS retardant effectiveness research",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/rmrs/products/dataandtools/fireline-effectiveness-fle-dashboard",
                    "USFS Fireline Effectiveness dashboard",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/firelab/projects/firedangerrating",
                    "NFDRS/RAWS fuel-moisture system",
                )
                + "<br/>"
                + _href(
                    "https://research.fs.usda.gov/firelab/projects/flammap",
                    "FlamMap fuel conditioning and WindNinja workflow",
                )
                + "<br/>"
                + _href(
                    "https://publications.iafss.org/publications/fss/11/1443",
                    "Level-set ensemble Kalman fire-front assimilation",
                )
                + "<br/>"
                + _href(
                    "https://arxiv.org/abs/1203.2230",
                    "WRF-Fire artificial fire history and moisture estimation",
                ),
                styles["small"],
            ),
            Spacer(1, 0.10 * inch),
            Paragraph(
                "Conclusion. The project now has a coherent operational state "
                "model and a coupled-state initialization path that can be "
                "tested against real perimeter sequences. The next empirical "
                "priority is independent calibration with incident weather and "
                "time-resolved suppression records, followed by coupled-model "
                "replay rather than additional tuning against these six fires.",
                styles["callout"],
            ),
        ]
    )

    document.build(story)
    return args.out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--arrival-figure", type=Path, required=True)
    parser.add_argument("--suppression-figure", type=Path, required=True)
    parser.add_argument("--forcing-figure", type=Path, required=True)
    parser.add_argument("--operations-frame", type=Path, required=True)
    parser.add_argument("--operations-3d", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = build(args)
    print(path.resolve())


if __name__ == "__main__":
    main()
