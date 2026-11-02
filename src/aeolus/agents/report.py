"""Self-contained HTML audit report; no CDN, network, or generated model scores."""

from __future__ import annotations

import html
import json
from pathlib import Path


def render_report(run_dir: Path, destination: Path | None = None) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    traces = []
    for record in summary["records"]:
        entries = [json.loads(line) for line in (run_dir / record["trace"]).read_text().splitlines()]
        traces.append(
            {"record": record, "decisions": [entry for entry in entries if entry["type"] == "decision"]}
        )
    data = json.dumps({"summary": summary, "manifest": manifest, "traces": traces}, allow_nan=False)
    # JSON inside a script element needs HTML escaping even though it is not executable JavaScript.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    rows = "".join(
        "<tr><td>"
        + html.escape(row["policy"])
        + "<small>"
        + html.escape(row["provider"])
        + "</small></td>"
        + f"<td>{row['mean_loss_growth']:.2f}</td><td>{row['mean_burned_area_ha']:.1f}</td>"
        + (
            f"<td>{row['mean_loss_reduction_pct']:+.1f}%</td>"
            if row["mean_loss_reduction_pct"] is not None
            else "<td>n/a</td>"
        )
        + f"<td>{row['failed_decisions']} / {row['decisions']}</td>"
        + f"<td>{row['input_tokens_reported'] + row['output_tokens_reported']:,}"
        + ("*" if row["calls_with_unknown_usage"] else "")
        + "</td>"
        + ("<td class='good'>Eligible</td>" if row["eligible"] else "<td class='bad'>Failed decisions</td>")
        + "</tr>"
        for row in summary["leaderboard"]
    )
    comparisons = "".join(
        f"<li><b>{html.escape(item['candidate_policy'])}</b>: {item['mean_improvement']:+.2f} loss units; "
        f"descriptive 95% interval [{item['ci95_improvement_low']:+.2f}, "
        f"{item['ci95_improvement_high']:+.2f}] "
        f"across {item['case_clusters']} scenario clusters / {item['paired_records']} pairs."
        + (" Ineligible: includes fallback decisions." if not item["eligible"] else "")
        + "</li>"
        for item in summary["comparisons"]
    )
    page = _TEMPLATE.replace("__TITLE__", html.escape(summary["suite_id"]))
    page = page.replace("__STATE__", "Complete" if summary["complete"] else "PARTIAL RUN")
    page = page.replace("__SPLIT__", html.escape(summary["split"]))
    page = page.replace("__COUNT__", str(len(summary["records"])))
    page = page.replace("__BASELINE__", html.escape(summary["baseline"]))
    page = page.replace("__ROWS__", rows).replace("__COMPARISONS__", comparisons)
    page = page.replace("__HASH__", html.escape(summary["suite_sha256"]))
    page = page.replace("__DATA__", data)
    destination = destination or run_dir / "report.html"
    destination.write_text(page)
    return destination


_TEMPLATE = Path(__file__).with_name("report.html").read_text()
