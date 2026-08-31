"""Synchronize local EDA provenance, plots, HTML index, and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def svg_lines(
    path: Path,
    title: str,
    series: dict[str, list[tuple[str, float]]],
) -> None:
    labels = list(dict.fromkeys(label for points in series.values() for label, _ in points))
    values = [value for points in series.values() for _, value in points]
    low, high = min(values), max(values)
    pad = max((high - low) * 0.15, 0.001)
    low, high = low - pad, high + pad
    width, height = 900, 430
    left, right, top, bottom = 80, 30, 55, 95
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]

    def xy(index: int, value: float) -> tuple[float, float]:
        x = left + index * (width - left - right) / max(1, len(labels) - 1)
        y = top + (high - value) * (height - top - bottom) / (high - low)
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
    ]
    for index, label in enumerate(labels):
        x, _ = xy(index, low)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 55}" text-anchor="middle" font-size="11">{html.escape(label)}</text>'
        )
    for series_index, (name, points) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        coords = []
        point_map = dict(points)
        for index, label in enumerate(labels):
            if label not in point_map:
                continue
            x, y = xy(index, point_map[label])
            coords.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(
            f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" '
            f'stroke-width="2"/><text x="{left + series_index * 210}" y="{height - 18}" '
            f'font-size="12" fill="{color}">{html.escape(name)}</text>'
        )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, title: str, table: pd.DataFrame) -> None:
    values = table.to_numpy(float)
    maximum = max(float(np.nanmax(values)), 1e-12)
    rows, columns = table.shape
    cell_w, cell_h = 145, 42
    left, top = 210, 65
    width, height = left + columns * cell_w + 20, top + rows * cell_h + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
    ]
    for column, label in enumerate(table.columns):
        parts.append(
            f'<text x="{left + column * cell_w + cell_w / 2}" y="52" '
            f'text-anchor="middle" font-size="11">{html.escape(str(label))}</text>'
        )
    for row, label in enumerate(table.index):
        parts.append(
            f'<text x="{left - 8}" y="{top + row * cell_h + 26}" '
            f'text-anchor="end" font-size="11">{html.escape(str(label))}</text>'
        )
        for column in range(columns):
            value = values[row, column]
            intensity = 0 if not np.isfinite(value) else min(abs(value) / maximum, 1)
            blue = int(245 - 145 * intensity)
            fill = f"rgb({blue},{blue + 5},245)"
            x, y = left + column * cell_w, top + row * cell_h
            text = "NA" if not np.isfinite(value) else f"{value:.4f}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" stroke="white"/><text x="{x + cell_w / 2}" '
                f'y="{y + 26}" text-anchor="middle" font-size="11">{text}</text>'
            )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def build_plots(output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    field = pd.read_csv(output / "tables/field_size_base_rates.csv")
    field = field.loc[field["starter_count"].between(6, 18)]
    series = {
        str(period): [(str(int(row.starter_count)), float(row.empirical_win_rate)) for row in part.itertuples()]
        for period, part in field.groupby("analysis_period", observed=True)
    }
    svg_lines(plots / "field_size_win_curve.svg", "Win base rate by field size", series)

    missing = pd.read_csv(output / "tables/content_missingness.csv").set_index("analysis_period")
    missing = missing.drop(columns=["runners", "races"])
    svg_heatmap(plots / "content_missingness_matrix.svg", "Content missingness by period", missing.T)

    transition = pd.read_csv(output / "workstreams/g_context/transition_summary.csv")
    transition = transition.loc[transition["transition"].eq("surface_transition")]
    matrix = transition.pivot(index="cell", columns="period", values="persistence_race_spearman")
    svg_heatmap(plots / "surface_transition_matrix.svg", "Surface transition persistence", matrix)

    interaction = pd.read_csv(output / "workstreams/g_context/bounded_interactions.csv")
    interaction = interaction.loc[
        interaction["interaction"].eq("rest_x_workload30")
        & interaction["cell"].isin(["14-29|starts30=1", "14-29|starts30=2+"])
    ]
    series = {
        str(cell): [(str(row.period), float(row.centered_win)) for row in part.itertuples()]
        for cell, part in interaction.groupby("cell", observed=True)
    }
    svg_lines(plots / "rest_workload_interaction.svg", "Race-macro rest/workload interaction", series)

    pairs = pd.read_csv(output / "workstreams/f_connections/connection_redundancy_pairs.csv").head(12)
    pair_table = pairs.assign(
        pair=pairs["feature_a"].str.replace("_", " ") + " / " + pairs["feature_b"].str.replace("_", " ")
    ).set_index("pair")[["absolute_spearman"]]
    svg_heatmap(plots / "connection_redundancy_top_pairs.svg", "Top connection feature dependencies", pair_table)


def build_report(output: Path) -> None:
    plot_links = "".join(
        f'<li><a href="{path.relative_to(output)}">{html.escape(path.stem)}</a></li>'
        for path in sorted(output.rglob("*.svg"))
    )
    table_links = "".join(
        f'<li><a href="{path.relative_to(output)}">{html.escape(path.stem)}</a></li>'
        for path in sorted(output.rglob("*.csv"))
    )
    chapter_links = "".join(
        f'<li><a href="../../docs/eda/{index:02d}_{name}.md">chapter {index:02d}</a></li>'
        for index, name in [
            (3, "data_quality_and_drift"),
            (4, "target_and_race_structure"),
            (5, "horse_history_and_temporal_dynamics"),
            (6, "past_race_performance_content"),
            (7, "opponent_and_field_structure"),
            (8, "connections_and_entity_stability"),
            (9, "context_suitability_and_interactions"),
            (10, "model_errors_and_market_gap"),
            (11, "external_eda_practices"),
            (12, "cross_review"),
            (13, "hypothesis_catalog"),
            (14, "eda_synthesis"),
            (15, "next_research_roadmap"),
        ]
    )
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 5A systematic EDA</title><style>
body{{font-family:system-ui;max-width:1080px;margin:2rem auto;line-height:1.5}}
code{{background:#eee;padding:.1rem .3rem}}</style></head><body>
<h1>Phase 5A systematic EDA</h1>
<p>Target-aware rows are physically capped at <code>2022-12-31</code>. This is a
diagnostic report; no production candidate was promoted.</p>
<h2>Chapters</h2><ul>{chapter_links}</ul><h2>Plots</h2><ul>{plot_links}</ul>
<h2>Aggregate source tables</h2><ul>{table_links}</ul>
<p>Runner-level views remain private and Git-ignored.</p></body></html>"""
    (output / "report.html").write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default="artifacts/eda_20260901")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    script_map = {
        "b_target": "target_and_race_structure.py",
        "c_history": "horse_history.py",
        "e_opponent": "opponent_and_field.py",
        "f_connections": "connections.py",
        "g_context": "context_and_interactions.py",
        "h_errors": "model_errors.py",
    }
    script_root = Path(__file__).parent / "workstreams"
    for namespace, filename in script_map.items():
        destination = output / "workstreams" / namespace / "analyze.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_root / filename, destination)

    build_plots(output)
    build_report(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--short"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "canonical_workstream_scripts": {
                name: {
                    "path": str((script_root / filename).relative_to(repo)),
                    "sha256": sha256(script_root / filename),
                }
                for name, filename in script_map.items()
            },
        }
    )
    manifest["files"] = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
