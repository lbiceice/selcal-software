"""Render the frozen SelCal in-memory benchmark receipt as a paper figure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SCHEMA: Final = "selcal.benchmark.in-memory.v1"
STANDARD_CONTRACT_SHA256: Final = (
    "c3786267f82c0313ecd433fef97e70ff845f620474a2a5e5af314d57da42c17f"
)
WALL_COLOR: Final = "#356E9A"
MEMORY_COLOR: Final = "#D97732"
INK: Final = "#252A2E"
GRID: Final = "#D9DEE3"


@dataclass(frozen=True)
class BenchmarkPoint:
    planned_replicates: int
    candidate_count: int
    series_length: int
    wall_min: float
    wall_median: float
    wall_max: float
    peak_mib: float


def _integer(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise ValueError(f"benchmark case field {key!r} must be an integer")
    return value


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"benchmark case field {key!r} must be numeric")
    return float(value)


def _points(receipt_path: Path) -> tuple[BenchmarkPoint, ...]:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark receipt must be a JSON object")
    if payload.get("schema") != SCHEMA or payload.get("preset") != "standard":
        raise ValueError("benchmark receipt has the wrong schema or preset")
    if payload.get("benchmark_contract_sha256") != STANDARD_CONTRACT_SHA256:
        raise ValueError("benchmark receipt does not match the standard contract")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("benchmark receipt cases must be a list")

    points: list[BenchmarkPoint] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise ValueError("benchmark case must be a JSON object")
        row: dict[str, object] = raw_case
        if row.get("result_status_counts") != {"complete": 3}:
            raise ValueError("figure accepts only three-repeat complete benchmark cases")
        wall_min = _number(row, "wall_seconds_min")
        wall_median = _number(row, "wall_seconds_median")
        wall_max = _number(row, "wall_seconds_max")
        if not 0 < wall_min <= wall_median <= wall_max:
            raise ValueError("benchmark wall-time summary is inconsistent")
        peak_bytes = _integer(row, "peak_traced_bytes_max")
        if peak_bytes <= 0:
            raise ValueError("benchmark peak memory must be positive")
        points.append(
            BenchmarkPoint(
                planned_replicates=_integer(row, "planned_replicates"),
                candidate_count=_integer(row, "candidate_count"),
                series_length=_integer(row, "series_length"),
                wall_min=wall_min,
                wall_median=wall_median,
                wall_max=wall_max,
                peak_mib=peak_bytes / (1024 * 1024),
            )
        )
    return tuple(points)


def _slice_points(
    points: tuple[BenchmarkPoint, ...],
    dimension: str,
) -> tuple[tuple[int, BenchmarkPoint], ...]:
    if dimension == "B":
        selected = (
            (point.planned_replicates, point)
            for point in points
            if point.candidate_count == 4 and point.series_length == 128
        )
        expected = (25, 100, 250)
    elif dimension == "C":
        selected = (
            (point.candidate_count, point)
            for point in points
            if point.planned_replicates == 100 and point.series_length == 128
        )
        expected = (2, 4, 8)
    elif dimension == "N":
        selected = (
            (point.series_length, point)
            for point in points
            if point.planned_replicates == 100 and point.candidate_count == 4
        )
        expected = (64, 128, 256)
    else:
        raise ValueError(f"unsupported benchmark dimension: {dimension}")
    ordered = tuple(sorted(selected, key=lambda item: item[0]))
    if tuple(value for value, _ in ordered) != expected:
        raise ValueError(f"benchmark receipt does not contain the expected {dimension} slice")
    return ordered


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(colors=INK, labelsize=7, width=0.7, length=3)
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def _wall_panel(axis: Axes, items: tuple[tuple[int, BenchmarkPoint], ...]) -> None:
    x = [value for value, _ in items]
    median = [point.wall_median for _, point in items]
    lower = [point.wall_median - point.wall_min for _, point in items]
    upper = [point.wall_max - point.wall_median for _, point in items]
    axis.errorbar(
        x,
        median,
        yerr=[lower, upper],
        color=WALL_COLOR,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=WALL_COLOR,
        markersize=4.5,
        linewidth=1.2,
        capsize=2.5,
        capthick=0.9,
    )
    axis.set_xticks(x)
    axis.set_ylim(0, max(point.wall_max for _, point in items) * 1.12)
    _style_axis(axis)


def _memory_panel(axis: Axes, items: tuple[tuple[int, BenchmarkPoint], ...]) -> None:
    x = [value for value, _ in items]
    peak = [point.peak_mib for _, point in items]
    axis.plot(
        x,
        peak,
        color=MEMORY_COLOR,
        marker="s",
        markerfacecolor="white",
        markeredgecolor=MEMORY_COLOR,
        markersize=4.5,
        linewidth=1.2,
    )
    axis.set_xticks(x)
    axis.set_ylim(0, max(peak) * 1.15)
    _style_axis(axis)


def render_figure(receipt_path: Path, output_stem: Path) -> None:
    """Render deterministic SVG and 300-dpi PNG files from a standard receipt."""

    points = _points(receipt_path)
    dimensions = ("B", "C", "N")
    slices = tuple(_slice_points(points, dimension) for dimension in dimensions)

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "selcal-in-memory-benchmark-v1",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    titles = ("Replicate scaling", "Candidate scaling", "Series-length scaling")
    x_labels = ("Replicates (B)", "Candidates (C)", "Series length (N)")
    panel_labels = ("a", "b", "c", "d", "e", "f")

    for column, items in enumerate(slices):
        _wall_panel(axes[0, column], items)
        _memory_panel(axes[1, column], items)
        axes[0, column].set_title(titles[column], fontsize=8, fontweight="bold", pad=6)
        axes[1, column].set_xlabel(x_labels[column], fontsize=7)
    axes[0, 0].set_ylabel("Wall time (s)", fontsize=7)
    axes[1, 0].set_ylabel("Peak traced memory (MiB)", fontsize=7)

    for label, axis in zip(panel_labels, axes.flat, strict=True):
        axis.text(
            -0.18,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )

    figure.suptitle(
        "SelCal in-memory performance characterization",
        fontsize=10,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.025,
        "Single process; three repeats per case; bars show observed min-max range.",
        ha="center",
        fontsize=6.5,
        color=INK,
    )
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.86, wspace=0.34, hspace=0.48)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".svg"),
        format="svg",
        metadata={"Creator": "SelCal benchmark plotter", "Date": None},
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        format="png",
        dpi=300,
        metadata={"Software": "SelCal benchmark plotter"},
    )
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the SelCal benchmark figure.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    arguments = parser.parse_args(argv)
    render_figure(arguments.input.resolve(strict=True), arguments.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
