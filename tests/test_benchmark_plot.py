from __future__ import annotations

import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.plot_in_memory_benchmark import _memory_panel, _points, _slice_points

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLOTTER = REPOSITORY_ROOT / "scripts" / "plot_in_memory_benchmark.py"
RECEIPT = REPOSITORY_ROOT / "docs" / "benchmarks" / "in_memory_standard_20260922.json"


def test_plotting_dependency_is_isolated_from_runtime_dependencies() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text("utf-8"))

    assert pyproject["project"]["dependencies"] == ["numpy>=1.26,<2.5"]
    assert pyproject["project"]["optional-dependencies"]["benchmark"] == [
        "matplotlib>=3.11.1,<4"
    ]


def _render(output_stem: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PLOTTER),
            "--input",
            str(RECEIPT),
            "--output-stem",
            str(output_stem),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )


def test_plotter_exports_deterministic_vector_and_high_resolution_raster(
    tmp_path: Path,
) -> None:
    first_stem = tmp_path / "first" / "in_memory_scaling"
    second_stem = tmp_path / "second" / "in_memory_scaling"
    first = _render(first_stem)
    second = _render(second_stem)

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout == ""

    first_svg = first_stem.with_suffix(".svg").read_bytes()
    second_svg = second_stem.with_suffix(".svg").read_bytes()
    first_png = first_stem.with_suffix(".png").read_bytes()
    second_png = second_stem.with_suffix(".png").read_bytes()
    assert first_svg == second_svg
    assert first_png == second_png

    svg_text = first_svg.decode("utf-8")
    for label in (
        "Replicates (B)",
        "Candidates (C)",
        "Series length (N)",
        "Wall time (s)",
        "Peak traced memory (MiB)",
    ):
        assert label in svg_text

    assert first_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", first_png[16:24]) == (2160, 1440)


def test_memory_panels_reserve_headroom_for_open_square_markers() -> None:
    points = _points(RECEIPT)
    for dimension in ("B", "C", "N"):
        items = _slice_points(points, dimension)
        figure, axis = plt.subplots()
        try:
            _memory_panel(axis, items)
            peak = max(point.peak_mib for _, point in items)
            assert axis.get_ylim()[1] >= peak * 1.12
        finally:
            plt.close(figure)
