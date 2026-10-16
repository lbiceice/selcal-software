"""The paper manifest names only shipped paths, matching versions and well-formed digests."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "paper" / "selcal_softwarex_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_release_paths_exist() -> None:
    claims = _manifest()["claims"]
    paths = [path for claim in claims for path in claim.get("release_paths", ())]
    assert paths
    assert not [path for path in paths if not (ROOT / path).is_file()]


def test_manifest_names_the_current_version_and_hex_digests() -> None:
    manifest = _manifest()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert manifest["software"]["cited_release"] == project["version"]
    for archive in ("validation_records", "dengue_climate_data"):
        assert re.fullmatch(r"[0-9a-f]{64}", manifest["archives"][archive]["sha256"])
    claims = manifest["claims"]
    assert all(claim["claim"] and claim["article"] and claim["check"] for claim in claims)


def test_exact_example_plan_is_the_documented_one() -> None:
    plan = json.loads((ROOT / "examples/workflow/pearson_exact.json").read_text(encoding="utf-8"))
    assert plan["plan"]["null_name"] == "circular_shift_exact_v1"
    assert plan["plan"]["replicates"] == 99
    guide = (ROOT / "docs/guide/exact_example.md").read_text(encoding="utf-8")
    assert "selected lag 2, E = 2 of 99, p = 0.03" in guide
