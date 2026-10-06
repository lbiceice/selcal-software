from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_PLAN_SHA256 = "dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931"
SCRIPT = f"""
import json
from selcal.randomness import ReplicateRandomSource

plan = {REAL_PLAN_SHA256!r}
payload = []
for replicate_id in (1, 0, 1, 0):
    source = ReplicateRandomSource(plan, replicate_id, 2)
    payload.append({{
        "replicate_id": replicate_id,
        "seed_digest_sha256": source.seed_digest_sha256,
        "words": [source._next_raw64() for _ in range(3)],
    }})
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
SYNTHETIC_PRODUCTION_CAPSULE_SCRIPT = f"""
import importlib
import json
import os
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

package_path = os.path.join(os.getcwd(), "src", "selcal")
package = ModuleType("selcal")
package.__package__ = "selcal"
package.__path__ = [package_path]
package_spec = ModuleSpec("selcal", loader=None, is_package=True)
package_spec.submodule_search_locations = [package_path]
package.__spec__ = package_spec
sys.modules["selcal"] = package

first_import = sys.argv[1]
importlib.import_module(first_import)
tracked_modules = (
    "selcal.contracts_v2",
    "selcal.randomness",
    "selcal.resolution_v2",
    "selcal.calibration_v2",
)
first_snapshot = [name for name in tracked_modules if name in sys.modules]

importlib.import_module("selcal.calibration_v2")
randomness = importlib.import_module("selcal.randomness")

capsule = randomness._RANDOM_CAPSULE_V2
assert type(capsule) is randomness._RandomCapsuleV2
assert capsule is randomness._RANDOM_CAPSULE_V2
assert type(capsule)._fields == (
    "create_stream",
    "seed_digest",
    "randbelow",
    "seed_digest_for_replicate",
    "external_leaves",
)
create_stream = tuple.__getitem__(capsule, 0)
seed_digest = tuple.__getitem__(capsule, 1)
randbelow = tuple.__getitem__(capsule, 2)
plan = {REAL_PLAN_SHA256!r}
seed_stream = create_stream(plan, 0, 2)
raw_stream = create_stream(plan, 0, 2)
bounded_stream = create_stream(plan, 0, 2)
payload = {{
    "first_import": first_import,
    "first_snapshot": first_snapshot,
    "seed_digest_sha256": seed_digest(seed_stream),
    "raw64": randbelow(raw_stream, 2**64),
    "bounded": [
        randbelow(bounded_stream, bound)
        for bound in (2, 3, 17, 2**64 + 1)
    ],
}}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
EAGER_PRODUCTION_CAPSULE_SCRIPT = f"""
import importlib
import json
import sys

import selcal

tracked_modules = (
    "selcal.contracts_v2",
    "selcal.randomness",
    "selcal.resolution_v2",
    "selcal.calibration_v2",
)
first_snapshot = [name for name in tracked_modules if name in sys.modules]
randomness = importlib.import_module("selcal.randomness")
capsule = randomness._RANDOM_CAPSULE_V2
create_stream = tuple.__getitem__(capsule, 0)
seed_digest = tuple.__getitem__(capsule, 1)
randbelow = tuple.__getitem__(capsule, 2)
plan = {REAL_PLAN_SHA256!r}
seed_stream = create_stream(plan, 0, 2)
raw_stream = create_stream(plan, 0, 2)
bounded_stream = create_stream(plan, 0, 2)
payload = {{
    "first_snapshot": first_snapshot,
    "seed_digest_sha256": seed_digest(seed_stream),
    "raw64": randbelow(raw_stream, 2**64),
    "bounded": [
        randbelow(bounded_stream, bound)
        for bound in (2, 3, 17, 2**64 + 1)
    ],
}}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


def _run_subprocess(hash_seed: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        cwd=ROOT,
        env=environment,
        timeout=10,
    )
    return completed.stdout.strip()


def _run_synthetic_production_subprocess(first_import: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", SYNTHETIC_PRODUCTION_CAPSULE_SCRIPT, first_import],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        cwd=ROOT,
        env=environment,
        timeout=10,
    )
    return completed.stdout.strip()


def _run_eager_production_subprocess(_mode: str = "eager") -> str:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", EAGER_PRODUCTION_CAPSULE_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        cwd=ROOT,
        env=environment,
        timeout=10,
    )
    return completed.stdout.strip()


def test_two_independent_processes_and_repeated_builds_are_byte_identical() -> None:
    first = _run_subprocess("1")
    second = _run_subprocess("987654")

    assert first == second
    assert json.loads(first) == [
        {
            "replicate_id": 1,
            "seed_digest_sha256": (
                "0fec37b5d56de9d84e86bb178f6715dcfb300db863201679e222a3e96240056e"
            ),
            "words": [3445068325290577610, 519030488576969648, 7848502882682531391],
        },
        {
            "replicate_id": 0,
            "seed_digest_sha256": (
                "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a"
            ),
            "words": [1874999576356411599, 4645183139923492514, 4210660991089764159],
        },
        {
            "replicate_id": 1,
            "seed_digest_sha256": (
                "0fec37b5d56de9d84e86bb178f6715dcfb300db863201679e222a3e96240056e"
            ),
            "words": [3445068325290577610, 519030488576969648, 7848502882682531391],
        },
        {
            "replicate_id": 0,
            "seed_digest_sha256": (
                "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a"
            ),
            "words": [1874999576356411599, 4645183139923492514, 4210660991089764159],
        },
    ]


@pytest.mark.parametrize(
    ("first_import", "expected_first_snapshot"),
    (
        ("selcal.contracts_v2", ["selcal.contracts_v2"]),
        (
            "selcal.randomness",
            ["selcal.contracts_v2", "selcal.randomness"],
        ),
        (
            "selcal.resolution_v2",
            [
                "selcal.contracts_v2",
                "selcal.randomness",
                "selcal.resolution_v2",
            ],
        ),
        (
            "selcal.calibration_v2",
            [
                "selcal.contracts_v2",
                "selcal.randomness",
                "selcal.resolution_v2",
                "selcal.calibration_v2",
            ],
        ),
    ),
)
def test_production_capsule_kat_is_stable_for_real_submodule_first_bootstraps(
    first_import: str,
    expected_first_snapshot: list[str],
) -> None:
    assert json.loads(_run_synthetic_production_subprocess(first_import)) == {
        "first_import": first_import,
        "first_snapshot": expected_first_snapshot,
        "seed_digest_sha256": (
            "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a"
        ),
        "raw64": 1874999576356411599,
        "bounded": [1, 2, 6, 17099354796136167930],
    }


def test_real_package_eager_import_smoke_retains_the_literal_production_kat() -> None:
    assert json.loads(_run_eager_production_subprocess()) == {
        "first_snapshot": [
            "selcal.contracts_v2",
            "selcal.randomness",
            "selcal.resolution_v2",
            "selcal.calibration_v2",
        ],
        "seed_digest_sha256": (
            "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a"
        ),
        "raw64": 1874999576356411599,
        "bounded": [1, 2, 6, 17099354796136167930],
    }


@pytest.mark.parametrize(
    "runner",
    (
        _run_subprocess,
        _run_synthetic_production_subprocess,
        _run_eager_production_subprocess,
    ),
)
def test_rng_subprocesses_have_an_explicit_timeout(runner: object) -> None:
    source = textwrap.dedent(inspect.getsource(runner))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]

    assert len(calls) == 1
    assert any(keyword.arg == "timeout" for keyword in calls[0].keywords)
