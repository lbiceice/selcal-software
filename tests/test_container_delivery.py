"""Static delivery contracts; real Docker build/runtime evidence is separate."""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import Path

import pytest
from _documentation import public_documentation_text

from scripts.audit_repository_hygiene import _release_surface_paths

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("Dockerfile", ".dockerignore", "compose.yaml", "compose.ui.yaml")
BASE = (
    "python:3.12-slim-bookworm@sha256:"
    "392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e"
)


def _asset(name: str) -> str:
    path = ROOT / name
    assert path.is_file(), f"container delivery asset is absent: {name}"
    assert not path.is_symlink(), f"container delivery asset is linked: {name}"
    return path.read_text(encoding="utf-8")


def _instructions() -> list[tuple[str, str]]:
    content = _asset("Dockerfile").replace("\\\n", " ")
    return [
        (parts[0].upper(), " ".join(parts[1].split()))
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for parts in [line.strip().split(maxsplit=1)]
    ]


@pytest.mark.parametrize("name", ASSETS)
def test_container_assets_are_ordinary_nonempty_files(name: str) -> None:
    assert _asset(name).strip()


def test_container_build_pins_both_bases_and_installs_built_wheels() -> None:
    instructions = _instructions()
    assert [value for key, value in instructions if key == "FROM"] == [
        BASE + " AS builder", BASE + " AS runtime"
    ]
    boundary = instructions.index(("FROM", BASE + " AS runtime"))
    build = instructions[:boundary]
    runtime = instructions[boundary:]
    build_text = "\n".join(value for _, value in build)
    runtime_text = "\n".join(value for _, value in runtime)
    assert "setuptools==80.9.0 wheel==0.45.1" in build_text
    assert "pip wheel --no-deps --no-build-isolation --wheel-dir /wheels ." in build_text
    assert "pip download --only-binary=:all: --no-deps --dest /wheels numpy==2.4.6" in build_text
    assert ("COPY", "--from=builder /wheels /wheels") in runtime
    assert "pip install --no-cache-dir --no-index --no-deps /wheels/*.whl" in runtime_text
    assert {value for key, value in runtime if key == "COPY"} == {
        "--from=builder /wheels /wheels",
        "examples/workflow/ /opt/selcal/examples/",
    }
    assert "PYTHONPATH" not in _asset("Dockerfile")
    assert not re.search(r"(?:--editable|\s-e\s|pip\s+install\s+\.)", build_text + runtime_text)


def test_runtime_uses_nonroot_isolated_exec_and_sigint() -> None:
    instructions = _instructions()
    for expected in (
        ("USER", "1000:1000"), ("WORKDIR", "/output"), ("STOPSIGNAL", "SIGINT")
    ):
        assert expected in instructions
    for key, expected in (("ENTRYPOINT", ["python", "-I", "-m", "selcal"]),
                          ("CMD", ["doctor"])):
        values = [value for kind, value in instructions if kind == key]
        assert len(values) == 1
        assert json.loads(values[0]) == expected
    text = " ".join(value for _, value in instructions)
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "groupadd --gid 1000 selcal" in text
    assert "useradd --uid 1000 --gid 1000 --create-home selcal" in text
    assert "chown 1000:1000 /output" in text


def test_build_context_is_closed_to_only_declared_product_inputs() -> None:
    rules = [line.strip() for line in _asset(".dockerignore").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert rules[0] == "**"
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    docs = set().union(*map(set, config["tool"]["setuptools"]["data-files"].values()))
    expected = {
        "Dockerfile", ".dockerignore", "pyproject.toml", "README.md",
        "LICENSE.txt", "Licence.txt", "src/", "src/selcal/", "src/selcal/*.py",
        "src/selcal/nulls/", "src/selcal/nulls/*.py",
        "src/selcal/statistics/", "src/selcal/statistics/*.py", "src/selcal/web/",
        "src/selcal/web/index.html", "src/selcal/web/style.css", "src/selcal/web/app.js",
        "src/selcal/web/series.csv", "src/selcal/web/pearson.json",
        "docs/", "docs/architecture/", "docs/provenance/",
        "examples/", "examples/workflow/", "examples/workflow/series.csv",
        "examples/workflow/pearson.json",
    } | docs
    includes = [rule[1:] for rule in rules[1:] if rule.startswith("!")]
    assert set(includes) == expected
    directories = {name for name in expected if name.endswith("/")}
    assert {rule for rule in rules[1:] if not rule.startswith("!")} == {
        name + "**" for name in directories
    }
    assert len(rules) == len(expected) + len(directories) + 1
    assert not any("**" in rule for rule in includes)
    for directory in directories:
        index = rules.index("!" + directory)
        assert rules[index + 1] == directory + "**"
    for name in docs:
        assert (ROOT / name).is_file()


@pytest.mark.parametrize("name", ("compose.yaml", "compose.ui.yaml"))
def test_compose_keeps_one_service_without_host_privilege_or_port_mapping(name: str) -> None:
    text = _asset(name)
    assert re.findall(r"^\S[^\n]*:", text, flags=re.MULTILINE) == ["services:"]
    assert re.findall(r"^  ([A-Za-z0-9_-]+):", text, flags=re.MULTILINE) == ["selcal"]
    assert not re.search(r"^\s*(ports|privileged|devices|pid|ipc|volumes_from|entrypoint):",
                         text, flags=re.MULTILINE)
    assert "docker.sock" not in text
    assert "PYTHONPATH" not in text


def test_cli_service_defaults_to_offline_readonly_and_persistent_output() -> None:
    text = _asset("compose.yaml")
    for expected in (
        "image: selcal:local", "build: .",
        "user: '${SELCAL_UID:-1000}:${SELCAL_GID:-1000}'",
        "read_only: true", "init: true", "network_mode: none",
        "cap_drop: [ALL]", "security_opt: [no-new-privileges:true]",
        "tmpfs: [/tmp]", "stop_signal: SIGINT", "stop_grace_period: 30s",
    ):
        assert expected in text
    mounts = text.split("      - type: bind\n")
    assert len(mounts) == 3
    for mount, source, target, readonly in (
        (mounts[1], "${SELCAL_INPUT_DIR:-./examples/workflow}", "/input", "true"),
        (mounts[2], "${SELCAL_OUTPUT_DIR:-./selcal-container-output}", "/output", "false"),
    ):
        assert f"source: '{source}'" in mount
        assert f"target: {target}" in mount
        assert f"read_only: {readonly}" in mount
        assert "bind:\n          create_host_path: false" in mount


def test_ui_override_changes_only_same_service_network_and_command() -> None:
    text = _asset("compose.ui.yaml")
    assert re.findall(r"^    ([a-z_]+):", text, flags=re.MULTILINE) == [
        "network_mode", "command"
    ]
    assert "network_mode: host" in text
    assert "command: [ui, --workspace, /output/ui, --port, '${SELCAL_UI_PORT:-8765}'," in text
    assert "--no-browser]" in text


def test_container_assets_are_in_the_exact_release_inventory() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    inventory = _release_surface_paths(ROOT)
    for name in ASSETS:
        assert (ROOT / name) in inventory
        assert f"include {name}\n" in manifest
    assert (ROOT / "tests/test_container_delivery.py") in inventory
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "selcal-container-output/" in ignored


def test_readme_explains_container_first_use_security_recovery_and_evidence_scope() -> None:
    text = public_documentation_text(ROOT)
    assert "## Run with containers" in text
    section = text.split("## Run with containers", 1)[1].split("\n## ", 1)[0]
    for command in (
        "mkdir -p selcal-container-output",
        "docker compose build",
        "docker compose run --rm selcal doctor",
        "docker compose run --rm selcal validate /input/series.csv /input/pearson.json",
        "docker compose run --rm selcal run /input/series.csv /input/pearson.json",
        "docker compose run --rm selcal verify /output/run.sqlite",
        "docker compose run --rm selcal report /output/run.sqlite",
        "docker compose run --rm selcal export /output/run.sqlite",
        "docker compose run --rm selcal verify-export /output/evidence",
        "docker compose -f compose.yaml -f compose.ui.yaml up",
        "docker compose run --rm selcal resume /output/run-checkpoint",
    ):
        assert command in section
    prose = " ".join(section.split())
    for marker in (
        "SELCAL_UID",
        "SELCAL_GID",
        "SELCAL_INPUT_DIR",
        "SELCAL_OUTPUT_DIR",
        "--replay",
        "SIGINT",
        "30 seconds",
        "Ctrl-C",
        "same image",
        "Python",
        "NumPy",
        "platform",
        "127.0.0.1",
        "token",
        "host networking",
        "Linux Engine",
        "OrbStack",
        "Docker Desktop",
        "opt-in",
        "untested",
        "network access",
        "networkless",
        "Local execution was checked on 2026-09-30",
        "linux/arm64",
        "Chrome",
        "Other engines and architectures remain untested",
    ):
        assert marker in prose, marker
    assert "final integration, CI and containers\nremain unfinished" not in text
    assert "F4C" in text and "F5A" in text


def test_container_examples_use_one_bundle_compatible_storage_budget() -> None:
    text = public_documentation_text(ROOT)
    section = text.split("## Run with containers", 1)[1].split("\n## ", 1)[0]
    commands = [
        shlex.split(line)
        for line in section.splitlines()
        if line.startswith("docker compose ") and "--max-bytes" in line
    ]
    assert commands, "container examples must document their storage budgets"
    for command in commands:
        assert command.count("--max-bytes") == 1
        assert command[command.index("--max-bytes") + 1] == "8388608", command
    prose = " ".join(section.split())
    for marker in ("8 MiB", "payload", "bundle", "not a process-memory bound"):
        assert marker in prose, marker
