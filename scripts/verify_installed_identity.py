"""Check that an installed SelCal is exactly the files of the intended wheel.

Usage: python -I scripts/verify_installed_identity.py [--wheel SELCAL.whl] [--distribution NAME]

Run it with the Python of the installation being checked. With --wheel (the trusted wheel that
was installed), every file of the wheel is compared byte for byte with the file the installer
placed for it: package files under site-packages, ``<name>.data/<scheme>/`` files under the
matching sysconfig path. Both RECORD files are only lists of paths here; a RECORD edited
together with a file does not hide the edit. Files the installer adds are accepted only as the
kinds pip writes, each checked on its own:

* ``INSTALLER``, ``REQUESTED``, ``direct_url.json`` and ``RECORD`` in the dist-info folder;
* ``__pycache__`` bytecode for this interpreter, whose code must equal a fresh compilation of
  the verified source next to it;
* console scripts named in the wheel's entry points, whose program must be exactly the text a
  supported pip version writes for that entry point (LAUNCHER_TEMPLATES) and whose interpreter
  line must name this Python. On Windows the ``.exe`` must be a launcher stub shipped with the
  installed pip, then that interpreter line, then a ZIP holding only that program.

Every file inside the installed package folders and the dist-info folder must be one of these,
so an injected module is found. Without --wheel, only the hashes listed in the installed RECORD
are checked (weaker: a RECORD edited together with a file is not found).

Prints one JSON document; exit 0 only if every check passed, 1 on a mismatch, 2 on a usage or
read error. An acceptance runner that reuses an existing installation should run this first and
install into a new folder if it fails, never trust the installation because a marker file exists.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import marshal
import os
import sys
import sysconfig
import zipfile
from pathlib import Path, PurePosixPath

INSTALLER_METADATA = ("INSTALLER", "REQUESTED", "direct_url.json")
# The program pip writes into a console-script launcher, by installer version, read from real pip
# copies. Any other text is refused, so a new pip version needs a reviewed entry here.
LAUNCHER_TEMPLATES = {
    # pip 22.3, 23.0.1, 23.2.1, 24.0, 24.3.1, 25.0.1, 25.1.1: distlib 0.3.6-0.3.9 SCRIPT_TEMPLATE
    # (22.3 and 23.2.1 checked from the official source by the 2026-10-06 review).
    "distlib-0.3": (
        "# -*- coding: utf-8 -*-\n"
        "import re\n"
        "import sys\n"
        "from %(module)s import %(import_name)s\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        "    sys.exit(%(func)s())\n"
    ),
    # pip 25.2, 25.3: PipScriptMaker.script_template.
    "pip-25.2": (
        "import sys\n"
        "from %(module)s import %(import_name)s\n"
        "if __name__ == '__main__':\n"
        "    if sys.argv[0].endswith('.exe'):\n"
        "        sys.argv[0] = sys.argv[0][:-4]\n"
        "    sys.exit(%(func)s())\n"
    ),
    # pip 26.0, 26.0.1, 26.2.1: PipScriptMaker.script_template.
    "pip-26.0": (
        "import sys\n"
        "from %(module)s import %(import_name)s\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        "    sys.exit(%(func)s())\n"
    ),
}
SH_EXEC_PREFIX, SH_EXEC_SUFFIX, SH_CLOSING = b"'''exec' ", b' "$0" "$@"', b"' '''"
LEGACY_WINDOWS_LINESEP = b"\r\n"


def _record_rows(text: str) -> dict[str, str]:
    """RECORD path -> "sha256=..." ("" when the row carries no hash)."""
    return {row[0]: row[1] if len(row) >= 2 else "" for row in csv.reader(io.StringIO(text))
            if row}


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def _key(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _record_only(
    dist: importlib.metadata.Distribution, installed: dict[str, str]
) -> dict[str, object]:
    mismatched, missing, checked = [], [], 0
    for name, expected in sorted(installed.items()):
        if not expected:
            continue
        if not expected.startswith("sha256="):
            raise ValueError(f"unsupported RECORD hash for {name}")
        checked += 1
        try:
            actual = _digest(Path(str(dist.locate_file(name))).read_bytes())
        except FileNotFoundError:
            missing.append(name)
            continue
        if actual != expected:
            mismatched.append(name)
    return {"files_checked": checked, "missing": missing, "changed": mismatched,
            "reference": "installed RECORD only"}


def _read_wheel(wheel: Path) -> tuple[str, dict[str, bytes]]:
    """Return the wheel's dist-info folder name and its members other than RECORD."""
    with zipfile.ZipFile(wheel) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()
                   if not info.is_dir()}
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1 or "/" in records[0][: -len("/RECORD")]:
        raise ValueError("the wheel must hold exactly one top-level dist-info RECORD")
    info_dir = records[0][: -len("/RECORD")]
    rows = _record_rows(members.pop(records[0]).decode("utf-8"))
    rows.pop(records[0], None)
    for name in members:
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or "\\" in name:
            raise ValueError(f"unsafe wheel member {name!r}")
    if {name: _digest(data) for name, data in members.items()} != rows:
        raise ValueError("the wheel's RECORD does not match its files")
    return info_dir, members


def _wheel_targets(info_dir: str, members: dict[str, bytes], root: Path) -> dict[str, str]:
    """Wheel member -> the absolute path a standard installer places it at."""
    data_dir = info_dir[: -len(".dist-info")] + ".data/"
    paths = sysconfig.get_paths()
    schemes = {"purelib": paths["purelib"], "platlib": paths["platlib"],
               "data": paths["data"], "scripts": paths["scripts"]}
    targets = {}
    for name in members:
        if name.startswith(data_dir):
            scheme, _, rest = name[len(data_dir):].partition("/")
            if scheme not in schemes or not rest:
                raise ValueError(f"unsupported wheel data scheme in {name!r}")
            targets[name] = _key(Path(schemes[scheme], *PurePosixPath(rest).parts))
        else:
            targets[name] = _key(root.joinpath(*PurePosixPath(name).parts))
    return targets


def _bytecode_matches(path: Path, verified_sources: dict[str, bytes]) -> bool:
    """A __pycache__ file whose code equals a fresh compilation of its verified source."""
    stem, _, rest = path.name.partition(".")
    tag = sys.implementation.cache_tag
    optimize = {f"{tag}.pyc": 0, f"{tag}.opt-1.pyc": 1, f"{tag}.opt-2.pyc": 2}.get(rest)
    source = path.parent.parent / f"{stem}.py"
    if path.parent.name != "__pycache__" or optimize is None:
        return False
    data = verified_sources.get(_key(source))
    raw = path.read_bytes()
    if data is None or raw[:4] != importlib.util.MAGIC_NUMBER:
        return False
    try:
        code = marshal.loads(raw[16:])
    except (EOFError, ValueError, TypeError):
        return False
    return bool(code == compile(data, str(source), "exec", dont_inherit=True, optimize=optimize))


def _split_shebang(data: bytes) -> tuple[str, bytes] | None:
    """Split a text launcher into the interpreter its first line names and the program after it.

    Both forms distlib writes are accepted: "#!<python>" and, for long POSIX paths or paths with
    spaces, the three-line "/bin/sh" form that execs the quoted interpreter.
    """
    first, newline, rest = data.partition(b"\n")
    if not newline or not first.startswith(b"#!"):
        return None
    if first == b"#!/bin/sh":
        command, newline, rest = rest.partition(b"\n")
        closing, closing_newline, rest = rest.partition(b"\n")
        if not (newline and closing_newline and closing == SH_CLOSING
                and command.startswith(SH_EXEC_PREFIX) and command.endswith(SH_EXEC_SUFFIX)):
            return None
        interpreter = command[len(SH_EXEC_PREFIX):-len(SH_EXEC_SUFFIX)]
    else:
        interpreter = first[2:]
    if len(interpreter) > 1 and interpreter.startswith(b'"') and interpreter.endswith(b'"'):
        interpreter = interpreter[1:-1]
    try:
        return interpreter.decode("utf-8"), rest
    except UnicodeDecodeError:
        return None


def _program_matches(
    program: bytes, entry: str, templates: tuple[str, ...] | None = None
) -> bool:
    """The program is exactly one supported installer template rendered for this entry point."""
    module, _, attribute = entry.partition(":")
    module, attribute = module.strip(), attribute.strip()
    if not module or not attribute.isidentifier():
        return False
    values = {"module": module, "import_name": attribute, "func": attribute}
    return any(program == (LAUNCHER_TEMPLATES[name] % values).encode("utf-8")
               for name in (templates or tuple(LAUNCHER_TEMPLATES)))


def _text_launcher_matches(data: bytes, entry: str, interpreter: str) -> bool:
    split = _split_shebang(data)
    return (split is not None and _key(split[0]) == _key(interpreter)
            and _program_matches(split[1], entry))


def _exe_launcher_matches(data: bytes, entry: str, stubs: list[bytes], interpreter: str) -> bool:
    """distlib's Windows layout: a trusted launcher stub, "#!<python>" and a newline, then a ZIP
    archive holding only __main__.py, starting right after the newline and ending the file.

    distlib 0.3.6-0.3.8 (pip 22.3-24.0) also appended os.linesep to the shebang when it did not
    already end with it, so on Windows exactly one CRLF follows the newline. That form is accepted
    only with the program those versions write (the "distlib-0.3" template).
    """
    for stub in stubs:
        if not stub or not data.startswith(stub):
            continue
        shebang, newline, archive = data[len(stub):].partition(b"\n")
        split = _split_shebang(shebang + newline)
        if split is None or split[1] or _key(split[0]) != _key(interpreter):
            continue
        templates: tuple[str, ...] | None = None
        if archive.startswith(LEGACY_WINDOWS_LINESEP):
            archive, templates = archive[len(LEGACY_WINDOWS_LINESEP):], ("distlib-0.3",)
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as opened:
                members = opened.infolist()
                if (len(members) != 1 or members[0].filename != "__main__.py"
                        or members[0].header_offset != 0 or opened.comment
                        or archive[-22:-18] != b"PK\x05\x06"):
                    continue
                program = opened.read(members[0])
        except (zipfile.BadZipFile, EOFError, ValueError, KeyError, NotImplementedError):
            continue
        if _program_matches(program, entry, templates):
            return True
    return False


def _windows_launchers() -> list[bytes]:
    """The launcher stubs shipped with the pip installed in this environment."""
    spec = importlib.util.find_spec("pip")
    if spec is None or not spec.submodule_search_locations:
        return []
    folder = Path(next(iter(spec.submodule_search_locations)), "_vendor", "distlib")
    return [path.read_bytes() for path in sorted(folder.glob("*.exe"))]


def _script_matches(path: Path, entries: dict[str, str]) -> bool:
    if _key(path.parent) != _key(sysconfig.get_path("scripts")):
        return False
    if os.name == "nt":
        if not path.name.lower().endswith(".exe") or path.name[:-4] not in entries:
            return False
        return _exe_launcher_matches(path.read_bytes(), entries[path.name[:-4]],
                                     _windows_launchers(), sys.executable)
    if path.name not in entries:
        return False
    return _text_launcher_matches(path.read_bytes(), entries[path.name], sys.executable)


def _entry_points(members: dict[str, bytes], info_dir: str) -> dict[str, str]:
    data = members.get(f"{info_dir}/entry_points.txt")
    if data is None:
        return {}
    parser = configparser.ConfigParser(delimiters=("=",), interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(data.decode("utf-8"))
    # GUI scripts are launched with another interpreter (pythonw) and are not supported here:
    # a GUI launcher is refused as a file the installer was not expected to write.
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))


def _installer_metadata_ok(name: str, data: bytes) -> bool:
    if name == "REQUESTED":
        return data.strip() == b""
    if name == "INSTALLER":
        return data.strip().isascii() and len(data.split()) == 1
    try:
        return isinstance(json.loads(data.decode("utf-8")).get("url"), str)
    except (UnicodeDecodeError, ValueError, AttributeError):
        return False


def _against_wheel(dist: importlib.metadata.Distribution, installed: dict[str, str],
                   wheel: Path) -> dict[str, object]:
    info_dir, members = _read_wheel(wheel)
    root = Path(str(dist.locate_file("")))
    if f"{info_dir}/RECORD" not in installed:
        raise ValueError(f"the installed distribution is not the wheel's {info_dir}")
    targets = _wheel_targets(info_dir, members, root)
    by_target = {target: name for name, target in targets.items()}
    changed, missing, record_inconsistent = [], [], []
    verified_sources = {}
    for name, target in sorted(targets.items()):
        try:
            data = Path(target).read_bytes()
        except FileNotFoundError:
            missing.append(name)
            continue
        if data != members[name]:
            changed.append(name)
        elif name.endswith(".py"):
            verified_sources[target] = data
    entries = _entry_points(members, info_dir)
    generated: list[dict[str, str]] = []
    not_from_wheel = []
    record_targets = {}
    for row, expected in sorted(installed.items()):
        # _key() is only for comparing locations (on Windows it lowercases); the file is
        # classified by its real path, so INSTALLER and REQUESTED keep their case.
        location = Path(str(dist.locate_file(row)))
        target = _key(location)
        record_targets[target] = row
        if target in by_target:
            if expected and expected != _digest(members[by_target[target]]):
                record_inconsistent.append(row)
            continue
        if not _generated_ok(location, info_dir, root, verified_sources, entries, generated):
            not_from_wheel.append(row)
    for name, target in targets.items():
        if target not in record_targets:
            record_inconsistent.append(name)
    unexpected = []
    folders = {PurePosixPath(name).parts[0] for name in members if "/" in name
               and not name.startswith(info_dir[: -len(".dist-info")] + ".data/")}
    for folder in sorted(folders):
        for base, directories, files in os.walk(root / folder, followlinks=False):
            for entry in sorted(directories + files):
                path = Path(base, entry)
                # Files listed in RECORD were classified above; anything else here must be
                # bytecode the interpreter wrote later, checked the same way.
                known = _key(path) in by_target or _key(path) in record_targets
                if path.is_symlink() or (path.is_file() and not known and not _generated_ok(
                        path, info_dir, root, verified_sources, entries, generated)):
                    unexpected.append(path.relative_to(root).as_posix())
    return {"wheel": wheel.name, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "reference": "trusted wheel bytes", "files_checked": len(targets),
            "missing": missing, "changed": changed, "not_from_wheel": not_from_wheel,
            "unexpected": unexpected, "record_inconsistent": sorted(set(record_inconsistent)),
            "installer_generated": generated}


def _generated_ok(path: Path, info_dir: str, root: Path, verified_sources: dict[str, bytes],
                  entries: dict[str, str], generated: list[dict[str, str]]) -> bool:
    """Accept one file the installer wrote, by kind, recording what it was checked as."""
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    if path.is_symlink() or not path.is_file():
        return False
    if relative == f"{info_dir}/RECORD":
        kind = "installer RECORD (path list only)"
    elif path.parent.name == info_dir and path.name in INSTALLER_METADATA:
        if not _installer_metadata_ok(path.name, path.read_bytes()):
            return False
        kind = f"installer metadata {path.name}"
    elif path.parent.name == "__pycache__":
        if not _bytecode_matches(path, verified_sources):
            return False
        kind = "bytecode equal to a fresh compilation of the verified source"
    elif _script_matches(path, entries):
        kind = "entry-point launcher for this Python"
    else:
        return False
    generated.append({"path": relative, "kind": kind})
    return True


def check(distribution: str, wheel: Path | None) -> dict[str, object]:
    dist = importlib.metadata.distribution(distribution)
    record = dist.read_text("RECORD")
    if record is None:
        raise ValueError("the installed distribution has no RECORD")
    installed = _record_rows(record)
    report: dict[str, object] = {"distribution": distribution, "version": dist.version,
                                 "python": sys.executable}
    report |= _record_only(dist, installed) if wheel is None else _against_wheel(
        dist, installed, wheel)
    failures = ("missing", "changed", "not_from_wheel", "unexpected", "record_inconsistent")
    report["status"] = "FAIL" if any(report.get(key) for key in failures) else "PASS"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--distribution", default="selcal")
    args = parser.parse_args(argv)
    try:
        report = check(args.distribution, args.wheel)
    except (OSError, ValueError, KeyError, UnicodeDecodeError, zipfile.BadZipFile,
            configparser.Error, importlib.metadata.PackageNotFoundError) as error:
        print(json.dumps({"status": "ERROR", "detail": str(error)}))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
