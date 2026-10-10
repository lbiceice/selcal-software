"""scripts/verify_installed_identity.py finds edited, missing and foreign installed files.

The --wheel checks run against real pip installations into new virtual environments, so the
files pip adds (bytecode, entry-point launchers, installer metadata, moved .data files) are
exercised as they occur; the SelCal wheel itself is checked the same way by scripts/ci_check.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import py_compile
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_installed_identity.py"


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def _install(site: Path, files: dict[str, bytes]) -> str:
    info = site / "fakepkg-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: fakepkg\nVersion: 1.0\n")
    rows = []
    for name, data in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(f"{name},{_digest(data)},{len(data)}")
    rows.append("fakepkg-1.0.dist-info/METADATA,,")
    rows.append("fakepkg-1.0.dist-info/RECORD,,")
    record = "\n".join(rows) + "\n"
    (info / "RECORD").write_text(record, encoding="utf-8")
    return record


def _wheel(path: Path, record: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fakepkg-1.0.dist-info/RECORD", record)
    return path


def _run(site: Path, *args: str) -> tuple[int, dict[str, object]]:
    code = (
        f"import sys, runpy; sys.path.insert(0, {str(site)!r}); "
        f"sys.argv = ['verify', '--distribution', 'fakepkg', *{list(args)!r}]; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                          encoding="ascii", errors="strict", timeout=60)
    return done.returncode, json.loads(done.stdout)


def test_an_edited_or_missing_file_fails(tmp_path):
    site = tmp_path / "site"
    _install(site, {"fakepkg/__init__.py": b"x = 1\n", "fakepkg/web/index.html": b"<p>a</p>\n"})
    (site / "fakepkg/web/index.html").write_bytes(b"<p>a</p><!-- edited -->\n")
    (site / "fakepkg/__init__.py").unlink()
    code, report = _run(site)
    assert (code, report["status"]) == (1, "FAIL")
    assert report["changed"] == ["fakepkg/web/index.html"]
    assert report["missing"] == ["fakepkg/__init__.py"]


def test_an_absent_distribution_is_a_usage_error(tmp_path):
    (tmp_path / "site").mkdir()
    code, report = _run(tmp_path / "site")
    assert (code, report["status"]) == (2, "ERROR")


def _wheel_bytes(core: bytes = b"def main():\n    return 0\n") -> dict[str, bytes]:
    info = "fakepkg-1.0.dist-info"
    members = {
        "fakepkg/__init__.py": b"VALUE = 1\n",
        "fakepkg/core.py": core,
        "fakepkg-1.0.data/data/share/fakepkg/guide.txt": b"read me\n",
        f"{info}/METADATA": b"Metadata-Version: 2.1\nName: fakepkg\nVersion: 1.0\n",
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                         b"Tag: py3-none-any\n",
        f"{info}/entry_points.txt": b"[console_scripts]\nfake-cli = fakepkg.core:main\n",
    }
    rows = [f"{name},{_digest(data)},{len(data)}" for name, data in members.items()]
    members[f"{info}/RECORD"] = ("\n".join([*rows, f"{info}/RECORD,,"]) + "\n").encode()
    return members


def _build_wheel(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _sysconfig_path(python: Path, name: str) -> Path:
    # R17 item 6: the child's console encoding (cp936 on a Chinese Windows) is not the parent's;
    # an ASCII-escaped JSON string survives either way and is decoded strictly.
    probe = f"import json, sysconfig; print(json.dumps(sysconfig.get_path({name!r})))"
    done = subprocess.run([str(python), "-I", "-c", probe],
                          check=True, capture_output=True, text=True, encoding="ascii",
                          errors="strict", timeout=60)
    return Path(json.loads(done.stdout))


def _pip_install(tmp_path: Path, wheel: Path) -> tuple[Path, Path]:
    """A new virtual environment with the wheel installed by its own pip; returns python, site."""
    environment = tmp_path / "env"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True,
                   capture_output=True, timeout=300)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-I", "-m", "pip", "--isolated", "install", "--no-deps",
                    "--no-index", "--disable-pip-version-check", str(wheel)],
                   check=True, capture_output=True, timeout=300)
    return python, _sysconfig_path(python, "purelib")


def _verify(python: Path, wheel: Path) -> tuple[int, dict[str, object]]:
    done = subprocess.run([str(python), "-I", str(SCRIPT), "--distribution", "fakepkg",
                           "--wheel", str(wheel)], capture_output=True, text=True,
                          encoding="ascii", errors="strict", timeout=120)
    return done.returncode, json.loads(done.stdout)


def _installed(tmp_path: Path) -> tuple[Path, Path, Path]:
    wheel = _build_wheel(tmp_path / "dist/fakepkg-1.0-py3-none-any.whl", _wheel_bytes())
    python, site = _pip_install(tmp_path, wheel)
    return wheel, python, site


def test_a_normal_pip_installation_of_the_wheel_passes(tmp_path):
    wheel, python, _ = _installed(tmp_path)
    code, report = _verify(python, wheel)
    assert (code, report["status"], report["reference"]) == (0, "PASS", "trusted wheel bytes")
    assert report["files_checked"] == 6
    kinds = {row["kind"] for row in report["installer_generated"]}
    assert "entry-point launcher for this Python" in kinds
    assert "bytecode equal to a fresh compilation of the verified source" in kinds
    assert "installer metadata INSTALLER" in kinds


def test_a_single_byte_change_or_a_missing_file_fails(tmp_path):
    wheel, python, site = _installed(tmp_path)
    (site / "fakepkg/__init__.py").write_bytes(b"VALUE = 2\n")
    (site / "fakepkg/core.py").unlink()
    code, report = _verify(python, wheel)
    assert (code, report["status"]) == (1, "FAIL")
    assert report["changed"] == ["fakepkg/__init__.py"]
    assert report["missing"] == ["fakepkg/core.py"]


def test_a_moved_data_file_is_compared_with_the_wheel(tmp_path):
    wheel, python, _ = _installed(tmp_path)
    guide = _sysconfig_path(python, "data") / "share/fakepkg/guide.txt"
    guide.write_bytes(b"read me!\n")
    code, report = _verify(python, wheel)
    assert (code, report["changed"]) == (1, ["fakepkg-1.0.data/data/share/fakepkg/guide.txt"])


def test_an_edit_hidden_by_an_edited_record_still_fails(tmp_path):
    wheel, python, site = _installed(tmp_path)
    edited = b"def main():\n    return 1\n"
    (site / "fakepkg/core.py").write_bytes(edited)
    record = site / "fakepkg-1.0.dist-info/RECORD"
    rows = record.read_text(encoding="utf-8").splitlines()
    rows = [f"fakepkg/core.py,{_digest(edited)},{len(edited)}" if row.startswith(
        "fakepkg/core.py,") else row for row in rows]
    record.write_text("\n".join(rows) + "\n", encoding="utf-8")
    code, report = _verify(python, wheel)
    assert (code, report["changed"]) == (1, ["fakepkg/core.py"])
    assert report["record_inconsistent"] == ["fakepkg/core.py"]


def test_an_installation_from_another_build_fails(tmp_path):
    _, python, _ = _installed(tmp_path)
    other = _build_wheel(tmp_path / "other/fakepkg-1.0-py3-none-any.whl",
                         _wheel_bytes(core=b"def main():\n    return 2\n"))
    code, report = _verify(python, other)
    assert (code, report["status"], report["changed"]) == (1, "FAIL", ["fakepkg/core.py"])


def test_injected_source_tampered_bytecode_and_an_edited_launcher_fail(tmp_path):
    wheel, python, site = _installed(tmp_path)
    (site / "fakepkg/extra.py").write_bytes(b"import os\n")
    caches = sorted((site / "fakepkg/__pycache__").glob("core.*.pyc"))
    assert caches, "pip wrote no bytecode"
    other = tmp_path / "other_core.py"
    other.write_bytes(b"def main():\n    return 3\n")
    py_compile.compile(str(other), cfile=str(caches[0]), doraise=True)
    scripts = _sysconfig_path(python, "scripts")
    launcher = scripts / ("fake-cli.exe" if os.name == "nt" else "fake-cli")
    if os.name != "nt":
        text = launcher.read_text(encoding="utf-8")
        launcher.write_text(text.replace("from fakepkg.core import main",
                                         "from fakepkg.extra import main"), encoding="utf-8")
    code, report = _verify(python, wheel)
    assert (code, report["status"]) == (1, "FAIL")
    assert report["unexpected"] == ["fakepkg/extra.py"]
    expected = {f"fakepkg/__pycache__/{caches[0].name}"}
    if os.name != "nt":
        expected.add(os.path.relpath(launcher, site).replace(os.sep, "/"))
    assert set(report["not_from_wheel"]) == expected


# R16-LAUNCH-01/02 (2026-10-06 review): entry-point launchers are compared with the exact text the
# installer writes. These format-level checks build launchers with the installer's own code in
# memory; a native Windows run is still required for a real .exe.
WINDOWS_PYTHON = r"C:\SelCal 验收\software\.selcal-user\venv\Scripts\python.exe"


def _verifier() -> dict[str, object]:
    import runpy

    return runpy.run_path(str(SCRIPT))


class _Captured:
    """distlib file operations that keep written bytes in memory."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dry_run = False

    def write_binary_file(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def set_executable_mode(self, paths: list[str]) -> None:
        pass


def _pip_launcher(tmp_path: Path, windows: bool) -> tuple[bytes, list[bytes]]:
    """A console launcher for fakepkg.core:main made by this pip's own ScriptMaker."""
    import pip._vendor.distlib as distlib
    from pip._internal.operations.install.wheel import PipScriptMaker

    stubs = [path.read_bytes() for path in sorted(Path(distlib.__file__).parent.glob("*.exe"))]
    maker = PipScriptMaker(None, str(tmp_path))
    maker.variants = {""}
    maker._fileop = _Captured()
    if windows:
        # distlib's Windows branch: launcher stub + "#!<interpreter>\n" + ZIP(__main__.py).
        maker._is_nt = True
        maker.add_launchers = True
        maker._get_launcher = lambda kind: (Path(distlib.__file__).parent / f"{kind}64.exe"
                                            ).read_bytes()
        maker._build_shebang = lambda executable, post: b"#!" + executable + post + b"\n"
        maker.executable = f'"{WINDOWS_PYTHON}"'
    else:
        # R17 item 7: on a Windows host distlib writes .exe launchers by default; a text launcher
        # must be asked for explicitly, and the fixture checks that it got one.
        maker._is_nt = False
        maker.add_launchers = False
        maker.executable = sys.executable
    maker.make("fake-cli = fakepkg.core:main")
    (data,) = maker._fileop.files.values()
    assert data.startswith(b"MZ") if windows else data.startswith(b"#!"), data[:4]
    return data, stubs


def test_the_running_pips_text_launcher_is_accepted(tmp_path):
    import pip

    data, _ = _pip_launcher(tmp_path, windows=False)
    if os.name == "nt" and b"\n\r\n" in data.split(b"import", 1)[0]:
        # distlib 0.3.6-0.3.8 (pip 22.3-24.0) appends os.linesep to the shebang; on Windows that
        # is a CRLF after the "\n". pip never installs a text launcher on Windows (it writes .exe,
        # checked by the exe tests), so this forced POSIX layout does not exist there.
        pytest.skip(f"pip {pip.__version__}: text launcher with Windows linesep is not installable")
    verifier = _verifier()
    assert verifier["_text_launcher_matches"](data, "fakepkg.core:main", sys.executable), (
        pip.__version__)
    tampered = data.replace(b"from fakepkg.core import main", b"from fakepkg.extra import main")
    assert not verifier["_text_launcher_matches"](tampered, "fakepkg.core:main", sys.executable)
    assert not verifier["_text_launcher_matches"](data, "fakepkg.core:main", "/usr/bin/python3")
    appended = data + b"import os\n"
    assert not verifier["_text_launcher_matches"](appended, "fakepkg.core:main", sys.executable)


def test_every_supported_installer_template_is_accepted_only_for_its_entry():
    verifier = _verifier()
    templates = verifier["LAUNCHER_TEMPLATES"]
    assert len(templates) == 3
    for template in templates.values():
        body = template % {"module": "fakepkg.core", "import_name": "main", "func": "main"}
        text = ("#!" + sys.executable + "\n" + body).encode()
        assert verifier["_text_launcher_matches"](text, "fakepkg.core:main", sys.executable)
        assert not verifier["_text_launcher_matches"](text, "fakepkg.other:main", sys.executable)
        long_form = ("#!/bin/sh\n'''exec' \"" + sys.executable + "\" \"$0\" \"$@\"\n' '''\n"
                     + body).encode()
        assert verifier["_text_launcher_matches"](long_form, "fakepkg.core:main", sys.executable)


def test_a_standard_windows_exe_launcher_is_accepted(tmp_path):
    data, stubs = _pip_launcher(tmp_path, windows=True)
    verifier = _verifier()
    check = verifier["_exe_launcher_matches"]
    assert check(data, "fakepkg.core:main", stubs, WINDOWS_PYTHON)


def test_altered_windows_exe_launchers_are_refused(tmp_path):
    data, stubs = _pip_launcher(tmp_path, windows=True)
    check = _verifier()["_exe_launcher_matches"]
    entry = "fakepkg.core:main"
    stub = next(stub for stub in stubs if data.startswith(stub))
    rest = data[len(stub):]
    shebang, _, archive = rest.partition(b"\n")
    assert not check(data, entry, stubs, r"C:\Other\python.exe")  # another interpreter
    assert not check(data, "fakepkg.other:main", stubs, WINDOWS_PYTHON)  # another entry
    unknown = bytearray(stub)
    unknown[len(unknown) // 2] ^= 1
    assert not check(bytes(unknown) + rest, entry, stubs, WINDOWS_PYTHON)  # unknown stub
    assert not check(data[:-1], entry, stubs, WINDOWS_PYTHON)  # truncated
    assert not check(stub + shebang + b"\nX" + archive, entry, stubs, WINDOWS_PYTHON)  # injected
    for members in ({"__main__.py": b"import os\n"},
                    {"__main__.py": zipfile.ZipFile(io.BytesIO(archive)).read("__main__.py"),
                     "extra.py": b"x = 1\n"}):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as rebuilt:
            for name, content in members.items():
                rebuilt.writestr(name, content)
        # tampered program; extra ZIP member
        assert not check(stub + shebang + b"\n" + stream.getvalue(), entry, stubs, WINDOWS_PYTHON)


# R16-LAUNCH-03 (2026-10-06 review): distlib 0.3.6-0.3.8 (pip 22.3-24.0) appends os.linesep to the
# shebang, so on Windows one CRLF sits between the interpreter line and the ZIP.
_LEGACY_EXE_PROBE = r'''
import base64, io, json, os, sys, zipfile
import pip
import pip._vendor.distlib as distlib
import pip._vendor.distlib.scripts as scripts
from pip._internal.operations.install.wheel import PipScriptMaker

class Captured:
    def __init__(self):
        self.files, self.dry_run = {}, False
    def write_binary_file(self, path, data):
        self.files[path] = data
    def set_executable_mode(self, paths):
        pass

maker = PipScriptMaker(None, sys.argv[1])
maker.variants = {""}
maker._fileop = Captured()
maker._is_nt = True
maker.add_launchers = True
folder = os.path.dirname(distlib.__file__)
maker._get_launcher = lambda kind: open(os.path.join(folder, kind + "64.exe"), "rb").read()
maker._build_shebang = lambda executable, post: b"#!" + executable + post + b"\n"
maker.executable = '"' + sys.argv[2] + '"'
os.linesep = "\r\n"  # the Windows value; only this process
maker.make("fake-cli = fakepkg.core:main")
(data,) = maker._fileop.files.values()
appends = "linesep" in open(scripts.__file__, encoding="utf-8").read()
print(json.dumps({"pip": pip.__version__, "appends_linesep": appends,
                  "stubs": [base64.b64encode(open(os.path.join(folder, name), "rb").read()).decode()
                            for name in sorted(os.listdir(folder)) if name.endswith(".exe")],
                  "exe": base64.b64encode(data).decode()}))
'''


def _legacy_pip_exe(tmp_path: Path) -> tuple[str, bool, bytes, list[bytes]]:
    """A Windows launcher written by the real code of the pip bundled with this Python."""
    environment = tmp_path / "bundled-pip"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True,
                   capture_output=True, timeout=300)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    done = subprocess.run([str(python), "-I", "-c", _LEGACY_EXE_PROBE, str(tmp_path),
                           WINDOWS_PYTHON], check=True, capture_output=True, text=True,
                          encoding="ascii", errors="strict", timeout=120)
    found = json.loads(done.stdout)
    return (found["pip"], found["appends_linesep"], base64.b64decode(found["exe"]),
            [base64.b64decode(stub) for stub in found["stubs"]])


def test_the_bundled_pips_own_windows_launcher_is_accepted(tmp_path):
    version, appends, data, stubs = _legacy_pip_exe(tmp_path)
    stub = next(stub for stub in stubs if data.startswith(stub))
    after_shebang = data[len(stub):].partition(b"\n")[2]
    # The probe reproduces the installer's own format: a CRLF exactly when its distlib appends one.
    assert after_shebang.startswith(b"\r\nPK\x03\x04") is appends, version
    check = _verifier()["_exe_launcher_matches"]
    assert check(data, "fakepkg.core:main", stubs, WINDOWS_PYTHON), version


def test_only_the_exact_legacy_separator_is_accepted(tmp_path):
    verifier = _verifier()
    check = verifier["_exe_launcher_matches"]
    entry = "fakepkg.core:main"
    data, stubs = _pip_launcher(tmp_path, windows=True)  # current pip: no CRLF
    stub = next(stub for stub in stubs if data.startswith(stub))
    shebang, _, archive = data[len(stub):].partition(b"\n")

    def launcher(separator: bytes, template: str) -> bytes:
        body = verifier["LAUNCHER_TEMPLATES"][template] % {
            "module": "fakepkg.core", "import_name": "main", "func": "main"}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as rebuilt:
            rebuilt.writestr("__main__.py", body.encode())
        return stub + shebang + b"\n" + separator + stream.getvalue()

    assert check(launcher(b"", "distlib-0.3"), entry, stubs, WINDOWS_PYTHON)
    assert check(launcher(b"\r\n", "distlib-0.3"), entry, stubs, WINDOWS_PYTHON)
    # Only that one separator, and only with the program those pip versions write.
    for separator in (b"\r\n\r\n", b"\n", b"\r", b" ", b"\r\n\n", b"X"):
        assert not check(launcher(separator, "distlib-0.3"), entry, stubs, WINDOWS_PYTHON)
    for template in ("pip-25.2", "pip-26.0"):
        assert check(launcher(b"", template), entry, stubs, WINDOWS_PYTHON)
        assert not check(launcher(b"\r\n", template), entry, stubs, WINDOWS_PYTHON)
    legacy = launcher(b"\r\n", "distlib-0.3")
    assert not check(legacy, entry, stubs, r"C:\Other\python.exe")  # another interpreter
    assert not check(legacy[:-1], entry, stubs, WINDOWS_PYTHON)  # truncated
    assert not check(legacy, "fakepkg.extra:main", stubs, WINDOWS_PYTHON)  # another entry
    # pip <= 24.0 (distlib 0.3.6-0.3.8) writes the CRLF on Windows; later pip writes none.
    assert archive.startswith(b"PK\x03\x04") or (
        os.linesep == "\r\n" and archive.startswith(b"\r\nPK\x03\x04"))
    assert check(data, entry, stubs, WINDOWS_PYTHON)  # whatever the running pip wrote


# R17 item 1 (R16 Windows return): Windows normcase lowercases the path keys used to compare
# locations; the real file names (INSTALLER, REQUESTED) must still be classified by their own case.
_CASE_FOLDING_RUN = (
    "import os.path, runpy, sys; os.path.normcase = str.lower; script, wheel = sys.argv[1:3]; "
    "sys.argv = ['verify', '--distribution', 'fakepkg', '--wheel', wheel]; "
    "runpy.run_path(script, run_name='__main__')"
)


def test_case_folded_path_keys_do_not_reject_installer_metadata(tmp_path):
    wheel, python, _ = _installed(tmp_path)
    done = subprocess.run([str(python), "-I", "-c", _CASE_FOLDING_RUN, str(SCRIPT), str(wheel)],
                          capture_output=True, text=True, encoding="ascii", errors="strict",
                          timeout=120)
    report = json.loads(done.stdout)
    assert (done.returncode, report["status"], report["not_from_wheel"]) == (0, "PASS", []), report
    kinds = {row["kind"] for row in report["installer_generated"]}
    assert {"installer metadata INSTALLER", "installer metadata REQUESTED"} <= kinds


def test_case_folded_path_keys_still_refuse_a_changed_or_injected_file(tmp_path):
    wheel, python, site = _installed(tmp_path)
    (site / "fakepkg/core.py").write_bytes(b"def main():\n    return 9\n")
    (site / "fakepkg/Extra.py").write_bytes(b"import os\n")
    (site / "fakepkg-1.0.dist-info/INSTALLER").write_bytes(b"pip\nsomething else\n")
    done = subprocess.run([str(python), "-I", "-c", _CASE_FOLDING_RUN, str(SCRIPT), str(wheel)],
                          capture_output=True, text=True, encoding="ascii", errors="strict",
                          timeout=120)
    report = json.loads(done.stdout)
    assert (done.returncode, report["status"]) == (1, "FAIL")
    assert report["changed"] == ["fakepkg/core.py"]
    assert report["unexpected"] == ["fakepkg/Extra.py"]
    # The edited INSTALLER is refused; so is the bytecode of the changed core.py, which no longer
    # equals a compilation of verified source.
    assert "fakepkg-1.0.dist-info/INSTALLER" in report["not_from_wheel"]
    assert all(row == "fakepkg-1.0.dist-info/INSTALLER" or row.startswith(
        "fakepkg/__pycache__/core.") for row in report["not_from_wheel"])
