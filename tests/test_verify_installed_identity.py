"""scripts/verify_installed_identity.py finds edited, missing and foreign installed files."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

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
                          encoding="utf-8", errors="replace", timeout=60)
    return done.returncode, json.loads(done.stdout)


def test_intact_installation_from_its_wheel_passes(tmp_path):
    record = _install(tmp_path / "site", {"fakepkg/__init__.py": b"x = 1\n"})
    code, report = _run(tmp_path / "site", "--wheel", str(_wheel(tmp_path / "f.whl", record)))
    assert (code, report["status"], report["files_checked"]) == (0, "PASS", 1)


def test_an_edited_or_missing_file_fails(tmp_path):
    site = tmp_path / "site"
    _install(site, {"fakepkg/__init__.py": b"x = 1\n", "fakepkg/web/index.html": b"<p>a</p>\n"})
    (site / "fakepkg/web/index.html").write_bytes(b"<p>a</p><!-- edited -->\n")
    (site / "fakepkg/__init__.py").unlink()
    code, report = _run(site)
    assert (code, report["status"]) == (1, "FAIL")
    assert report["changed"] == ["fakepkg/web/index.html"]
    assert report["missing"] == ["fakepkg/__init__.py"]


def test_an_installation_from_another_build_fails(tmp_path):
    _install(tmp_path / "site", {"fakepkg/__init__.py": b"x = 1\n"})
    other = f"fakepkg/__init__.py,{_digest(b'x = 2' + bytes([10]))},6\n"
    code, report = _run(tmp_path / "site", "--wheel", str(_wheel(tmp_path / "o.whl", other)))
    assert (code, report["status"]) == (1, "FAIL")
    assert report["not_from_wheel"]


def test_an_absent_distribution_is_a_usage_error(tmp_path):
    (tmp_path / "site").mkdir()
    code, report = _run(tmp_path / "site")
    assert (code, report["status"]) == (2, "ERROR")
