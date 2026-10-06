"""Every text read or write names its encoding (R13 Windows return, W13-01).

On Chinese Windows the default text encoding is cp936, and ``python -I`` ignores PYTHONUTF8.
Four tests read UTF-8 web assets with a bare ``read_text()`` and failed there; macOS and Linux
default to UTF-8 and never showed it. This scan fails on the Mac whenever a text-mode
``read_text``, ``write_text`` or ``open`` lacks ``encoding=``. Binary modes and archive or OS
interfaces (``os.open``, ``tarfile.open``, ``zipfile``, ``gzip``, ``webbrowser.open``) are not text.
This is a static guard over direct calls, not a proof for dynamically built calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOT_TEXT_FILES = {"os", "webbrowser", "tarfile", "zipfile", "gzip"}
# Hash-bound historical evidence harnesses: their bytes are pinned by retained evidence and they
# are never executed by the test suite, so editing them would falsify that evidence.
HASH_BOUND = {"scripts/verify_input_package_install.py"}


def _missing_encoding(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if name in {"run", "Popen", "check_output", "check_call", "call"}:
            # R15 deep-path check: text-mode child output was decoded with the locale (GBK) and a
            # UTF-8 tool printing a Chinese temp path crashed the test. Name the encoding or say
            # how undecodable bytes are handled.
            text = any(
                isinstance(keywords.get(k), ast.Constant) and keywords[k].value is True
                for k in ("text", "universal_newlines")
            )
            if text and "encoding" not in keywords and "errors" not in keywords:
                yield node.lineno, f"{name}(text=True) without encoding/errors"
            continue
        encoding = next((kw.value for kw in node.keywords if kw.arg == "encoding"), None)
        # encoding=None is the locale default again, so it counts as unspecified.
        if encoding is not None and not (
            isinstance(encoding, ast.Constant) and encoding.value is None
        ):
            continue
        if name == "read_text" and not node.args:
            yield node.lineno, "read_text()"
        elif name == "write_text" and len(node.args) == 1:
            yield node.lineno, "write_text()"
        elif name == "open":
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in NOT_TEXT_FILES
            ):
                continue
            mode = None
            if isinstance(func, ast.Attribute):
                if node.args and isinstance(node.args[0], ast.Constant):
                    mode = node.args[0].value
            elif len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = keyword.value.value
            if isinstance(mode or "r", str) and "b" not in (mode or "r"):
                yield node.lineno, f"open({mode or 'r'!r})"


def test_every_text_read_and_write_names_its_encoding():
    findings = []
    for folder in ("src", "tests", "scripts", "docs/manuscript"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" in path.parts or path.relative_to(ROOT).as_posix() in HASH_BOUND:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            findings += [
                f"{path.relative_to(ROOT)}:{line}: {call}" for line, call in _missing_encoding(tree)
            ]
    assert not findings, "text I/O without encoding=:\n" + "\n".join(findings)


def test_the_scan_catches_the_windows_failure_pattern():
    """Control: the four W13-01 call shapes must be reported, binary and archive calls not."""
    sample = ast.parse(
        "html = (web / 'index.html').read_text()\n"
        "out.write_text(text)\n"
        "with open(path) as handle: pass\n"
        "with path.open('w') as handle: pass\n"
        "with open(path, 'rb') as handle: pass\n"
        "with tarfile.open(path, 'r:gz') as archive: pass\n"
        "fd = os.open(path, flags)\n"
        "text = path.read_text(encoding=None)\n"
        "text = path.read_text(encoding='utf-8')\n"
        "done = subprocess.run(command, capture_output=True, text=True)\n"
        "done = subprocess.run(command, capture_output=True, text=True, errors='replace')\n"
        "done = subprocess.run(command, capture_output=True)\n"
    )
    assert sorted(line for line, _ in _missing_encoding(sample)) == [1, 2, 3, 4, 8, 10]
