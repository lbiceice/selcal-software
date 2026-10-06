"""UTF-8 packaged resources must not depend on a Windows user's code page."""

from __future__ import annotations

from pathlib import Path

import pytest

from selcal import ui


def test_example_route_decodes_resources_as_utf8_under_cp936(monkeypatch):
    values = {"series.csv": "x,y\n1,2\n# 温度°\n", "pearson.json": '{"label":"温度°"}'}

    class Resource:
        def __init__(self, name=""):
            self.name = name

        def joinpath(self, *parts):
            return Resource(parts[-1])

        def read_text(self, encoding=None, errors="strict"):
            return values[self.name].encode("utf-8").decode(encoding or "cp936", errors)

    monkeypatch.setattr(ui, "files", lambda package: Resource())
    handler = object.__new__(ui._Handler)
    handler.command, handler.path = "GET", "/api/example"
    handler._authorize = lambda: None
    handler._length = lambda cap: 0
    replies = []
    handler._json = lambda status, value: replies.append((status, value))
    handler._dispatch()
    assert replies == [(200, {"input_text": values["series.csv"],
                              "config_text": values["pearson.json"]})]


@pytest.mark.parametrize("name", ["test_ui_assets.py", "test_ui_artifacts.py"])
def test_utf8_resource_test_readers_have_an_explicit_encoding(name):
    """Keep the actual Windows regression tests runnable without -X utf8."""
    import ast

    path = Path(__file__).with_name(name)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    implicit = [node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "read_text" and not node.args
                and not any(kw.arg == "encoding" for kw in node.keywords)]
    assert implicit == [], f"{name}: locale-dependent read_text at {implicit}"
