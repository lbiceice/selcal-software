"""Sphinx configuration for the SelCal public API reference."""

from importlib.metadata import version as distribution_version

project = "SelCal"
author = "SelCal developers"
release = distribution_version("selcal")

needs_sphinx = "8.2"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_preserve_defaults = True
autodoc_typehints = "description"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
