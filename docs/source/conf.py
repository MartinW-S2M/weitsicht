from __future__ import annotations

import os
import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive

# This is a patch that TypeAliasForwardref is not shown in typehints
# Have no idea why this is but like this it works
from sphinx.util import inspect

inspect.TypeAliasForwardRef.__repr__ = lambda self: self.name
inspect.TypeAliasForwardRef.__hash__ = lambda self: hash(self.name)

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../examples"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "weitsicht"
copyright = "2026, Martin Wieser"
author = "Martin Wieser"
version = "0.0.2"
release = version

# add_module_names = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_design",
    # "sphinx_github_changelog",
    # "autoapi.extension",
]

# -- Optional CI-only extensions --------------------------------------------
ENABLE_CHANGELOG = bool(os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN"))
if ENABLE_CHANGELOG:
    try:
        extensions.append("sphinx_github_changelog")
        print("[conf.py] sphinx_github_changelog enabled (CI mode)")

    except Exception as e:
        print(f"[conf.py] Could not enable sphinx_github_changelog: {e}")
        ENABLE_CHANGELOG = False
else:
    print("[conf.py] sphinx_github_changelog disabled (local build)")


todo_include_todos = True

autodoc_default_options = {
    "inherited-members": False,
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "private-members": True,
    "exclude-members": "__weakref__",
    "inherit_docstrings": True,
}

html_scaled_image_link = False
add_module_names = True
autodoc_typehints = "both"
autodoc_preserve_defaults = True
python_use_unqualified_type_names = True
always_use_bars_union = True
typehints_fully_qualified = False  # Display class names without full module paths
always_document_param_types = True  # Add type info for undocumented parameters
typehints_document_rtype = True

# autoapi_dirs = ["../../src"]

autodoc_type_aliases = {
    "Vector2D": "Vector2D",
    "Vector3D": "Vector3D",
    "Array3x3": "Array3x3",
    "ArrayNx2": "ArrayNx2",
    "ArrayNx3": "ArrayNx3",
    "MaskN_": "MaskN_",
    "ArrayN_": "ArrayN_",
    #'tuple[ArrayNx2, MaskN_] | None': 'tuple[ArrayNx2, MaskN_] | None'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments styles for light/dark themes to prevent theme override flicker
# pygments_style = "friendly"
# pygments_dark_style = "friendly"
# pygments_style = "sphinx"
# pygments_dark_style = "sphinx"

source_suffix = {".rst": "restructuredtext"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static", "../../logos"]
DOCS_DIR = Path(__file__).parent
html_static_path = ["_static", str((DOCS_DIR.parent.parent / "logos").resolve())]


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes
# shutil.copy("../../logos/weitsicht.svg", "_static/weitsicht.svg")
html_theme = "pydata_sphinx_theme"
html_logo = str((DOCS_DIR.parent.parent / "logos" / "weitsicht.svg").resolve())
html_favicon = str((DOCS_DIR.parent.parent / "logos" / "weitsicht.svg").resolve())


html_css_files = [
    "css/custom.css",
]

# version-switcher details
version_json = "https://weitsicht.readthedocs.io/en/latest/_static/version_switcher.json"


html_theme_options = {
    "collapse_navigation": False,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "header_links_before_dropdown": 10,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MartinW-S2M/weitsicht",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "WISDAMapp",
            "url": "https://github.com/WISDAMapp/WISDAM",
            "icon": "_static/WISDAM_Hero_Logo_Black.svg",
            "type": "local",
        },
    ],
    "switcher": {"json_url": version_json, "version_match": "latest"},
}


class DummyChangelog(Directive):
    has_content = False
    option_spec = {
        "github": str,
        "pypi": str,
        "changelog-url": str,
    }

    def run(self):
        github = self.options.get("github", "https://github.com/MartinW-S2M/weitsicht/releases/")
        pypi = self.options.get("pypi", "https://pypi.org/project/weitsicht/")
        nodes_list = [
            nodes.paragraph(text="Changelog generation is disabled for this build."),
            nodes.paragraph("", "", nodes.reference(text=github, refuri=github)),
            nodes.paragraph("", "", nodes.reference(text=pypi, refuri=pypi)),
        ]
        return nodes_list


def setup(app):
    if not ENABLE_CHANGELOG:
        app.add_directive("changelog", DummyChangelog)
