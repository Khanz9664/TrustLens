import os
import sys

from trustlens._version import __version__

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "TrustLens"
copyright = "2026, Shahid Ul Islam"
author = "Shahid Ul Islam"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "nbsphinx",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "html_image"]
myst_fence_as_directive = ["mermaid"]

source_suffix = {
    ".md": "markdown",
}

autosummary_generate = True
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
