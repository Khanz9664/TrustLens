import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(".."))

from trustlens._version import __version__

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

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
]

myst_fence_as_directive = ["mermaid"]

source_suffix = {
    ".md": "markdown",
}

autosummary_generate = True
autodoc_member_order = "bysource"

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

# Static assets
html_static_path = ["_static"]

# GitHub Pages deployment URL
html_baseurl = "https://khanz9664.github.io/trustlensdocs/"

# Prevent GitHub Pages path issues
html_theme_options = {}

# Ensure clean relative asset loading
html_css_files = []
html_js_files = []

# GitHub integration (optional but useful)
html_context = {
    "display_github": True,
    "github_user": "Khanz9664",
    "github_repo": "TrustLens",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
