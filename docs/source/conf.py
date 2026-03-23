"""Sphinx configuration for inspect-mlflow docs."""

project = "inspect-mlflow"
copyright = "2026, Debu Sinha"
author = "Debu Sinha"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mlflow": ("https://mlflow.org/docs/latest", None),
}
