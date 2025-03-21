import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Add project root to Python path
sys.path.insert(0, os.path.abspath('../utils')) # Add utils to the path
sys.path.insert(0, os.path.abspath('../01_Bigram_Language_Model/src'))


# -- Project information -----------------------------------------------------

project = 'LLM Course'
copyright = '2025, Nirajan Dhakal'
author = 'Nirajan Dhakal'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',     # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',    # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',   # Add links to source code
    'sphinx.ext.mathjax',    # Render LaTeX math equations
    'myst_parser',           # Enable Markdown support
    'sphinx.ext.intersphinx', # Link to other Sphinx documentation (e.g., PyTorch)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme
html_static_path = ['_static']

# -- Options for myst-parser ----------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Options for intersphinx -------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for autodoc --------------------------------------------------

autodoc_default_options = {
    'members': True,         # Document members (functions, classes, methods)
    'member-order': 'bysource',  # Order members by their source code order
    'special-members': '__init__',  # Include __init__ methods
    'undoc-members': True,    # Include members that don't have docstrings
    'exclude-members': '__weakref__',  # Exclude __weakref__ attribute
    'show-inheritance': True, #Show inheritance
}

autoclass_content = 'both'  # Include both class docstring and __init__ docstring