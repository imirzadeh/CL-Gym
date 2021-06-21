# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# print(f"{os.path.join(os.path.abspath('..'), 'cl_gym')}")
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'CL-Gym'
copyright = '2021, CL-Gym Authors'
author = 'Iman Mirzadeh'

# The full version, including alpha/beta/rc tags
release = '1.0.0-beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx_rtd_theme',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        'sphinx.ext.intersphinx',
]
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
# html_css_files = [
#     'css/custom.css',
# ]

# html_style = 'css/custom.css'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    '.AppleDouble', '.LSOverride', '*.so',
                    'Session.vim', ]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
