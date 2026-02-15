# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

import inspect
import types

# ----------------------------------------------------------------------
# 1. Setup Exclusions
# ----------------------------------------------------------------------
try:
    import torch.nn as nn
except ImportError:
    nn = types.SimpleNamespace(Module=type("DummyModule", (), {}))

try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = type("DummyBaseEstimator", (), {})

EXCLUDE_BASES = (nn.Module, BaseEstimator)

# ----------------------------------------------------------------------
# 2. Setup Inclusions (Mixins)
# ----------------------------------------------------------------------
# Replace this with the actual module import
import jmstate.mixins as my_mixins 

MIXIN_CLASSES = tuple(
    cls for name, cls in inspect.getmembers(my_mixins, inspect.isclass)
)

# ----------------------------------------------------------------------
# 3. The Handler
# ----------------------------------------------------------------------
def skip_member_handler(app, what, name, obj, skip, options):
    """
    Control whether a member is skipped.
    """
    # 'obj' is the member itself. 
    # To check where it's defined, we often need the class it belongs to.
    # Sphinx passes the class being documented in 'options' sometimes, 
    # but relying on simple inclusion/exclusion lists is safer.

    # 1. PRIORITY: Always include members from your Mixins
    # We check if this member name defines a method/attr in one of your mixins
    for mixin_cls in MIXIN_CLASSES:
        if name in mixin_cls.__dict__:
             return False

    # 2. EXCLUSION: Skip members that are defined in the Excluded Bases
    # We use 'dir()' or 'hasattr()' carefully. 
    for base in EXCLUDE_BASES:
        # Check if the base has this member
        if hasattr(base, name):
            # CRITICAL CHECK:
            # We must ensure we don't skip it if the user RE-DEFINED it 
            # in the child class.
            
            # However, inside this handler, obtaining the "child class" can be tricky.
            # A cleaner way usually relies on the fact that if it's in EXCLUDE_BASES,
            # we want to skip it UNLESS it matches criteria above.
            
            # If you want to keep strictly to "Skip if it comes from nn.Module",
            # you simply return True here.
            # BUT, be aware this hides 'forward' if you don't explicitly document it.
            return True

    # 3. Default behavior
    return None

def setup(app):
    app.connect("autodoc-skip-member", skip_member_handler)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jmstate"
release = ""
version = ""
copyright = "2026, Félix Laplante"
author = "Félix Laplante"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True
autosummary_generate = True
add_module_names = False
napoleon_use_ivar = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]



