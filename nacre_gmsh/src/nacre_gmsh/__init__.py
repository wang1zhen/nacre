"""Gmsh frontend for nacre: CAD in, baked ``SurfaceInput`` out.

This distribution is GPL-3.0-or-later because Gmsh is. It is the only place in
the project allowed to import ``gmsh``; the MPL-2.0 ``nacre`` core consumes the
``.npz`` files produced here and never links a CAD kernel. A CI job installs the
core alone and asserts that ``import gmsh`` fails, so the boundary is checked by
machine rather than by discipline.
"""

from nacre_gmsh.bake import (
    BakeSettings,
    bake_current_model,
    bake_step,
    bake_step_to_npz,
)
from nacre_gmsh.corpus import CORPUS_CASES, CorpusCase, bake_case, write_goldens
from nacre_gmsh.orient import fluid_normal_sign
from nacre_gmsh.query import BakeError
from nacre_gmsh.session import gmsh_session

__all__ = [
    "CORPUS_CASES",
    "BakeError",
    "BakeSettings",
    "CorpusCase",
    "bake_case",
    "bake_current_model",
    "bake_step",
    "bake_step_to_npz",
    "fluid_normal_sign",
    "gmsh_session",
    "write_goldens",
]
