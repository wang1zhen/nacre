"""Deterministic Gmsh session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import gmsh


@contextmanager
def gmsh_session(*, verbose: bool = False) -> Iterator[None]:
    """Initialize Gmsh with reproducible settings and always finalize.

    Baked ``.npz`` files are committed golden data, so the session pins the
    meshing algorithm and forces single-threaded meshing: Gmsh's parallel
    front advance is not deterministic.
    """

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
        # Import STEP/IGES product and face labels so named CAD faces can
        # become patch names.
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        yield
    finally:
        gmsh.finalize()


__all__ = ["gmsh_session"]
