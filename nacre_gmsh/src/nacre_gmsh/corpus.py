"""Analytic primitives whose baked ``SurfaceInput`` is committed as golden data.

Each case has a closed-form curvature field, so the golden ``.npz`` files these
build are validated against analytic values rather than against a previous run
of this code. Committing them is what lets the MPL-2.0 core test its full
contract with Gmsh absent.

The cases deliberately include the same sphere twice, once with the fluid
outside and once with the fluid inside, because that pair is what pins down the
curvature sign convention.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import gmsh
from nacre.contract import SurfaceInput, write_surface_input

from nacre_gmsh.bake import BakeSettings, bake_current_model

#: Sphere radius shared by the convex and concave sphere cases.
SPHERE_RADIUS = 2.0
#: Cylinder radius shared by the external-flow and pipe cases.
CYLINDER_RADIUS = 1.5
CYLINDER_LENGTH = 4.0
#: Torus major and minor radii.
TORUS_MAJOR = 3.0
TORUS_MINOR = 1.0
#: Half-width of the far-field box used by the external-flow cases.
FARFIELD_HALF_WIDTH = 6.0
#: Edge length of the box-shaped fluid domain used by the plane case.
BOX_EDGE = 2.0


@dataclass(frozen=True)
class CorpusCase:
    """One analytic geometry and the settings used to bake it."""

    name: str
    description: str
    build: Callable[[], None]
    settings: BakeSettings


def _name_surfaces(body: str, farfield: str | None = None) -> None:
    """Tag curved bodies and the far field so patch names survive the bake.

    Physical groups are the explicit-intent path for patch naming; named STEP
    faces reach the same resolver through ``gmsh.model.getEntityName``.
    """

    curved: list[int] = []
    planar: list[int] = []
    for _, tag in gmsh.model.getEntities(2):
        target = planar if gmsh.model.getType(2, tag) == "Plane" else curved
        target.append(tag)
    if curved:
        gmsh.model.addPhysicalGroup(2, curved, name=body)
    if planar:
        gmsh.model.addPhysicalGroup(2, planar, name=farfield or body)


def _build_box() -> None:
    """Fluid inside a cube: six planes, twelve feature edges, eight corners."""

    gmsh.model.occ.addBox(0.0, 0.0, 0.0, BOX_EDGE, BOX_EDGE, BOX_EDGE)
    gmsh.model.occ.synchronize()
    _name_surfaces("walls")


def _build_sphere_external() -> None:
    """Fluid outside a solid sphere: the convex reference case."""

    half = FARFIELD_HALF_WIDTH
    box = gmsh.model.occ.addBox(-half, -half, -half, 2 * half, 2 * half, 2 * half)
    sphere = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, SPHERE_RADIUS)
    gmsh.model.occ.cut([(3, box)], [(3, sphere)])
    gmsh.model.occ.synchronize()
    _name_surfaces("sphere", farfield="farfield")


def _build_sphere_chamber() -> None:
    """Fluid inside a spherical chamber: the concave reference case."""

    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, SPHERE_RADIUS)
    gmsh.model.occ.synchronize()
    _name_surfaces("chamber")


def _build_cylinder_external() -> None:
    """Fluid outside a solid cylinder: ``k1 = 1/R`` with ``k2 = 0``."""

    half = FARFIELD_HALF_WIDTH
    box = gmsh.model.occ.addBox(-half, -half, -half, 2 * half, 2 * half, 2 * half)
    cylinder = gmsh.model.occ.addCylinder(
        0.0, 0.0, -CYLINDER_LENGTH / 2.0, 0.0, 0.0, CYLINDER_LENGTH, CYLINDER_RADIUS
    )
    gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
    gmsh.model.occ.synchronize()
    _name_surfaces("cylinder", farfield="farfield")


def _build_cylinder_pipe() -> None:
    """Fluid inside a pipe: a concave cylinder, where layers converge."""

    gmsh.model.occ.addCylinder(
        0.0, 0.0, 0.0, 0.0, 0.0, CYLINDER_LENGTH, CYLINDER_RADIUS
    )
    gmsh.model.occ.synchronize()
    _name_surfaces("pipe", farfield="ends")


def _build_torus_external() -> None:
    """Fluid outside a solid torus: curvature that varies over the surface."""

    half = FARFIELD_HALF_WIDTH
    box = gmsh.model.occ.addBox(-half, -half, -half, 2 * half, 2 * half, 2 * half)
    torus = gmsh.model.occ.addTorus(0.0, 0.0, 0.0, TORUS_MAJOR, TORUS_MINOR)
    gmsh.model.occ.cut([(3, box)], [(3, torus)])
    gmsh.model.occ.synchronize()
    _name_surfaces("torus", farfield="farfield")


CORPUS_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        name="box",
        description="fluid inside a cube; plane anchor, k1 = k2 = 0",
        build=_build_box,
        settings=BakeSettings(size_max=0.7),
    ),
    CorpusCase(
        name="sphere-external",
        description=(
            "fluid outside a solid sphere; convex anchor k1 = k2 = 1/R, and the "
            "experiment that established the into-fluid normal convention"
        ),
        build=_build_sphere_external,
        settings=BakeSettings(size_max=3.0, size_min=0.5, size_from_curvature=12.0),
    ),
    CorpusCase(
        name="sphere-chamber",
        description="fluid inside a spherical chamber; concave anchor k1 = k2 = -1/R",
        build=_build_sphere_chamber,
        settings=BakeSettings(size_max=0.6),
    ),
    CorpusCase(
        name="cylinder-external",
        description="fluid outside a solid cylinder; k1 = 1/R, k2 = 0",
        build=_build_cylinder_external,
        settings=BakeSettings(size_max=3.0, size_min=0.4, size_from_curvature=12.0),
    ),
    CorpusCase(
        name="cylinder-pipe",
        description="fluid inside a pipe; concave cylinder, k1 = 0, k2 = -1/R",
        build=_build_cylinder_pipe,
        settings=BakeSettings(size_max=0.6),
    ),
    CorpusCase(
        name="torus-external",
        description=(
            "fluid outside a solid torus; k1 = 1/r everywhere while k2 varies "
            "from +1/(R+r) on the outer equator to -1/(R-r) in the throat"
        ),
        build=_build_torus_external,
        settings=BakeSettings(size_max=3.0, size_min=0.25, size_from_curvature=16.0),
    ),
)

CASES_BY_NAME = {case.name: case for case in CORPUS_CASES}


def bake_case(case: CorpusCase) -> SurfaceInput:
    """Build and bake one corpus case in a fresh Gmsh model.

    Requires an active Gmsh session; see ``nacre_gmsh.gmsh_session``.
    """

    gmsh.model.add(case.name)
    try:
        case.build()
        return bake_current_model(case.settings)
    finally:
        gmsh.model.remove()


def bake_corpus() -> Iterator[tuple[CorpusCase, SurfaceInput]]:
    """Bake every corpus case in order."""

    for case in CORPUS_CASES:
        yield case, bake_case(case)


def write_goldens(directory: str | Path) -> list[Path]:
    """Bake the corpus and write one ``.npz`` golden file per case."""

    destination = Path(directory)
    written: list[Path] = []
    for case, surface in bake_corpus():
        written.append(write_surface_input(surface, destination / f"{case.name}.npz"))
    return written


__all__ = [
    "BOX_EDGE",
    "CASES_BY_NAME",
    "CORPUS_CASES",
    "CYLINDER_LENGTH",
    "CYLINDER_RADIUS",
    "CorpusCase",
    "FARFIELD_HALF_WIDTH",
    "SPHERE_RADIUS",
    "TORUS_MAJOR",
    "TORUS_MINOR",
    "bake_case",
    "bake_corpus",
    "write_goldens",
]
