"""Closed-form expectations for the baked ``SurfaceInput`` golden corpus.

Everything here is written from the analytic geometry, independently of the
Gmsh queries that produced the golden files. That independence is the point:
``tests/test_surface_input.py`` uses it to validate the committed ``.npz``
without Gmsh installed, and ``tests/test_gmsh_bake.py`` reuses it to validate a
live bake. Neither test may fall back to comparing a bake against a previous
bake, which would let a wrong convention validate itself.

Radii and box extents mirror ``nacre_gmsh.corpus``; they are restated rather
than imported because the core test suite must not import the Gmsh
distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nacre.contract import SurfaceInput, feature_vertices
from numpy.typing import NDArray

GOLDEN_DIR = Path(__file__).parent / "goldens"

SPHERE_RADIUS = 2.0
CYLINDER_RADIUS = 1.5
CYLINDER_LENGTH = 4.0
TORUS_MAJOR = 3.0
TORUS_MINOR = 1.0
FARFIELD_HALF_WIDTH = 6.0
BOX_EDGE = 2.0

#: Curvature agreement demanded of the analytic anchors. Gmsh's BREP queries
#: are exact for these primitives, so the measured error is at machine
#: precision; this leaves nine orders of headroom over the 1% the roadmap asks
#: for while still failing on any confusion of k1 with k2, of principal with
#: mean curvature, or of convex with concave.
CURVATURE_RTOL = 1.0e-9
CURVATURE_ATOL = 1.0e-12

#: Relative offset used when testing that a normal points into the fluid.
FLUID_PROBE_EPS = 1.0e-4


def _plane_curvature(points: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.zeros((len(points), 2), dtype=np.float64)


def _sphere_curvature(
    points: NDArray[np.float64], radius: float, sign: float
) -> NDArray[np.float64]:
    return np.full((len(points), 2), sign / radius, dtype=np.float64)


def _cylinder_curvature(
    points: NDArray[np.float64], radius: float, sign: float
) -> NDArray[np.float64]:
    """``1/R`` around the axis and ``0`` along it, ordered ``k1 >= k2``."""

    pair = np.stack(
        [
            np.full(len(points), sign / radius),
            np.zeros(len(points)),
        ],
        axis=1,
    )
    return np.stack([pair.max(axis=1), pair.min(axis=1)], axis=1)


def _torus_curvature(
    points: NDArray[np.float64], major: float, minor: float
) -> NDArray[np.float64]:
    """Analytic principal curvatures of a solid torus with fluid outside.

    In the meridian plane at angle ``t`` from the outer equator the meridian
    circle contributes ``1/r`` and the hoop direction contributes
    ``cos(t) / (R + r cos(t))``. The hoop term runs from ``+1/(R+r)`` on the
    outer equator down to ``-1/(R-r)`` in the inner throat, which is what makes
    this case reject a producer that returns a constant.
    """

    radial = np.hypot(points[:, 0], points[:, 1])
    cos_t = (radial - major) / minor
    hoop = cos_t / (major + minor * cos_t)
    meridian = np.full(len(points), 1.0 / minor)
    pair = np.stack([meridian, hoop], axis=1)
    return np.stack([pair.max(axis=1), pair.min(axis=1)], axis=1)


def _inside_box(points: NDArray[np.float64], half: float) -> NDArray[np.bool_]:
    return np.all(np.abs(points) < half, axis=1)


def _fluid_box(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.all((points > 0.0) & (points < BOX_EDGE), axis=1)


def _fluid_sphere_external(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    outside_sphere = np.linalg.norm(points, axis=1) > SPHERE_RADIUS
    return outside_sphere & _inside_box(points, FARFIELD_HALF_WIDTH)


def _fluid_sphere_chamber(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.linalg.norm(points, axis=1) < SPHERE_RADIUS


def _fluid_cylinder_external(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    radial = np.hypot(points[:, 0], points[:, 1])
    outside_cylinder = (radial > CYLINDER_RADIUS) | (
        np.abs(points[:, 2]) > CYLINDER_LENGTH / 2.0
    )
    return outside_cylinder & _inside_box(points, FARFIELD_HALF_WIDTH)


def _fluid_cylinder_pipe(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    radial = np.hypot(points[:, 0], points[:, 1])
    return (
        (radial < CYLINDER_RADIUS)
        & (points[:, 2] > 0.0)
        & (points[:, 2] < CYLINDER_LENGTH)
    )


def _fluid_torus_external(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    radial = np.hypot(points[:, 0], points[:, 1])
    tube = np.hypot(radial - TORUS_MAJOR, points[:, 2])
    return (tube > TORUS_MINOR) & _inside_box(points, FARFIELD_HALF_WIDTH)


@dataclass(frozen=True)
class AnchorCase:
    """One golden file and the closed-form truth it must reproduce.

    Attributes:
        name: Golden file stem under ``tests/goldens``.
        curvature: Expected ``(k1, k2)`` per point, per patch name.
        in_fluid: Analytic membership test for the fluid domain, used to check
            that ``vert_normal`` points into it.
        expected_corners: Number of ``feat_corners`` entries.
        has_feat_edges: Whether the case has any sharp CAD edge at all.
    """

    name: str
    curvature: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    in_fluid: Callable[[NDArray[np.float64]], NDArray[np.bool_]]
    expected_corners: int
    has_feat_edges: bool


ANCHOR_CASES: tuple[AnchorCase, ...] = (
    AnchorCase(
        name="box",
        curvature={"walls": _plane_curvature},
        in_fluid=_fluid_box,
        expected_corners=8,
        has_feat_edges=True,
    ),
    AnchorCase(
        name="sphere-external",
        curvature={
            "sphere": lambda p: _sphere_curvature(p, SPHERE_RADIUS, +1.0),
            "farfield": _plane_curvature,
        },
        in_fluid=_fluid_sphere_external,
        expected_corners=8,
        has_feat_edges=True,
    ),
    AnchorCase(
        name="sphere-chamber",
        curvature={
            "chamber": lambda p: _sphere_curvature(p, SPHERE_RADIUS, -1.0),
        },
        in_fluid=_fluid_sphere_chamber,
        expected_corners=0,
        has_feat_edges=False,
    ),
    AnchorCase(
        name="cylinder-external",
        curvature={
            "cylinder": lambda p: _cylinder_curvature(p, CYLINDER_RADIUS, +1.0),
            "farfield": _plane_curvature,
        },
        in_fluid=_fluid_cylinder_external,
        expected_corners=8,
        has_feat_edges=True,
    ),
    AnchorCase(
        name="cylinder-pipe",
        curvature={
            "pipe": lambda p: _cylinder_curvature(p, CYLINDER_RADIUS, -1.0),
            "ends": _plane_curvature,
        },
        in_fluid=_fluid_cylinder_pipe,
        expected_corners=0,
        has_feat_edges=True,
    ),
    AnchorCase(
        name="torus-external",
        curvature={
            "torus": lambda p: _torus_curvature(p, TORUS_MAJOR, TORUS_MINOR),
            "farfield": _plane_curvature,
        },
        in_fluid=_fluid_torus_external,
        expected_corners=8,
        has_feat_edges=True,
    ),
)

ANCHORS_BY_NAME = {case.name: case for case in ANCHOR_CASES}


def patch_smooth_vertices(
    surface: SurfaceInput, patch_name: str
) -> NDArray[np.int32]:
    """Vertices that belong to one patch and to one CAD face.

    Feature vertices are excluded because the contract deliberately stores one
    of several valid curvature samples there, so no single closed-form value
    applies.
    """

    patch = surface.patch_names.index(patch_name)
    selected = surface.tri_patch == patch
    mine = np.unique(surface.tris[selected])
    if np.all(selected):
        candidates = mine
    else:
        candidates = np.setdiff1d(mine, np.unique(surface.tris[~selected]))
    smooth = np.setdiff1d(candidates, feature_vertices(surface))
    if not len(smooth):
        raise AssertionError(f"patch {patch_name!r} has no smooth interior vertex")
    return smooth


def torus_hoop_curvature(cos_t: float) -> float:
    """Analytic hoop curvature of the golden torus at meridian angle ``t``."""

    return cos_t / (TORUS_MAJOR + TORUS_MINOR * cos_t)


__all__ = [
    "ANCHORS_BY_NAME",
    "ANCHOR_CASES",
    "BOX_EDGE",
    "CURVATURE_ATOL",
    "CURVATURE_RTOL",
    "CYLINDER_LENGTH",
    "CYLINDER_RADIUS",
    "FARFIELD_HALF_WIDTH",
    "FLUID_PROBE_EPS",
    "GOLDEN_DIR",
    "SPHERE_RADIUS",
    "TORUS_MAJOR",
    "TORUS_MINOR",
    "AnchorCase",
    "patch_smooth_vertices",
    "torus_hoop_curvature",
]
