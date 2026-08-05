"""Decide which side of each CAD face the fluid is on.

This is the whole of convention B in ``nacre.contract``: ``vert_normal`` points
into the fluid. Gmsh will not tell you that directly. ``getNormal`` returns the
normal of the underlying parametric surface and ignores how the face is oriented
inside the solid, and ``getBoundary(oriented=True)`` is not an inside/outside
flag either. The side has to be established geometrically, which is what this
module does.
"""

from __future__ import annotations

import gmsh
import numpy as np
from numpy.typing import NDArray

from nacre_gmsh.query import BakeError, project_onto_surface, surface_normals

#: Probe offsets tried in order, relative to the model's reference length.
#:
#: Too large an offset can cross into another region of the model; too small a
#: one falls inside OCC's own point-classification tolerance, where both sides
#: answer the same. The ladder starts at the value that works for every corpus
#: case and widens, then narrows, rather than committing to one guess.
EPS_LADDER = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-6, 1.0e-7)


def fluid_normal_sign(
    tag: int,
    volumes: list[int],
    probe_xyz: NDArray[np.float64],
    ref_length: float,
    samples: int,
) -> int:
    """Return ``+1`` if the parametric normal of ``tag`` enters the fluid.

    ``volumes`` are the model's 3D entities, which by contract *are* the fluid
    domain. Each probe point is projected onto the surface, stepped off it both
    ways, and classified with ``gmsh.model.isInside``. Exactly one side must
    come back inside the fluid, and every sample must agree.

    Projecting first is not optional. Probing from a raw triangle centroid
    fails on a coarsely meshed convex surface, where the chord sag exceeds any
    usable offset and both steps land on the solid side.
    """

    stride = max(1, len(probe_xyz) // samples)
    base, uv = project_onto_surface(tag, probe_xyz[::stride])
    normals = surface_normals(tag, uv)

    for relative_eps in EPS_LADDER:
        eps = relative_eps * ref_length
        votes: list[int] = []
        for point, normal in zip(base, normals):
            forward = _inside_fluid(volumes, point + eps * normal)
            backward = _inside_fluid(volumes, point - eps * normal)
            if forward == backward:
                continue  # This offset cannot separate the two sides.
            votes.append(1 if forward else -1)
        if not votes:
            continue
        if len(set(votes)) == 1:
            return votes[0]
        raise BakeError(
            f"surface {tag} disagrees with itself about which side the fluid is "
            f"on ({votes.count(1)} outward votes, {votes.count(-1)} inward) at "
            f"eps={eps:.17g}; the model is probably not a valid fluid domain"
        )

    raise BakeError(
        f"could not decide which side of surface {tag} the fluid is on: no probe "
        f"offset in {EPS_LADDER} separated the two sides"
    )


def _inside_fluid(volumes: list[int], point: NDArray[np.float64]) -> bool:
    coord = list(point)
    return any(gmsh.model.isInside(3, volume, coord) for volume in volumes)


def into_fluid_normals_at(
    surface_tags: NDArray[np.int32],
    points: NDArray[np.float64],
    sign_of: dict[int, int],
) -> NDArray[np.float64]:
    """Analytic into-fluid normals at arbitrary points, batched per CAD face."""

    out = np.empty_like(points)
    for tag in np.unique(surface_tags):
        selected = surface_tags == tag
        _, uv = project_onto_surface(int(tag), points[selected])
        out[selected] = sign_of[int(tag)] * surface_normals(int(tag), uv)
    return out


__all__ = ["EPS_LADDER", "fluid_normal_sign", "into_fluid_normals_at"]
