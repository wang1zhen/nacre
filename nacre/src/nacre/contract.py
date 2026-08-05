"""Frozen geometry contract between a CAD frontend and the nacre core.

``SurfaceInput`` is the only channel through which BREP information reaches the
core. The core never calls a CAD kernel: a frontend bakes geometry once into
these arrays, the arrays are persisted as ``.npz``, and every downstream module
consumes them. ``nacre_gmsh`` is the first producer; it is a separate
GPL-3.0-or-later distribution precisely so that this MPL-2.0 module stays free
of Gmsh.

Layout is structure-of-arrays. ``points`` and the ``vert_*`` arrays are indexed
by vertex; ``tris`` and ``tri_patch`` are indexed by triangle. Floating-point
arrays are ``float64`` and index arrays are ``int32``, matching ``PolyMeshIR``.

Sign conventions
================

Two conventions decide whether boundary layers grow into the fluid or into the
solid, and whether a wall reads as convex or concave. Neither is recoverable
from a Gmsh query alone, both fail silently when wrong, and both were therefore
fixed by experiment against Gmsh 4.15.2 rather than by assumption. The
experiments live in ``tests/test_gmsh_bake.py`` and run on every commit that
has Gmsh available.

Normal orientation: ``vert_normal`` points INTO the fluid
---------------------------------------------------------

The baked model's 3D entities *are* the fluid domain. ``vert_normal`` is a unit
vector pointing from the wall into that fluid domain, and ``tris`` are wound so
that the right-hand-rule triangle normal points the same way (positive dot
product with the mean of its three vertex normals). A boundary layer is
extruded along ``+vert_normal``.

*Establishing experiment* -- external flow, a sphere of radius 2 centred in a
box spanning ``[-5, 5]^3``, fluid domain ``box \\ sphere``:

  ``gmsh.model.getNormal`` returns the normal of the *underlying parametric
  surface* and ignores how the face is oriented inside the solid. This was
  proven by building a sphere alone and then cutting the same sphere out of a
  block: the reported normal at ``(2, 0, 0)`` stayed ``(+1, 0, 0)`` in both
  models even though the material swapped sides. ``getBoundary(oriented=True)``
  is no help either -- on a box its signs multiply the parametric normals into
  ``+x, +y, +z`` for all six faces, so the sign is not an inside/outside flag.

  Orientation is therefore resolved geometrically. Stepping a distance ``eps``
  off a triangle centroid along ``+n`` and along ``-n`` and calling
  ``gmsh.model.isInside(3, fluid_volume, ...)`` gave, for every sampled
  centroid, exactly one side inside the fluid: ``+n`` for the sphere and ``-n``
  for all six box planes. The sphere's normals and the box's normals point in
  opposite absolute directions, and both point into the fluid, which is the
  assertion the convention needs.

Curvature sign: ``k > 0`` is convex toward the fluid
----------------------------------------------------

``vert_kappa``, sorted descending, holds the principal curvatures measured
with respect to
``vert_normal``, signed so that curvature is **positive where the centre of
curvature lies on the solid side** -- that is, where the wall bulges into the
fluid -- and **negative where it lies on the fluid side**, where the wall wraps
around the fluid and boundary layers converge. Concave regions are exactly the
ones where layers collide, so the distinction cannot be lost.

*Establishing experiment* -- a solid sphere of radius ``R`` with fluid outside,
against the same sphere used as a chamber with fluid inside:

  ``gmsh.model.getPrincipalCurvatures`` returned ``-1/R`` for *both*, and also
  for a sphere cut out of a block. The raw Gmsh value therefore cannot
  distinguish convex from concave: it is measured against the parametric
  normal, which does not move when the material does.

  The sign is recovered by folding in the orientation flag established above.
  With ``n_p`` the parametric normal, ``s = +/-1`` such that
  ``vert_normal = s * n_p``, and ``(kmax_g, kmin_g)`` the Gmsh pair, this
  contract requires ``{-s * kmax_g, -s * kmin_g}`` sorted descending, with the
  principal directions carried along by the same reordering. Gmsh reports
  ``-1/R`` for a sphere whose parametric normal points away from the centre, so
  the leading minus sign is what makes an isolated convex body positive.

  Verified consequences: a solid sphere of radius ``R`` with fluid outside has
  ``s = +1`` and gives ``k1 = k2 = +1/R``; the same sphere as a fluid-filled
  chamber has ``s = -1`` and gives ``k1 = k2 = -1/R``. Opposite signs, as
  required.

  Note that a spherical cavity carved out of a block is *not* the concave case.
  Its fluid domain is ``box \\ sphere``, so an observer in the fluid still sees
  the wall bulge toward them and the sign stays positive. Concavity depends on
  which side the fluid is on, not on how the CAD model was built; the
  sign-opposite case is a fluid-filled chamber.

Analytic anchors
================

Every anchor below is asserted by the test suite against baked ``.npz`` golden
files, so the core suite validates the convention without Gmsh installed:

===========================  ==============================================
plane                        ``k1 = k2 = 0``
sphere radius ``R``          ``k1 = k2 = 1/R``
cylinder radius ``R``        ``k1 = 1/R``, ``k2 = 0``
torus ``(R, r)``             ``k1 = 1/r`` everywhere and
                             ``k2 = cos(t)/(R + r cos(t))``, hence
                             ``+1/(R+r)`` on the outer equator and
                             ``-1/(R-r)`` in the inner throat
spherical chamber ``R``      ``k1 = k2 = -1/R``, opposite the solid sphere
===========================  ==============================================

The cylinder anchor catches confusing ``k1`` with ``k2`` and catches returning
mean or Gaussian curvature instead of principal curvatures. The torus anchor
catches returning a constant, which both the sphere and the cylinder would
otherwise accept.

Curvature at feature vertices
=============================

A vertex on a feature edge or corner lies on two or more distinct surfaces with
genuinely different curvature. Such a vertex stores the sample from the
incident surface with the largest ``max(|k1|, |k2|)``, which is the
conservative choice for curvature-driven sizing. ``feature_vertices`` lists
these vertices so consumers know the value is one of several valid ones.

``vert_normal`` at such a vertex is the area-weighted average of the incident
faces' into-fluid normals, so it still enters the fluid, but it is not the
normal of any one face. The curvature frame therefore is not tangent to it; see
``nacre.check.check_surface_input``, which restricts the tangency invariant to
smooth vertices for exactly this reason.

Degenerate principal directions
===============================

At a parametrization singularity -- a sphere pole, for instance -- Gmsh returns
exact zero vectors for both principal directions while the curvatures stay
correct. The two rows of ``vert_kappa_dir`` are therefore either an orthonormal
tangent pair or both exactly zero. Consumers needing a tangent frame at such a
vertex must build one from ``vert_normal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Bumped only by an explicit decision to change the frozen layout.
CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SurfaceInput:
    """A baked, watertight triangulation of the fluid-domain boundary.

    This class is only a data container. Use ``nacre.check.check_surface_input``
    for validation and the module docstring for the sign conventions.

    Attributes:
        points: ``(n_verts, 3)`` vertex coordinates.
        tris: ``(n_tris, 3)`` vertex indices, wound so the right-hand-rule
            normal points into the fluid.
        tri_patch: ``(n_tris,)`` index into ``patch_names``/``patch_types``.
        patch_names: Patch names, carried through from named CAD faces where
            the source provides them.
        patch_types: OpenFOAM boundary type per patch, aligned with
            ``patch_names``.
        vert_normal: ``(n_verts, 3)`` unit normals pointing into the fluid.
        vert_kappa: ``(n_verts, 2)`` principal curvatures from the CAD BREP,
            sorted so ``vert_kappa[:, 0] >= vert_kappa[:, 1]``.
        vert_kappa_dir: ``(n_verts, 2, 3)`` principal directions, aligned with
            ``vert_kappa``. Both rows are exactly zero at a parametrization
            singularity; see the module docstring.
        feat_edges: ``(n_feat_edges, 2)`` vertex-index pairs naming the mesh
            edges that lie on a sharp CAD feature.
        feat_corners: ``(n_feat_corners,)`` vertex indices where feature edges
            meet at a genuine corner.
        ref_length: Geometry reference length, the bounding-box diagonal.
            Tolerances elsewhere in nacre are expressed relative to it.
    """

    points: NDArray[np.float64]
    tris: NDArray[np.int32]
    tri_patch: NDArray[np.int32]
    patch_names: tuple[str, ...]
    patch_types: tuple[str, ...]
    vert_normal: NDArray[np.float64]
    vert_kappa: NDArray[np.float64]
    vert_kappa_dir: NDArray[np.float64]
    feat_edges: NDArray[np.int32]
    feat_corners: NDArray[np.int32]
    ref_length: float

    @property
    def n_verts(self) -> int:
        return len(self.points)

    @property
    def n_tris(self) -> int:
        return len(self.tris)

    @property
    def n_patches(self) -> int:
        return len(self.patch_names)


def feature_vertices(surface: SurfaceInput) -> NDArray[np.int32]:
    """Vertices whose curvature sample is one of several valid ones.

    A vertex on a feature edge or corner lies on two or more CAD faces, so
    ``vert_kappa`` there is the single most conservative sample rather than a
    unique value. Consumers that must not mix faces should exclude these.
    """

    return np.union1d(
        surface.feat_edges.reshape(-1).astype(np.int32),
        surface.feat_corners.astype(np.int32),
    ).astype(np.int32)


def write_surface_input(surface: SurfaceInput, path: str | Path) -> Path:
    """Persist ``surface`` as a compressed ``.npz`` golden file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        contract_version=np.asarray(CONTRACT_VERSION, dtype=np.int32),
        points=surface.points,
        tris=surface.tris,
        tri_patch=surface.tri_patch,
        patch_names=np.asarray(surface.patch_names, dtype=np.str_),
        patch_types=np.asarray(surface.patch_types, dtype=np.str_),
        vert_normal=surface.vert_normal,
        vert_kappa=surface.vert_kappa,
        vert_kappa_dir=surface.vert_kappa_dir,
        feat_edges=surface.feat_edges,
        feat_corners=surface.feat_corners,
        ref_length=np.asarray(surface.ref_length, dtype=np.float64),
    )
    return destination if destination.suffix else destination.with_suffix(".npz")


def read_surface_input(path: str | Path) -> SurfaceInput:
    """Load a ``SurfaceInput`` written by :func:`write_surface_input`."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        version = int(archive["contract_version"])
        if version != CONTRACT_VERSION:
            raise ValueError(
                f"{source} declares SurfaceInput contract version {version}; "
                f"this build reads version {CONTRACT_VERSION}"
            )
        return SurfaceInput(
            points=np.ascontiguousarray(archive["points"], dtype=np.float64),
            tris=np.ascontiguousarray(archive["tris"], dtype=np.int32),
            tri_patch=np.ascontiguousarray(archive["tri_patch"], dtype=np.int32),
            patch_names=tuple(str(name) for name in archive["patch_names"]),
            patch_types=tuple(str(kind) for kind in archive["patch_types"]),
            vert_normal=np.ascontiguousarray(
                archive["vert_normal"], dtype=np.float64
            ),
            vert_kappa=np.ascontiguousarray(archive["vert_kappa"], dtype=np.float64),
            vert_kappa_dir=np.ascontiguousarray(
                archive["vert_kappa_dir"], dtype=np.float64
            ),
            feat_edges=np.ascontiguousarray(archive["feat_edges"], dtype=np.int32),
            feat_corners=np.ascontiguousarray(
                archive["feat_corners"], dtype=np.int32
            ),
            ref_length=float(archive["ref_length"]),
        )


__all__ = [
    "CONTRACT_VERSION",
    "SurfaceInput",
    "feature_vertices",
    "read_surface_input",
    "write_surface_input",
]
