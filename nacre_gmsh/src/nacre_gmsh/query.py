"""Low-level Gmsh BREP queries shared by the bake stages.

Everything here is a thin, batched wrapper over one Gmsh call, plus the node
table that reconciles Gmsh's per-entity node numbering with a single global
vertex indexing. Keeping them in one place is what lets ``bake``, ``orient``,
and ``features`` stay independent of each other.
"""

from __future__ import annotations

import gmsh
import numpy as np
from numpy.typing import NDArray

#: Gmsh MSH element type for a 3-node triangle.
TRIANGLE_ELEMENT_TYPE = 2


class BakeError(RuntimeError):
    """Raised when the geometry or Gmsh state makes a faithful bake impossible."""


class NodeTable:
    """Map Gmsh node tags, which repeat across shared curves, onto vertices.

    ``getNodes`` is called once per CAD face with ``includeBoundary``, so a node
    on a shared curve is returned by every face that touches it. Interning the
    tags collapses those repeats into one vertex, which is what makes the baked
    surface watertight rather than a pile of disconnected faces.
    """

    def __init__(self) -> None:
        self._index: dict[int, int] = {}
        self.coords: list[NDArray[np.float64]] = []

    def intern(
        self, tags: NDArray[np.int64], coords: NDArray[np.float64]
    ) -> NDArray[np.int32]:
        """Register node tags, returning their vertex indices."""

        out = np.empty(len(tags), dtype=np.int32)
        for position, tag in enumerate(tags.tolist()):
            existing = self._index.get(tag)
            if existing is None:
                existing = len(self.coords)
                self._index[tag] = existing
                self.coords.append(coords[position])
            out[position] = existing
        return out

    def get(self, tag: int) -> int | None:
        """Return the vertex for a Gmsh node tag, or ``None`` if unseen."""

        return self._index.get(tag)

    def lookup(self, tags: NDArray[np.int64]) -> NDArray[np.int32]:
        """Resolve already-interned node tags, failing loudly on a stranger."""

        try:
            return np.fromiter(
                (self._index[tag] for tag in tags.tolist()),
                dtype=np.int32,
                count=len(tags),
            )
        except KeyError as error:  # pragma: no cover - Gmsh invariant
            raise BakeError(
                f"triangle references node {error.args[0]}, which was not "
                "returned for any surface"
            ) from error

    def vertices(self, tris: NDArray[np.int32]) -> NDArray[np.float64]:
        """Corner coordinates of ``tris``, shaped ``(n_tris, 3, 3)``."""

        return np.asarray(self.coords, dtype=np.float64)[tris]

    @property
    def n_verts(self) -> int:
        return len(self.coords)


def surface_normals(tag: int, uv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Parametric surface normals at ``(u, v)``.

    These ignore how the face is oriented inside a solid, so they are not
    into-fluid normals until multiplied by the sign from
    ``nacre_gmsh.orient.fluid_normal_sign``.
    """

    flat = list(uv.reshape(-1))
    return np.asarray(gmsh.model.getNormal(tag, flat), dtype=np.float64).reshape(-1, 3)


def project_onto_surface(
    tag: int, xyz: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Orthogonally project points onto a surface, returning ``(xyz, uv)``."""

    closest, uv = gmsh.model.getClosestPoint(2, tag, list(xyz.reshape(-1)))
    return (
        np.asarray(closest, dtype=np.float64).reshape(-1, 3),
        np.asarray(uv, dtype=np.float64).reshape(-1, 2),
    )


def triangle_area_vectors(corners: NDArray[np.float64]) -> NDArray[np.float64]:
    """Right-hand-rule area vectors for triangles given as ``(n, 3, 3)`` corners."""

    return 0.5 * np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )


def edge_map(
    tris: NDArray[np.int32], n_verts: int
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Undirected edges of ``tris`` and the up-to-two triangles on each.

    Returns ``(edges, edge_tris)`` where ``edges`` is ``(n_edges, 2)`` sorted
    vertex pairs and ``edge_tris`` is ``(n_edges, 2)`` triangle indices with
    ``-1`` marking a boundary edge.
    """

    directed = np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0
    )
    owner = np.tile(np.arange(len(tris), dtype=np.int32), 3)
    undirected = np.sort(directed, axis=1)
    keys = undirected[:, 0].astype(np.int64) * n_verts + undirected[:, 1]

    order = np.argsort(keys, kind="stable")
    unique, starts, counts = np.unique(
        keys[order], return_index=True, return_counts=True
    )
    edges = np.stack([unique // n_verts, unique % n_verts], axis=1).astype(np.int32)
    edge_tris = np.full((len(unique), 2), -1, dtype=np.int32)
    edge_tris[:, 0] = owner[order][starts]
    paired = counts == 2
    edge_tris[paired, 1] = owner[order][starts[paired] + 1]
    return edges, edge_tris


__all__ = [
    "TRIANGLE_ELEMENT_TYPE",
    "BakeError",
    "NodeTable",
    "edge_map",
    "project_onto_surface",
    "surface_normals",
    "triangle_area_vectors",
]
