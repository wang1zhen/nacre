"""Extract feature edges and corners from CAD topology and exact CAD geometry.

Both classifications are deliberately independent of mesh resolution. A facet
angle threshold over the triangulation cannot separate a genuinely sharp edge
from a coarsely faceted smooth one, and cannot separate a corner from a node on
a coarse circular rim. Working from the CAD faces, curves, and model points
instead makes the result a property of the geometry rather than of the mesh
size the bake happened to use.
"""

from __future__ import annotations

import math

import gmsh
import numpy as np
from numpy.typing import NDArray

from nacre_gmsh.orient import into_fluid_normals_at
from nacre_gmsh.query import BakeError, NodeTable, edge_map

_NO_FEATURES = (np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32))


def classify_features(
    tris: NDArray[np.int32],
    tri_surface: NDArray[np.int32],
    points: NDArray[np.float64],
    sign_of: dict[int, int],
    table: NodeTable,
    feature_angle_deg: float,
    corner_angle_deg: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Return ``(feat_edges, feat_corners)`` for a baked surface.

    Candidate feature edges are the mesh edges whose two triangles belong to
    different CAD faces, so a seam inside one face never registers and neither
    does the facetting of a smooth face. Each candidate's dihedral angle is then
    measured between the two faces' own analytic into-fluid normals at the
    shared edge midpoint.
    """

    n_verts = len(points)
    edges, edge_tris = edge_map(tris, n_verts)

    interior = edge_tris[:, 1] >= 0
    left = tri_surface[edge_tris[interior, 0]]
    right = tri_surface[edge_tris[interior, 1]]
    candidates = np.flatnonzero(interior)[left != right]
    if not len(candidates):
        return _NO_FEATURES

    midpoints = points[edges[candidates]].mean(axis=1)
    left_normal = into_fluid_normals_at(
        tri_surface[edge_tris[candidates, 0]], midpoints, sign_of
    )
    right_normal = into_fluid_normals_at(
        tri_surface[edge_tris[candidates, 1]], midpoints, sign_of
    )
    alignment = np.einsum("ij,ij->i", left_normal, right_normal)
    sharp = candidates[alignment < math.cos(math.radians(feature_angle_deg))]
    if not len(sharp):
        return _NO_FEATURES

    feat_edges = np.ascontiguousarray(edges[sharp], dtype=np.int32)
    return feat_edges, _corner_vertices(feat_edges, table, n_verts, corner_angle_deg)


def _corner_vertices(
    feat_edges: NDArray[np.int32],
    table: NodeTable,
    n_verts: int,
    corner_angle_deg: float,
) -> NDArray[np.int32]:
    """Locate corners from CAD topology and CAD tangents, not the mesh polyline.

    Measuring the kink in the chain of feature *mesh* edges does not work: a
    coarsely meshed circular rim turns exactly ``360/n`` degrees at every node,
    so any threshold below that promotes a perfectly smooth rim to a ring of
    corners. Corners are therefore decided at CAD model points, comparing the
    analytic tangents of the feature curves that meet there.
    """

    incident = np.bincount(feat_edges.reshape(-1), minlength=n_verts)
    corners = set(np.flatnonzero(incident >= 3).tolist())

    feature_curves = _feature_curves(feat_edges, table, n_verts)
    kink_threshold = math.cos(math.radians(corner_angle_deg))
    for point_tag, vertex in _model_point_vertices(table).items():
        if incident[vertex] < 2:
            continue
        adjacent = np.asarray(gmsh.model.getAdjacencies(0, point_tag)[0]).tolist()
        meeting = [curve for curve in adjacent if int(curve) in feature_curves]
        if len(meeting) >= 3:
            corners.add(vertex)
        elif len(meeting) == 2:
            first, second = (
                _curve_tangent(int(curve), gmsh.model.getValue(0, point_tag, []))
                for curve in meeting
            )
            # Curve parametrization direction is arbitrary, so compare
            # collinearity rather than a signed angle.
            if abs(float(first @ second)) < kink_threshold:
                corners.add(vertex)
    return np.asarray(sorted(corners), dtype=np.int32)


def _feature_curves(
    feat_edges: NDArray[np.int32], table: NodeTable, n_verts: int
) -> set[int]:
    """CAD curves carrying at least one sharp mesh edge."""

    owner = np.full(n_verts, -1, dtype=np.int64)
    for _, curve in gmsh.model.getEntities(1):
        node_tags, _, _ = gmsh.model.mesh.getNodes(1, curve, False, False)
        for tag in np.asarray(node_tags, dtype=np.int64).tolist():
            vertex = table.get(tag)
            if vertex is not None:
                owner[vertex] = curve

    found: set[int] = set()
    for vertex in np.unique(feat_edges).tolist():
        if owner[vertex] >= 0:
            found.add(int(owner[vertex]))
    return found


def _model_point_vertices(table: NodeTable) -> dict[int, int]:
    """Map each CAD model point onto the mesh vertex sitting on it."""

    located: dict[int, int] = {}
    for _, point_tag in gmsh.model.getEntities(0):
        node_tags, _, _ = gmsh.model.mesh.getNodes(0, point_tag, False, False)
        for tag in np.asarray(node_tags, dtype=np.int64).tolist():
            vertex = table.get(tag)
            if vertex is not None:
                located[int(point_tag)] = vertex
    return located


def _curve_tangent(curve: int, point: list[float]) -> NDArray[np.float64]:
    parameter = gmsh.model.getParametrization(1, curve, list(point))
    derivative = np.asarray(
        gmsh.model.getDerivative(1, curve, list(parameter)), dtype=np.float64
    ).reshape(3)
    length = float(np.linalg.norm(derivative))
    if length == 0.0:
        raise BakeError(f"curve {curve} has a zero tangent at {point}")
    return derivative / length


__all__ = ["classify_features"]
