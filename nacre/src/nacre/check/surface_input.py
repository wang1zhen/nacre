"""Invariants for ``SurfaceInput``, including both sign conventions.

The conventions in ``nacre.contract`` fail silently when a producer gets them
wrong: a flipped normal extrudes boundary layers into the solid, and a flipped
curvature makes a concave collision zone read as a convex one. Every invariant
that can catch such a flip is asserted here rather than left to discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SurfaceInputLike(Protocol):
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


class SurfaceInvariantError(ValueError):
    """Raised when a ``SurfaceInput`` violates a contract invariant."""


@dataclass(frozen=True)
class SurfaceInputCheckResult:
    n_verts: int
    n_tris: int
    n_patches: int
    n_feat_edges: int
    n_feat_verts: int
    n_feat_corners: int
    total_area: float
    signed_volume: float
    min_kappa: float
    max_kappa: float
    ref_length: float


def _require_array(
    name: str,
    value: object,
    dtype: np.dtype[np.generic],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SurfaceInvariantError(f"{name} must be a numpy array")
    if value.dtype != dtype:
        raise SurfaceInvariantError(
            f"{name} must have dtype {dtype}, got {value.dtype}"
        )
    if value.ndim != len(shape):
        raise SurfaceInvariantError(
            f"{name} must be {len(shape)}D, got shape {value.shape}"
        )
    for axis, expected in enumerate(shape):
        if expected is not None and value.shape[axis] != expected:
            raise SurfaceInvariantError(
                f"{name} must have shape {shape}, got {value.shape}"
            )
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(value)):
        raise SurfaceInvariantError(f"{name} contains NaN or infinity")
    return value


def check_surface_input(
    surface: SurfaceInputLike,
    *,
    require_closed: bool = True,
    unit_atol: float = 1.0e-9,
    orthogonality_atol: float = 1.0e-9,
) -> SurfaceInputCheckResult:
    """Validate a baked surface against the frozen contract.

    A successful return is the verdict. The first violated invariant raises
    ``SurfaceInvariantError`` with a diagnostic naming the offending entity.

    ``require_closed`` demands a watertight, consistently wound triangulation:
    every edge shared by exactly two triangles traversed in opposite
    directions. Leave it enabled for a fluid-domain boundary. The STL frontend
    planned for a later task may produce open surfaces, which is the only
    intended reason to disable it.
    """

    points = _require_array("points", surface.points, np.dtype(np.float64), (None, 3))
    n_verts = len(points)
    if n_verts == 0:
        raise SurfaceInvariantError("surface must contain vertices")

    tris = _require_array("tris", surface.tris, np.dtype(np.int32), (None, 3))
    n_tris = len(tris)
    if n_tris == 0:
        raise SurfaceInvariantError("surface must contain triangles")
    tri_patch = _require_array(
        "tri_patch", surface.tri_patch, np.dtype(np.int32), (n_tris,)
    )
    normal = _require_array(
        "vert_normal", surface.vert_normal, np.dtype(np.float64), (n_verts, 3)
    )
    kappa = _require_array(
        "vert_kappa", surface.vert_kappa, np.dtype(np.float64), (n_verts, 2)
    )
    kappa_dir = _require_array(
        "vert_kappa_dir",
        surface.vert_kappa_dir,
        np.dtype(np.float64),
        (n_verts, 2, 3),
    )
    feat_edges = _require_array(
        "feat_edges", surface.feat_edges, np.dtype(np.int32), (None, 2)
    )
    feat_corners = _require_array(
        "feat_corners", surface.feat_corners, np.dtype(np.int32), (None,)
    )

    if not isinstance(surface.ref_length, float):
        raise SurfaceInvariantError("ref_length must be a Python float")
    if not np.isfinite(surface.ref_length) or surface.ref_length <= 0.0:
        raise SurfaceInvariantError(
            f"ref_length must be finite and positive, got {surface.ref_length!r}"
        )
    ref_length = surface.ref_length

    _check_patches(surface, tri_patch)
    _check_triangles(points, tris, n_verts, ref_length)
    edge_report = _check_edge_topology(tris, n_verts, require_closed=require_closed)
    _check_normals(normal, unit_atol)
    _check_winding(points, tris, normal)
    smooth = _check_features(feat_edges, feat_corners, n_verts, edge_report)
    _check_curvature_frame(
        kappa, kappa_dir, normal, smooth, unit_atol, orthogonality_atol, ref_length
    )

    area_vectors = _triangle_area_vectors(points, tris)
    centroids = points[tris].mean(axis=1)
    return SurfaceInputCheckResult(
        n_verts=n_verts,
        n_tris=n_tris,
        n_patches=len(surface.patch_names),
        n_feat_edges=len(feat_edges),
        n_feat_verts=int(n_verts - np.count_nonzero(smooth)),
        n_feat_corners=len(feat_corners),
        total_area=float(np.sum(np.linalg.norm(area_vectors, axis=1))),
        signed_volume=float(np.sum(centroids * area_vectors) / 3.0),
        min_kappa=float(np.min(kappa)),
        max_kappa=float(np.max(kappa)),
        ref_length=ref_length,
    )


def _check_patches(surface: SurfaceInputLike, tri_patch: NDArray[np.int32]) -> None:
    names = surface.patch_names
    types = surface.patch_types
    if not isinstance(names, tuple) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise SurfaceInvariantError("patch_names must be a tuple of non-empty strings")
    if not isinstance(types, tuple) or not all(
        isinstance(kind, str) and kind for kind in types
    ):
        raise SurfaceInvariantError("patch_types must be a tuple of non-empty strings")
    if len(names) != len(types):
        raise SurfaceInvariantError(
            "patch_names and patch_types must have equal length"
        )
    if not names:
        raise SurfaceInvariantError("surface must declare at least one patch")
    if len(set(names)) != len(names):
        raise SurfaceInvariantError("patch names must be unique")
    if np.any(tri_patch < 0) or np.any(tri_patch >= len(names)):
        raise SurfaceInvariantError("tri_patch contains an out-of-range patch index")
    used = np.bincount(tri_patch, minlength=len(names))
    empty = np.flatnonzero(used == 0)
    if len(empty):
        missing = ", ".join(names[i] for i in empty.tolist())
        raise SurfaceInvariantError(f"patches without triangles: {missing}")


def _triangle_area_vectors(
    points: NDArray[np.float64], tris: NDArray[np.int32]
) -> NDArray[np.float64]:
    corners = points[tris]
    return 0.5 * np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )


def _check_triangles(
    points: NDArray[np.float64],
    tris: NDArray[np.int32],
    n_verts: int,
    ref_length: float,
) -> None:
    if np.any(tris < 0) or np.any(tris >= n_verts):
        raise SurfaceInvariantError("tris contains an out-of-range vertex index")
    degenerate = np.flatnonzero(
        (tris[:, 0] == tris[:, 1])
        | (tris[:, 1] == tris[:, 2])
        | (tris[:, 0] == tris[:, 2])
    )
    if len(degenerate):
        raise SurfaceInvariantError(f"triangle {int(degenerate[0])} repeats a vertex")

    canonical = np.sort(tris, axis=1)
    unique, counts = np.unique(canonical, axis=0, return_counts=True)
    duplicated = np.flatnonzero(counts > 1)
    if len(duplicated):
        vertices = unique[duplicated[0]].tolist()
        raise SurfaceInvariantError(f"duplicate triangle on vertices {vertices}")

    used = np.zeros(n_verts, dtype=bool)
    used[tris.reshape(-1)] = True
    if not np.all(used):
        unused = np.flatnonzero(~used)
        raise SurfaceInvariantError(
            f"{len(unused)} vertices belong to no triangle, first is {int(unused[0])}"
        )

    areas = np.linalg.norm(_triangle_area_vectors(points, tris), axis=1)
    tolerance = 1.0e-24 * ref_length * ref_length
    sliver = np.flatnonzero(areas <= tolerance)
    if len(sliver):
        index = int(sliver[0])
        raise SurfaceInvariantError(
            f"triangle {index} has degenerate area {areas[index]:.17g}"
        )


@dataclass(frozen=True)
class _EdgeReport:
    edge_keys: NDArray[np.int64]
    edge_tris: NDArray[np.int32]


def _check_edge_topology(
    tris: NDArray[np.int32], n_verts: int, *, require_closed: bool
) -> _EdgeReport:
    """Verify manifoldness and, when required, closure and consistent winding."""

    n_tris = len(tris)
    directed = np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0
    ).astype(np.int64)
    owning_tri = np.tile(np.arange(n_tris, dtype=np.int32), 3)
    undirected = np.sort(directed, axis=1)
    keys = undirected[:, 0] * n_verts + undirected[:, 1]

    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    unique_keys, starts, counts = np.unique(
        keys, return_index=True, return_counts=True
    )
    if np.any(counts > 2):
        index = int(np.flatnonzero(counts > 2)[0])
        key = int(unique_keys[index])
        raise SurfaceInvariantError(
            f"edge ({key // n_verts}, {key % n_verts}) is shared by "
            f"{int(counts[index])} triangles; the surface is non-manifold"
        )

    if require_closed:
        boundary = np.flatnonzero(counts == 1)
        if len(boundary):
            key = int(unique_keys[boundary[0]])
            raise SurfaceInvariantError(
                f"{len(boundary)} edges border a single triangle; the surface is "
                f"not watertight, first is ({key // n_verts}, {key % n_verts})"
            )
        # A closed, consistently wound surface traverses each edge once in each
        # direction. Equal directed keys on both sides means one triangle is
        # wound backwards, which is exactly a flipped into-fluid normal.
        directed_keys = directed[order, 0] * n_verts + directed[order, 1]
        clashing = np.flatnonzero(directed_keys[starts] == directed_keys[starts + 1])
        if len(clashing):
            key = int(unique_keys[clashing[0]])
            raise SurfaceInvariantError(
                f"edge ({key // n_verts}, {key % n_verts}) is traversed in the "
                "same direction by both its triangles; triangle winding is "
                "inconsistent"
            )

    edge_tris = np.full((len(unique_keys), 2), -1, dtype=np.int32)
    sorted_tris = owning_tri[order]
    edge_tris[:, 0] = sorted_tris[starts]
    paired = counts == 2
    edge_tris[paired, 1] = sorted_tris[starts[paired] + 1]
    return _EdgeReport(edge_keys=unique_keys, edge_tris=edge_tris)


def _check_normals(normal: NDArray[np.float64], unit_atol: float) -> None:
    lengths = np.linalg.norm(normal, axis=1)
    offending = np.flatnonzero(np.abs(lengths - 1.0) > unit_atol)
    if len(offending):
        index = int(offending[0])
        raise SurfaceInvariantError(
            f"vert_normal[{index}] has length {lengths[index]:.17g}, not unit"
        )


def _check_winding(
    points: NDArray[np.float64],
    tris: NDArray[np.int32],
    normal: NDArray[np.float64],
) -> None:
    """Triangle normals must agree with the into-fluid vertex normals.

    The test is per corner rather than against the mean of the three normals.
    Averaging first lets a single reversed normal hide behind its two
    neighbours, and a single reversed normal is precisely the failure that
    extrudes one boundary-layer column into the solid.
    """

    area_vectors = _triangle_area_vectors(points, tris)
    alignment = np.einsum("ij,ikj->ik", area_vectors, normal[tris])
    offending = np.argwhere(alignment <= 0.0)
    if len(offending):
        triangle, corner = (int(value) for value in offending[0])
        vertex = int(tris[triangle, corner])
        raise SurfaceInvariantError(
            f"triangle {triangle} is wound against vert_normal[{vertex}] "
            f"(alignment {alignment[triangle, corner]:.17g}); the into-fluid "
            "orientation convention is violated"
        )


def _check_curvature_frame(
    kappa: NDArray[np.float64],
    kappa_dir: NDArray[np.float64],
    normal: NDArray[np.float64],
    smooth: NDArray[np.bool_],
    unit_atol: float,
    orthogonality_atol: float,
    ref_length: float,
) -> None:
    unordered = np.flatnonzero(kappa[:, 1] > kappa[:, 0])
    if len(unordered):
        index = int(unordered[0])
        raise SurfaceInvariantError(
            f"vert_kappa[{index}]={kappa[index].tolist()} is not sorted "
            "descending; the larger principal curvature must come first"
        )
    # A curvature radius far below machine resolution of the geometry is a
    # producer bug, not a feature; a true sharp edge belongs in feat_edges.
    limit = 1.0e12 / ref_length
    excessive = np.flatnonzero(np.any(np.abs(kappa) > limit, axis=1))
    if len(excessive):
        index = int(excessive[0])
        raise SurfaceInvariantError(
            f"vertex {index} curvature {kappa[index].tolist()} exceeds the "
            f"resolvable limit {limit:.17g} for ref_length {ref_length:.17g}"
        )

    lengths = np.linalg.norm(kappa_dir, axis=2)
    degenerate = np.all(lengths == 0.0, axis=1)
    half_degenerate = np.flatnonzero(np.any(lengths == 0.0, axis=1) & ~degenerate)
    if len(half_degenerate):
        index = int(half_degenerate[0])
        raise SurfaceInvariantError(
            f"vertex {index} has exactly one zero principal direction; a "
            "parametrization singularity must zero both or neither"
        )

    live = ~degenerate
    if not np.any(live):
        return
    for row in (0, 1):
        offending = np.flatnonzero(np.abs(lengths[live, row] - 1.0) > unit_atol)
        if len(offending):
            index = int(np.flatnonzero(live)[offending[0]])
            raise SurfaceInvariantError(
                f"vert_kappa_dir[{index}, {row}] has length "
                f"{lengths[index, row]:.17g}, not unit"
            )

    # The curvature frame must be tangent to the surface it was sampled on. At
    # a feature vertex, vert_normal is an average over two or more incident
    # faces while the frame belongs to just one of them, so the frame is not
    # expected to be tangent to that average; only smooth vertices, where
    # vert_normal is the exact single-face normal, are checked against it.
    tangent = live & smooth
    first, second = kappa_dir[:, 0], kappa_dir[:, 1]
    for left_name, left, right_name, right, selection in (
        ("vert_kappa_dir[:, 0]", first, "vert_kappa_dir[:, 1]", second, live),
        ("vert_kappa_dir[:, 0]", first, "vert_normal", normal, tangent),
        ("vert_kappa_dir[:, 1]", second, "vert_normal", normal, tangent),
    ):
        if not np.any(selection):
            continue
        dots = np.abs(np.einsum("ij,ij->i", left[selection], right[selection]))
        offending = np.flatnonzero(dots > orthogonality_atol)
        if len(offending):
            index = int(np.flatnonzero(selection)[offending[0]])
            raise SurfaceInvariantError(
                f"{left_name} at vertex {index} is not orthogonal to "
                f"{right_name}: dot product {dots[offending[0]]:.17g}"
            )


def _check_features(
    feat_edges: NDArray[np.int32],
    feat_corners: NDArray[np.int32],
    n_verts: int,
    edge_report: _EdgeReport,
) -> NDArray[np.bool_]:
    """Validate the feature lists and return the smooth-vertex mask."""

    smooth = np.ones(n_verts, dtype=bool)
    if np.any(feat_corners < 0) or np.any(feat_corners >= n_verts):
        raise SurfaceInvariantError("feat_corners contains an out-of-range index")
    if len(np.unique(feat_corners)) != len(feat_corners):
        raise SurfaceInvariantError("feat_corners contains a duplicate vertex")

    if not len(feat_edges):
        if len(feat_corners):
            raise SurfaceInvariantError(
                f"vertex {int(feat_corners[0])} is a corner but feat_edges is empty"
            )
        return smooth

    if np.any(feat_edges < 0) or np.any(feat_edges >= n_verts):
        raise SurfaceInvariantError("feat_edges contains an out-of-range vertex index")
    if np.any(feat_edges[:, 0] == feat_edges[:, 1]):
        raise SurfaceInvariantError("feat_edges contains a self-loop")

    ordered = np.sort(feat_edges, axis=1)
    keys = ordered[:, 0].astype(np.int64) * n_verts + ordered[:, 1]
    if len(np.unique(keys)) != len(keys):
        raise SurfaceInvariantError("feat_edges contains a duplicate edge")
    unknown = ~np.isin(keys, edge_report.edge_keys)
    if np.any(unknown):
        index = int(np.flatnonzero(unknown)[0])
        raise SurfaceInvariantError(
            f"feature edge {feat_edges[index].tolist()} is not an edge of any "
            "triangle"
        )

    endpoints = np.unique(feat_edges)
    stray = feat_corners[~np.isin(feat_corners, endpoints)]
    if len(stray):
        raise SurfaceInvariantError(
            f"corner vertex {int(stray[0])} lies on no feature edge"
        )

    incident = np.bincount(feat_edges.reshape(-1), minlength=n_verts)
    under_connected = feat_corners[incident[feat_corners] < 2]
    if len(under_connected):
        index = int(under_connected[0])
        raise SurfaceInvariantError(
            f"corner vertex {index} touches only {int(incident[index])} feature "
            "edges; a corner needs at least two"
        )
    junctions = np.flatnonzero(incident >= 3)
    unmarked = junctions[~np.isin(junctions, feat_corners)]
    if len(unmarked):
        index = int(unmarked[0])
        raise SurfaceInvariantError(
            f"vertex {index} joins {int(incident[index])} feature edges but is "
            "not listed in feat_corners"
        )

    smooth[endpoints] = False
    smooth[feat_corners] = False
    return smooth


__all__ = [
    "SurfaceInputCheckResult",
    "SurfaceInvariantError",
    "check_surface_input",
]
