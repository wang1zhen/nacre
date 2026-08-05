"""Structural and OpenFOAM-ordering invariants for ``PolyMeshIR``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class MeshIRLike(Protocol):
    points: NDArray[np.float64]
    face_verts: NDArray[np.int32]
    face_offset: NDArray[np.int32]
    owner: NDArray[np.int32]
    neighbour: NDArray[np.int32]
    patch_names: tuple[str, ...]
    patch_types: tuple[str, ...]
    patch_offset: NDArray[np.int32]


class MeshInvariantError(ValueError):
    """Raised when a mesh violates a structural or OpenFOAM invariant."""


@dataclass(frozen=True)
class MeshIRCheckResult:
    n_points: int
    n_faces: int
    n_internal_faces: int
    n_boundary_faces: int
    n_cells: int
    max_closure_residual: float
    total_volume: float


def _require_array(
    name: str,
    value: object,
    dtype: np.dtype[np.generic],
    ndim: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise MeshInvariantError(f"{name} must be a numpy array")
    if value.dtype != dtype:
        raise MeshInvariantError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.ndim != ndim:
        raise MeshInvariantError(f"{name} must be {ndim}D, got shape {value.shape}")
    return value


def _face_area_and_centre(
    vertices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Match OpenFOAM's face decomposition for planar and warped polygons."""

    average = vertices.mean(axis=0)
    following = np.roll(vertices, -1, axis=0)
    triangle_areas = 0.5 * np.cross(following - vertices, average - vertices)
    triangle_centres = (vertices + following + average) / 3.0
    weights = np.linalg.norm(triangle_areas, axis=1)
    weight_sum = float(np.sum(weights))
    if weight_sum == 0.0:
        return np.sum(triangle_areas, axis=0), average
    centre = np.sum(triangle_centres * weights[:, None], axis=0) / weight_sum
    return np.sum(triangle_areas, axis=0), centre


def check_meshir(
    mesh: MeshIRLike,
    *,
    closure_rtol: float = 1.0e-10,
    closure_atol: float = 1.0e-12,
    normal_rtol: float = 1.0e-12,
) -> MeshIRCheckResult:
    """Validate topology, geometry, and OpenFOAM face-ordering conventions.

    A successful return is the verdict. The first violated invariant raises
    ``MeshInvariantError`` with a diagnostic naming the offending entity.
    """

    points = _require_array("points", mesh.points, np.dtype(np.float64), 2)
    face_verts = _require_array(
        "face_verts", mesh.face_verts, np.dtype(np.int32), 1
    )
    face_offset = _require_array(
        "face_offset", mesh.face_offset, np.dtype(np.int32), 1
    )
    owner = _require_array("owner", mesh.owner, np.dtype(np.int32), 1)
    neighbour = _require_array(
        "neighbour", mesh.neighbour, np.dtype(np.int32), 1
    )
    patch_offset = _require_array(
        "patch_offset", mesh.patch_offset, np.dtype(np.int32), 1
    )

    if points.shape[1:] != (3,):
        raise MeshInvariantError(f"points must have shape (Np, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise MeshInvariantError("points contain NaN or infinity")

    n_points = len(points)
    n_faces = len(owner)
    n_internal = len(neighbour)
    n_boundary = n_faces - n_internal

    if n_points == 0 or n_faces == 0:
        raise MeshInvariantError("mesh must contain points and faces")
    if n_boundary < 0:
        raise MeshInvariantError("neighbour cannot contain more entries than owner")
    if face_offset.shape != (n_faces + 1,):
        raise MeshInvariantError(
            f"face_offset must have shape ({n_faces + 1},), got {face_offset.shape}"
        )
    if face_offset[0] != 0:
        raise MeshInvariantError("face_offset[0] must be zero")
    if np.any(np.diff(face_offset) < 0):
        raise MeshInvariantError("face_offset must be monotonic")
    if int(face_offset[-1]) != len(face_verts):
        raise MeshInvariantError("face_offset[-1] must equal len(face_verts)")
    if np.any(np.diff(face_offset) < 3):
        raise MeshInvariantError("every face must contain at least three vertices")
    if np.any(face_verts < 0) or np.any(face_verts >= n_points):
        raise MeshInvariantError("face_verts contains an out-of-range point index")
    if np.any(owner < 0) or np.any(neighbour < 0):
        raise MeshInvariantError("owner and neighbour indices must be non-negative")

    if not isinstance(mesh.patch_names, tuple) or not all(
        isinstance(name, str) and name for name in mesh.patch_names
    ):
        raise MeshInvariantError("patch_names must be a tuple of non-empty strings")
    if not isinstance(mesh.patch_types, tuple) or not all(
        isinstance(kind, str) and kind for kind in mesh.patch_types
    ):
        raise MeshInvariantError("patch_types must be a tuple of non-empty strings")
    if len(mesh.patch_names) != len(mesh.patch_types):
        raise MeshInvariantError("patch_names and patch_types must have equal length")
    if len(set(mesh.patch_names)) != len(mesh.patch_names):
        raise MeshInvariantError("patch names must be unique")
    if patch_offset.shape != (len(mesh.patch_names) + 1,):
        raise MeshInvariantError(
            "patch_offset must contain one more entry than patch_names"
        )
    if patch_offset[0] != 0 or np.any(np.diff(patch_offset) < 0):
        raise MeshInvariantError("patch_offset must start at zero and be monotonic")
    if int(patch_offset[-1]) != n_boundary:
        raise MeshInvariantError(
            "patch_offset[-1] must equal the number of boundary faces"
        )
    if n_boundary and np.any(np.diff(patch_offset) == 0):
        raise MeshInvariantError("boundary patches must not be empty")

    if n_internal and np.any(owner[:n_internal] >= neighbour):
        raise MeshInvariantError("owner must be less than neighbour on internal faces")
    internal_pairs = list(zip(owner[:n_internal].tolist(), neighbour.tolist()))
    if internal_pairs != sorted(internal_pairs):
        raise MeshInvariantError(
            "internal faces must come first in (owner, neighbour) order"
        )

    face_indices: list[NDArray[np.int32]] = []
    canonical_faces: set[tuple[int, ...]] = set()
    used_points = np.zeros(n_points, dtype=bool)
    for face_i in range(n_faces):
        start = int(face_offset[face_i])
        stop = int(face_offset[face_i + 1])
        indices = face_verts[start:stop]
        if len(np.unique(indices)) != len(indices):
            raise MeshInvariantError(f"face {face_i} repeats a point")
        canonical = tuple(sorted(indices.tolist()))
        if canonical in canonical_faces:
            raise MeshInvariantError(f"face {face_i} duplicates another face")
        canonical_faces.add(canonical)
        used_points[indices] = True
        face_indices.append(indices)
    if not np.all(used_points):
        unused = np.flatnonzero(~used_points).tolist()
        raise MeshInvariantError(f"unused points: {unused}")

    max_cell = max(
        int(np.max(owner, initial=-1)),
        int(np.max(neighbour, initial=-1)),
    )
    n_cells = max_cell + 1
    referenced_cells = np.zeros(n_cells, dtype=bool)
    referenced_cells[owner] = True
    referenced_cells[neighbour] = True
    if not np.all(referenced_cells):
        missing = np.flatnonzero(~referenced_cells).tolist()
        raise MeshInvariantError(f"unreferenced cell indices: {missing}")

    cell_points: list[set[int]] = [set() for _ in range(n_cells)]
    for face_i, indices in enumerate(face_indices):
        cell_points[int(owner[face_i])].update(indices.tolist())
        if face_i < n_internal:
            cell_points[int(neighbour[face_i])].update(indices.tolist())
    cell_centres = np.vstack(
        [points[np.fromiter(ids, dtype=np.int32)].mean(axis=0) for ids in cell_points]
    )

    closure = np.zeros((n_cells, 3), dtype=np.float64)
    area_sum = np.zeros(n_cells, dtype=np.float64)
    cell_volume = np.zeros(n_cells, dtype=np.float64)
    for face_i, indices in enumerate(face_indices):
        vertices = points[indices]
        area, face_centre = _face_area_and_centre(vertices)
        area_magnitude = float(np.linalg.norm(area))
        if not np.isfinite(area_magnitude) or area_magnitude <= closure_atol:
            raise MeshInvariantError(f"face {face_i} has zero or invalid area")

        owner_i = int(owner[face_i])
        closure[owner_i] += area
        area_sum[owner_i] += area_magnitude
        volume_contribution = float(np.dot(area, face_centre)) / 3.0
        cell_volume[owner_i] += volume_contribution
        if face_i < n_internal:
            neighbour_i = int(neighbour[face_i])
            direction = cell_centres[neighbour_i] - cell_centres[owner_i]
            scale = area_magnitude * float(np.linalg.norm(direction))
            if float(np.dot(area, direction)) <= normal_rtol * scale:
                raise MeshInvariantError(
                    f"internal face {face_i} normal does not point owner-to-neighbour"
                )
            closure[neighbour_i] -= area
            area_sum[neighbour_i] += area_magnitude
            cell_volume[neighbour_i] -= volume_contribution
        else:
            direction = face_centre - cell_centres[owner_i]
            scale = area_magnitude * float(np.linalg.norm(direction))
            if float(np.dot(area, direction)) <= normal_rtol * scale:
                raise MeshInvariantError(
                    f"boundary face {face_i} normal does not point out of owner"
                )

    residuals = np.linalg.norm(closure, axis=1)
    tolerances = closure_atol + closure_rtol * area_sum
    bad_cells = np.flatnonzero(residuals > tolerances)
    if len(bad_cells):
        cell_i = int(bad_cells[0])
        raise MeshInvariantError(
            f"cell {cell_i} is not closed: residual={residuals[cell_i]:.17g}, "
            f"tolerance={tolerances[cell_i]:.17g}"
        )

    bad_volume = np.flatnonzero(cell_volume <= closure_atol)
    if len(bad_volume):
        cell_i = int(bad_volume[0])
        raise MeshInvariantError(
            f"cell {cell_i} has non-positive volume {cell_volume[cell_i]:.17g}"
        )

    return MeshIRCheckResult(
        n_points=n_points,
        n_faces=n_faces,
        n_internal_faces=n_internal,
        n_boundary_faces=n_boundary,
        n_cells=n_cells,
        max_closure_residual=float(np.max(residuals, initial=0.0)),
        total_volume=float(np.sum(cell_volume)),
    )


__all__ = ["MeshIRCheckResult", "MeshInvariantError", "check_meshir"]
