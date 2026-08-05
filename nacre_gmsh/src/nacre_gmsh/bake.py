"""Bake a Gmsh BREP model into a ``nacre.contract.SurfaceInput``.

The pipeline is STEP (or any Gmsh-readable CAD) -> surface mesh -> BREP queries
-> ``SurfaceInput`` -> ``.npz``. It runs once, offline; the nacre core then
consumes the ``.npz`` and never links Gmsh.

Both sign conventions this pipeline has to honour are documented, with the
experiments that established them, in ``nacre.contract``. The short version:
``gmsh.model.getNormal`` and ``gmsh.model.getPrincipalCurvatures`` are measured
against the *parametric* surface normal, which carries no information about
which side of the wall the fluid is on. ``nacre_gmsh.orient`` resolves that side
geometrically, and this module folds the resulting sign into both the normals
and the curvatures.

The stages live in separate modules so each is testable on its own:
``nacre_gmsh.query`` for the batched BREP calls and the global vertex table,
``nacre_gmsh.orient`` for the fluid side, ``nacre_gmsh.features`` for feature
edges and corners, and this module for meshing, per-face sampling, and assembly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np
from nacre.check import check_surface_input
from nacre.contract import SurfaceInput, write_surface_input
from numpy.typing import NDArray

from nacre_gmsh.features import classify_features
from nacre_gmsh.orient import fluid_normal_sign
from nacre_gmsh.query import (
    TRIANGLE_ELEMENT_TYPE,
    BakeError,
    NodeTable,
    project_onto_surface,
    surface_normals,
    triangle_area_vectors,
)

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class BakeSettings:
    """Meshing and classification knobs for a bake.

    Attributes:
        size_max: Upper bound on element size. ``None`` leaves Gmsh's default.
        size_min: Lower bound on element size. ``None`` leaves Gmsh's default.
        size_from_curvature: Elements per ``2*pi`` of curvature radius. Zero
            disables curvature-driven sizing.
        feature_angle_deg: Two CAD faces meeting at more than this angle share
            a feature edge.
        corner_angle_deg: Where exactly two feature curves meet at a CAD model
            point, their tangents must be collinear to within this angle for
            the point to stay a smooth feature line rather than a corner.
        orientation_samples: Triangle centroids probed per surface when
            resolving which side the fluid is on.
        default_patch_type: OpenFOAM boundary type for patches with no
            explicit override.
        patch_types: Per-patch-name overrides of ``default_patch_type``.
        algorithm: ``Mesh.Algorithm``; pinned so golden files are reproducible.
    """

    size_max: float | None = None
    size_min: float | None = None
    size_from_curvature: float = 0.0
    feature_angle_deg: float = 30.0
    corner_angle_deg: float = 30.0
    orientation_samples: int = 16
    default_patch_type: str = "wall"
    patch_types: dict[str, str] = field(default_factory=dict)
    algorithm: int = 6


def bake_step(
    path: str | Path, settings: BakeSettings | None = None
) -> SurfaceInput:
    """Import a CAD file into the current Gmsh model and bake it.

    The imported solids are taken to be the fluid domain, matching the
    ``vert_normal`` convention in ``nacre.contract``.
    """

    source = Path(path).resolve(strict=True)
    gmsh.model.add(source.stem)
    gmsh.model.occ.importShapes(str(source))
    gmsh.model.occ.synchronize()
    return bake_current_model(settings)


def bake_step_to_npz(
    path: str | Path,
    destination: str | Path,
    settings: BakeSettings | None = None,
) -> Path:
    """Bake a CAD file and persist the result as a ``.npz`` golden file."""

    return write_surface_input(bake_step(path, settings), destination)


def bake_current_model(settings: BakeSettings | None = None) -> SurfaceInput:
    """Mesh and bake the already-synchronized current Gmsh model.

    The model's 3D entities are the fluid domain. Raises ``BakeError`` if the
    model has no volume, because without one there is nothing to orient the
    normals against.
    """

    settings = settings or BakeSettings()
    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    if not volumes:
        raise BakeError(
            "the model has no 3D entity, so the fluid side of each surface "
            "cannot be determined; the STL frontend is a separate, later task"
        )

    ref_length = _reference_length()
    _apply_mesh_settings(settings, ref_length)
    gmsh.model.mesh.generate(2)

    surfaces = _fluid_boundary_surfaces(volumes)
    if not surfaces:
        raise BakeError("no surface bounds a fluid volume")

    table = NodeTable()
    per_surface: list[_SurfaceBake] = []
    for tag in surfaces:
        per_surface.append(_bake_surface(tag, volumes, ref_length, table, settings))

    return _assemble(per_surface, table, ref_length, settings)


# --------------------------------------------------------------------------- #
# Gmsh model queries
# --------------------------------------------------------------------------- #


def _reference_length() -> float:
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
    diagonal = math.dist((x0, y0, z0), (x1, y1, z1))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise BakeError(f"model bounding-box diagonal is degenerate: {diagonal!r}")
    return diagonal


def _apply_mesh_settings(settings: BakeSettings, ref_length: float) -> None:
    gmsh.option.setNumber("Mesh.Algorithm", settings.algorithm)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", settings.size_from_curvature)
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax", settings.size_max if settings.size_max else ref_length
    )
    if settings.size_min is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", settings.size_min)


def _fluid_boundary_surfaces(volumes: list[int]) -> list[int]:
    wanted = set(volumes)
    return [
        tag
        for _, tag in gmsh.model.getEntities(2)
        if wanted & set(np.asarray(gmsh.model.getAdjacencies(2, tag)[0]).tolist())
    ]


def _patch_name(tag: int, physical: dict[int, str]) -> str:
    """Resolve a patch name, preferring explicit intent over CAD labels."""

    candidate = physical.get(tag) or gmsh.model.getEntityName(2, tag)
    # Gmsh reports OCC labels as a path, e.g. "Shapes/Solid 1/inlet".
    leaf = candidate.rsplit("/", 1)[-1].strip() if candidate else ""
    sanitized = _UNSAFE_NAME.sub("_", leaf).strip("_")
    return sanitized or f"surface_{tag}"


def _physical_surface_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for dim, group in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(dim, group)
        if not name:
            continue
        for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
            names[int(tag)] = name
    return names


# --------------------------------------------------------------------------- #
# Per-surface bake
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SurfaceBake:
    """Everything one CAD face contributes, in that face's own indexing."""

    tag: int
    sign: int
    verts: NDArray[np.int32]
    normal: NDArray[np.float64]
    kappa: NDArray[np.float64]
    kappa_dir: NDArray[np.float64]
    tris: NDArray[np.int32]


def _bake_surface(
    tag: int,
    volumes: list[int],
    ref_length: float,
    table: NodeTable,
    settings: BakeSettings,
) -> _SurfaceBake:
    node_tags, node_coords, node_uv = gmsh.model.mesh.getNodes(2, tag, True, True)
    if not len(node_tags):
        raise BakeError(f"surface {tag} carries no mesh nodes")
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(node_uv, dtype=np.float64).reshape(-1, 2)
    if len(uv) != len(node_tags):
        raise BakeError(
            f"surface {tag} returned {len(uv)} parametric coordinates for "
            f"{len(node_tags)} nodes"
        )
    _assert_parametrization(tag, uv, coords, ref_length)

    verts = table.intern(node_tags, coords)
    tris = _surface_triangles(tag, table)
    sign = fluid_normal_sign(
        tag,
        volumes,
        table.vertices(tris).mean(axis=1),
        ref_length,
        settings.orientation_samples,
    )

    kappa, kappa_dir = _principal_curvatures(tag, uv, sign)
    return _SurfaceBake(
        tag=tag,
        sign=sign,
        verts=verts,
        normal=sign * surface_normals(tag, uv),
        kappa=kappa,
        kappa_dir=kappa_dir,
        tris=_wind_into_fluid(tag, tris, table, sign),
    )


def _assert_parametrization(
    tag: int,
    uv: NDArray[np.float64],
    coords: NDArray[np.float64],
    ref_length: float,
) -> None:
    """Guard the node -> parametric-coordinate lookup, the fragile step here.

    Gmsh returns the ``(u, v)`` it stored when it placed each node, so this
    round-trip is exact in practice. It is asserted anyway: a silent mismatch
    would attach every curvature and normal in the bake to the wrong point.
    """

    back = np.asarray(
        gmsh.model.getValue(2, tag, list(uv.reshape(-1))), dtype=np.float64
    ).reshape(-1, 3)
    error = float(np.max(np.linalg.norm(back - coords, axis=1), initial=0.0))
    tolerance = 1.0e-9 * ref_length
    if error > tolerance:
        raise BakeError(
            f"surface {tag} parametric coordinates do not reproduce their node "
            f"positions: max error {error:.17g} exceeds {tolerance:.17g}"
        )


def _surface_triangles(tag: int, table: NodeTable) -> NDArray[np.int32]:
    types, _, node_tags = gmsh.model.mesh.getElements(2, tag)
    found = [int(element_type) for element_type in types]
    if found != [TRIANGLE_ELEMENT_TYPE]:
        raise BakeError(
            f"surface {tag} carries element types {found}; the contract stores "
            f"triangles only (Gmsh type {TRIANGLE_ELEMENT_TYPE})"
        )
    return table.lookup(np.asarray(node_tags[0], dtype=np.int64)).reshape(-1, 3)


def _principal_curvatures(
    tag: int, uv: NDArray[np.float64], sign: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert Gmsh's parametric-normal curvatures to the nacre convention.

    Gmsh reports ``-1/R`` for a sphere whose parametric normal points away from
    the centre. The contract wants ``+1/R`` there when the fluid is outside, so
    each value is negated and multiplied by the fluid sign; the pair is then
    re-sorted descending, because negation swaps maximum and minimum, and the
    principal directions are carried along by the same reordering.
    """

    flat = list(uv.reshape(-1))
    gmsh_max, gmsh_min, gmsh_dir_max, gmsh_dir_min = (
        gmsh.model.getPrincipalCurvatures(tag, flat)
    )
    kappa = -sign * np.stack(
        [
            np.asarray(gmsh_max, dtype=np.float64),
            np.asarray(gmsh_min, dtype=np.float64),
        ],
        axis=1,
    )
    kappa_dir = np.stack(
        [
            np.asarray(gmsh_dir_max, dtype=np.float64).reshape(-1, 3),
            np.asarray(gmsh_dir_min, dtype=np.float64).reshape(-1, 3),
        ],
        axis=1,
    )
    rows = np.arange(len(kappa))[:, None]
    order = np.argsort(-kappa, axis=1, kind="stable")
    return kappa[rows, order], kappa_dir[rows, order]


def _wind_into_fluid(
    tag: int,
    tris: NDArray[np.int32],
    table: NodeTable,
    sign: int,
) -> NDArray[np.int32]:
    """Reorder triangle vertices so the right-hand-rule normal enters the fluid.

    The reference direction is the analytic surface normal at the projected
    triangle centroid, not an averaged vertex normal: on a coarse mesh near a
    feature the averaged normal can sit closer to the neighbouring face.
    """

    corners = table.vertices(tris)
    area = triangle_area_vectors(corners)
    _, uv = project_onto_surface(tag, corners.mean(axis=1))
    reference = sign * surface_normals(tag, uv)
    alignment = np.einsum("ij,ij->i", area, reference)

    magnitude = np.linalg.norm(area, axis=1)
    undecidable = np.flatnonzero(np.abs(alignment) <= 1.0e-12 * magnitude)
    if len(undecidable):
        index = int(undecidable[0])
        raise BakeError(
            f"surface {tag} triangle {index} lies edge-on to its own surface "
            "normal, so its winding cannot be resolved"
        )

    flipped = tris.copy()
    backwards = alignment < 0.0
    flipped[backwards] = flipped[backwards][:, [0, 2, 1]]
    return flipped


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _assemble(
    per_surface: list[_SurfaceBake],
    table: NodeTable,
    ref_length: float,
    settings: BakeSettings,
) -> SurfaceInput:
    n_verts = table.n_verts
    points = np.ascontiguousarray(table.coords, dtype=np.float64)

    physical = _physical_surface_names()
    patch_of_surface: dict[int, int] = {}
    patch_names: list[str] = []
    for bake in per_surface:
        name = _patch_name(bake.tag, physical)
        if name not in patch_names:
            patch_names.append(name)
        patch_of_surface[bake.tag] = patch_names.index(name)

    tris = np.concatenate([bake.tris for bake in per_surface], axis=0)
    tri_patch = np.concatenate(
        [
            np.full(len(bake.tris), patch_of_surface[bake.tag], dtype=np.int32)
            for bake in per_surface
        ]
    )
    tri_surface = np.concatenate(
        [np.full(len(bake.tris), bake.tag, dtype=np.int32) for bake in per_surface]
    )

    normal, kappa, kappa_dir = _aggregate_vertex_fields(per_surface, points, n_verts)
    feat_edges, feat_corners = classify_features(
        tris,
        tri_surface,
        points,
        {bake.tag: bake.sign for bake in per_surface},
        table,
        settings.feature_angle_deg,
        settings.corner_angle_deg,
    )

    surface = SurfaceInput(
        points=points,
        tris=np.ascontiguousarray(tris, dtype=np.int32),
        tri_patch=np.ascontiguousarray(tri_patch, dtype=np.int32),
        patch_names=tuple(patch_names),
        patch_types=tuple(
            settings.patch_types.get(name, settings.default_patch_type)
            for name in patch_names
        ),
        vert_normal=normal,
        vert_kappa=kappa,
        vert_kappa_dir=kappa_dir,
        feat_edges=feat_edges,
        feat_corners=feat_corners,
        ref_length=float(ref_length),
    )
    check_surface_input(surface)
    return surface


def _aggregate_vertex_fields(
    per_surface: list[_SurfaceBake],
    points: NDArray[np.float64],
    n_verts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Merge per-face samples at vertices shared by several CAD faces.

    Normals are averaged with the incident triangle area on each face as the
    weight. Curvature cannot be averaged meaningfully across a sharp edge, so
    the face with the largest curvature magnitude wins; that is the
    conservative choice for curvature-driven sizing, and the vertex is marked
    as a feature so consumers know the value is one of several.
    """

    accumulated = np.zeros((n_verts, 3), dtype=np.float64)
    best_magnitude = np.full(n_verts, -1.0, dtype=np.float64)
    kappa = np.zeros((n_verts, 2), dtype=np.float64)
    kappa_dir = np.zeros((n_verts, 2, 3), dtype=np.float64)

    for bake in per_surface:
        corners = points[bake.tris]
        tri_area = np.linalg.norm(triangle_area_vectors(corners), axis=1)
        weight = np.zeros(n_verts, dtype=np.float64)
        np.add.at(weight, bake.tris.reshape(-1), np.repeat(tri_area, 3) / 3.0)
        accumulated += weight[:, None] * _scatter(bake.verts, bake.normal, n_verts)

        magnitude = np.max(np.abs(bake.kappa), axis=1)
        source = np.flatnonzero(magnitude > best_magnitude[bake.verts])
        wins = bake.verts[source]
        best_magnitude[wins] = magnitude[source]
        kappa[wins] = bake.kappa[source]
        kappa_dir[wins] = bake.kappa_dir[source]

    lengths = np.linalg.norm(accumulated, axis=1)
    if np.any(lengths <= 0.0):
        index = int(np.flatnonzero(lengths <= 0.0)[0])
        raise BakeError(
            f"vertex {index} has cancelling face normals, so no into-fluid "
            "direction survives averaging"
        )
    return accumulated / lengths[:, None], kappa, kappa_dir


def _scatter(
    verts: NDArray[np.int32], values: NDArray[np.float64], n_verts: int
) -> NDArray[np.float64]:
    out = np.zeros((n_verts, values.shape[1]), dtype=np.float64)
    out[verts] = values
    return out


__all__ = [
    "BakeError",
    "BakeSettings",
    "bake_current_model",
    "bake_step",
    "bake_step_to_npz",
]
