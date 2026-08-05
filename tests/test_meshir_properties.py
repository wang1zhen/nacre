from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, strategies as st

from fixtures import (
    block_2x2x2,
    block_2x2x2_multi_patch,
    single_hexahedron,
    two_cells_with_hanging_node,
    two_cells_with_warped_face,
)
from nacre.check import MeshInvariantError, check_meshir
from nacre.meshir import PolyMeshIR


_FINITE = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_POSITIVE = st.floats(
    min_value=0.5,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_SHEAR = st.floats(
    min_value=-0.2,
    max_value=0.2,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def valid_csr_meshes(draw: st.DrawFn) -> PolyMeshIR:
    factory = draw(
        st.sampled_from(
            [
                single_hexahedron,
                block_2x2x2,
                two_cells_with_hanging_node,
                two_cells_with_warped_face,
                block_2x2x2_multi_patch,
            ]
        )
    )
    mesh = factory()
    sx, sy, sz = draw(st.tuples(_POSITIVE, _POSITIVE, _POSITIVE))
    xy, xz, yz = draw(st.tuples(_SHEAR, _SHEAR, _SHEAR))
    translation = np.asarray(
        draw(st.tuples(_FINITE, _FINITE, _FINITE)), dtype=np.float64
    )
    transform = np.asarray(
        [[sx, xy, xz], [0.0, sy, yz], [0.0, 0.0, sz]], dtype=np.float64
    )
    points = np.asarray(mesh.points @ transform.T + translation, dtype=np.float64)
    return replace(mesh, points=points)


def _replace_face(mesh: PolyMeshIR, face_i: int, vertices: np.ndarray) -> PolyMeshIR:
    start = int(mesh.face_offset[face_i])
    stop = int(mesh.face_offset[face_i + 1])
    if len(vertices) != stop - start:
        raise ValueError("replacement face must preserve CSR length")
    face_verts = mesh.face_verts.copy()
    face_verts[start:stop] = vertices
    return replace(mesh, face_verts=face_verts)


def _replace_faces(
    mesh: PolyMeshIR, replacements: dict[int, tuple[int, ...]]
) -> PolyMeshIR:
    faces = [
        tuple(
            int(vertex)
            for vertex in mesh.face_verts[
                mesh.face_offset[face_i] : mesh.face_offset[face_i + 1]
            ]
        )
        for face_i in range(mesh.n_faces)
    ]
    for face_i, vertices in replacements.items():
        faces[face_i] = vertices
    sizes = np.asarray([len(face) for face in faces], dtype=np.int32)
    offsets = np.empty(len(faces) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    flat = np.asarray([vertex for face in faces for vertex in face], dtype=np.int32)
    return replace(mesh, face_verts=flat, face_offset=offsets)


@given(valid_csr_meshes())
def test_random_valid_csr_meshes_pass_every_invariant(mesh: PolyMeshIR) -> None:
    result = check_meshir(mesh)

    assert result.n_points == len(mesh.points)
    assert result.n_faces == mesh.n_faces
    assert result.n_internal_faces == mesh.n_internal_faces
    assert result.n_boundary_faces == mesh.n_boundary_faces
    assert result.n_cells == mesh.n_cells
    assert result.total_volume > 0.0


def test_rejects_internal_face_without_two_adjacencies() -> None:
    mesh = single_hexahedron()
    too_many_neighbours = np.zeros(mesh.n_faces + 1, dtype=np.int32)

    with pytest.raises(MeshInvariantError, match="more entries than owner"):
        check_meshir(replace(mesh, neighbour=too_many_neighbours))


@given(st.integers(min_value=0, max_value=11))
def test_rejects_owner_not_less_than_neighbour(face_i: int) -> None:
    mesh = block_2x2x2()
    neighbour = mesh.neighbour.copy()
    neighbour[face_i] = mesh.owner[face_i]

    with pytest.raises(MeshInvariantError, match="owner must be less"):
        check_meshir(replace(mesh, neighbour=neighbour))


def test_rejects_internal_faces_out_of_upper_triangular_order() -> None:
    mesh = block_2x2x2()
    owner = mesh.owner.copy()
    neighbour = mesh.neighbour.copy()
    owner[[0, 1]] = owner[[1, 0]]
    neighbour[[0, 1]] = neighbour[[1, 0]]

    with pytest.raises(MeshInvariantError, match="internal faces must come first"):
        check_meshir(replace(mesh, owner=owner, neighbour=neighbour))


def test_rejects_internal_face_outside_internal_prefix() -> None:
    mesh = block_2x2x2()
    first_internal = mesh.face_verts[mesh.face_offset[0] : mesh.face_offset[1]].copy()
    first_boundary_i = mesh.n_internal_faces
    first_boundary = mesh.face_verts[
        mesh.face_offset[first_boundary_i] : mesh.face_offset[first_boundary_i + 1]
    ].copy()
    corrupted = _replace_face(mesh, 0, first_boundary)
    corrupted = _replace_face(corrupted, first_boundary_i, first_internal)

    with pytest.raises(MeshInvariantError, match="internal face 0 normal"):
        check_meshir(corrupted)


def test_rejects_noncontiguous_boundary_patch_ranges() -> None:
    mesh = block_2x2x2_multi_patch()
    patch_offset = np.asarray([0, 8, 4, 24], dtype=np.int32)

    with pytest.raises(MeshInvariantError, match="patch_offset.*monotonic"):
        check_meshir(replace(mesh, patch_offset=patch_offset))


def test_rejects_boundary_faces_not_fully_assigned_to_patches() -> None:
    mesh = block_2x2x2_multi_patch()
    patch_offset = np.asarray([0, 4, 8, 23], dtype=np.int32)

    with pytest.raises(MeshInvariantError, match="number of boundary faces"):
        check_meshir(replace(mesh, patch_offset=patch_offset))


@given(st.integers(min_value=0, max_value=11))
def test_rejects_face_normal_not_pointing_owner_to_neighbour(face_i: int) -> None:
    mesh = block_2x2x2()
    start = int(mesh.face_offset[face_i])
    stop = int(mesh.face_offset[face_i + 1])
    reversed_vertices = mesh.face_verts[start:stop][::-1]

    with pytest.raises(MeshInvariantError, match="normal does not point"):
        check_meshir(_replace_face(mesh, face_i, reversed_vertices))


def test_rejects_cell_with_nonzero_closure_residual() -> None:
    mesh = single_hexahedron()
    open_mesh = _replace_faces(mesh, {1: (4, 5, 6)})

    with pytest.raises(MeshInvariantError, match="cell 0 is not closed"):
        check_meshir(open_mesh)


@given(
    source=st.integers(min_value=0, max_value=35),
    target=st.integers(min_value=0, max_value=35),
)
def test_rejects_duplicate_face(source: int, target: int) -> None:
    if source == target:
        target = (target + 1) % 36
    mesh = block_2x2x2()
    start = int(mesh.face_offset[source])
    stop = int(mesh.face_offset[source + 1])
    source_vertices = mesh.face_verts[start:stop]

    with pytest.raises(MeshInvariantError, match="duplicates another face"):
        check_meshir(_replace_face(mesh, target, source_vertices))


@given(st.tuples(_FINITE, _FINITE, _FINITE))
def test_rejects_unused_point(point: tuple[float, float, float]) -> None:
    mesh = single_hexahedron()
    points = np.vstack([mesh.points, np.asarray(point, dtype=np.float64)])

    with pytest.raises(MeshInvariantError, match="unused points"):
        check_meshir(replace(mesh, points=points))


@given(st.integers(min_value=1, max_value=5))
def test_rejects_nonmonotonic_face_offset(offset_i: int) -> None:
    mesh = single_hexahedron()
    face_offset = mesh.face_offset.copy()
    face_offset[offset_i] = face_offset[offset_i - 1] - 1

    with pytest.raises(MeshInvariantError, match="face_offset must be monotonic"):
        check_meshir(replace(mesh, face_offset=face_offset))


def test_rejects_face_offset_with_wrong_terminal_length() -> None:
    mesh = single_hexahedron()
    face_offset = mesh.face_offset.copy()
    face_offset[-1] -= 1

    with pytest.raises(MeshInvariantError, match=r"face_offset\[-1\]"):
        check_meshir(replace(mesh, face_offset=face_offset))


def test_rejects_float32_core_points() -> None:
    mesh = single_hexahedron()

    with pytest.raises(MeshInvariantError, match="points must have dtype float64"):
        check_meshir(replace(mesh, points=mesh.points.astype(np.float32)))
