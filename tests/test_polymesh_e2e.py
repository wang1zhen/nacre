import numpy as np
import pytest

from fixtures import (
    FIXTURE_SPECS,
    FixtureSpec,
    block_2x2x2,
    block_2x2x2_multi_patch,
    single_hexahedron,
    two_cells_with_hanging_node,
    two_cells_with_warped_face,
)
from conftest import EvaluatedFixture


@pytest.mark.parametrize(
    "spec",
    FIXTURE_SPECS,
    ids=lambda spec: spec.name,
)
def test_fixture_exports_and_passes_checkmesh(
    spec: FixtureSpec,
    evaluated_fixtures: dict[str, EvaluatedFixture],
) -> None:
    evaluated = evaluated_fixtures[spec.name]
    mesh = evaluated.mesh
    structural = evaluated.structural
    external = evaluated.external

    assert structural.n_faces == mesh.n_faces
    assert external.n_errors == 0, external.raw_output
    assert np.isfinite(external.max_non_ortho), external.raw_output
    assert np.isfinite(external.max_skewness), external.raw_output
    assert np.isfinite(external.total_volume), external.raw_output
    relative_volume_error = (
        abs(structural.total_volume - external.total_volume)
        / abs(external.total_volume)
    )
    assert relative_volume_error <= 1.0e-12, (
        f"computed volume {structural.total_volume:.17g}, "
        f"checkMesh volume {external.total_volume:.17g}, "
        f"relative error {relative_volume_error:.17g}"
    )


def test_hanging_node_is_on_pentagonal_internal_face() -> None:
    mesh = two_cells_with_hanging_node()

    first_face_size = int(mesh.face_offset[1] - mesh.face_offset[0])

    assert mesh.n_internal_faces == 1
    assert first_face_size == 5
    assert 12 in mesh.face_verts[mesh.face_offset[0] : mesh.face_offset[1]]


def test_warped_internal_face_is_not_coplanar() -> None:
    mesh = two_cells_with_warped_face()
    indices = mesh.face_verts[mesh.face_offset[0] : mesh.face_offset[1]]
    vertices = mesh.points[indices]
    signed_six_volume = np.dot(
        vertices[3] - vertices[0],
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]),
    )

    assert abs(signed_six_volume) > 0.0


def test_multi_patch_fixture_has_contiguous_named_ranges() -> None:
    mesh = block_2x2x2_multi_patch()

    assert mesh.patch_names == ("inlet", "outlet", "walls")
    np.testing.assert_array_equal(mesh.patch_offset, [0, 4, 8, 24])
