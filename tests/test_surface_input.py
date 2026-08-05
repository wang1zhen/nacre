"""Core-side validation of the ``SurfaceInput`` contract, with Gmsh absent.

These tests read the committed ``.npz`` golden files and compare them with
closed-form geometry from ``surface_anchors``. They must never import
``nacre_gmsh``: the same suite runs in a CI job where only the MPL-2.0 core is
installed and ``import gmsh`` is asserted to fail.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from nacre.check import check_surface_input
from nacre.check.surface_input import SurfaceInvariantError
from nacre.contract import SurfaceInput, read_surface_input, write_surface_input
from surface_anchors import (
    ANCHOR_CASES,
    CURVATURE_ATOL,
    CURVATURE_RTOL,
    FLUID_PROBE_EPS,
    GOLDEN_DIR,
    SPHERE_RADIUS,
    TORUS_MAJOR,
    TORUS_MINOR,
    AnchorCase,
    patch_smooth_vertices,
    torus_hoop_curvature,
)

CASE_IDS = [case.name for case in ANCHOR_CASES]


@pytest.fixture(scope="session")
def goldens() -> dict[str, SurfaceInput]:
    loaded: dict[str, SurfaceInput] = {}
    for case in ANCHOR_CASES:
        path = GOLDEN_DIR / f"{case.name}.npz"
        assert path.is_file(), (
            f"missing golden {path}; regenerate with "
            f"'uv run python -m nacre_gmsh goldens tests/goldens'"
        )
        loaded[case.name] = read_surface_input(path)
    return loaded


# --------------------------------------------------------------------------- #
# Contract invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_golden_satisfies_every_contract_invariant(
    case: AnchorCase, goldens: dict[str, SurfaceInput]
) -> None:
    report = check_surface_input(goldens[case.name])
    assert report.n_tris > 0
    assert report.n_feat_corners == case.expected_corners
    assert (report.n_feat_edges > 0) is case.has_feat_edges


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_golden_round_trips_through_npz(
    case: AnchorCase, goldens: dict[str, SurfaceInput], tmp_path
) -> None:
    original = goldens[case.name]
    reloaded = read_surface_input(
        write_surface_input(original, tmp_path / f"{case.name}.npz")
    )
    assert reloaded.patch_names == original.patch_names
    assert reloaded.patch_types == original.patch_types
    assert reloaded.ref_length == original.ref_length
    for field in (
        "points",
        "tris",
        "tri_patch",
        "vert_normal",
        "vert_kappa",
        "vert_kappa_dir",
        "feat_edges",
        "feat_corners",
    ):
        left = getattr(original, field)
        right = getattr(reloaded, field)
        assert right.dtype == left.dtype, field
        np.testing.assert_array_equal(right, left, err_msg=field)


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_core_float_arrays_are_float64(
    case: AnchorCase, goldens: dict[str, SurfaceInput]
) -> None:
    surface = goldens[case.name]
    for field in ("points", "vert_normal", "vert_kappa", "vert_kappa_dir"):
        assert getattr(surface, field).dtype == np.float64, field


# --------------------------------------------------------------------------- #
# Convention B: vert_normal points into the fluid
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_normals_point_into_the_fluid(
    case: AnchorCase, goldens: dict[str, SurfaceInput]
) -> None:
    """Step off every vertex both ways and test membership analytically.

    This is the core-side restatement of the establishing experiment in
    ``nacre.contract``, with an analytic fluid predicate standing in for
    ``gmsh.model.isInside``.
    """

    surface = goldens[case.name]
    step = FLUID_PROBE_EPS * surface.ref_length
    forward = case.in_fluid(surface.points + step * surface.vert_normal)
    backward = case.in_fluid(surface.points - step * surface.vert_normal)

    wrong_way = np.flatnonzero(~forward)
    assert not len(wrong_way), (
        f"{case.name}: vert_normal at {len(wrong_way)} vertices does not enter "
        f"the fluid, first is {surface.points[wrong_way[0]]}"
    )
    into_solid = np.flatnonzero(backward)
    assert not len(into_solid), (
        f"{case.name}: reversing vert_normal at {len(into_solid)} vertices also "
        f"stays in the fluid, first is {surface.points[into_solid[0]]}"
    )


def test_sphere_and_box_normals_oppose_yet_both_enter_the_fluid(
    goldens: dict[str, SurfaceInput],
) -> None:
    """The external-flow assertion the normal convention is built on."""

    surface = goldens["sphere-external"]
    sphere = patch_smooth_vertices(surface, "sphere")
    farfield = patch_smooth_vertices(surface, "farfield")

    # Both patches are radial about the origin, so the sign of n . p separates
    # "away from the origin" from "toward it".
    sphere_radial = np.einsum(
        "ij,ij->i", surface.vert_normal[sphere], surface.points[sphere]
    )
    farfield_radial = np.einsum(
        "ij,ij->i", surface.vert_normal[farfield], surface.points[farfield]
    )
    assert np.all(sphere_radial > 0.0), "sphere normals should point outward"
    assert np.all(farfield_radial < 0.0), "far-field normals should point inward"


# --------------------------------------------------------------------------- #
# Convention A and the analytic curvature anchors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_curvature_matches_analytic_anchor(
    case: AnchorCase, goldens: dict[str, SurfaceInput]
) -> None:
    surface = goldens[case.name]
    for patch_name, expected_curvature in case.curvature.items():
        vertices = patch_smooth_vertices(surface, patch_name)
        expected = expected_curvature(surface.points[vertices])
        np.testing.assert_allclose(
            surface.vert_kappa[:, 0][vertices],
            expected[:, 0],
            rtol=CURVATURE_RTOL,
            atol=CURVATURE_ATOL,
            err_msg=f"{case.name}/{patch_name}: k1",
        )
        np.testing.assert_allclose(
            surface.vert_kappa[:, 1][vertices],
            expected[:, 1],
            rtol=CURVATURE_RTOL,
            atol=CURVATURE_ATOL,
            err_msg=f"{case.name}/{patch_name}: k2",
        )


def test_solid_sphere_and_spherical_chamber_have_opposite_curvature_sign(
    goldens: dict[str, SurfaceInput],
) -> None:
    """The experiment that fixes the curvature sign convention.

    Gmsh reports ``-1/R`` for the same sphere whichever side the material is
    on, so if the bake did not fold in the fluid-side orientation these two
    cases would come out identical and convex could not be told from concave.
    """

    convex = goldens["sphere-external"]
    concave = goldens["sphere-chamber"]
    outside = patch_smooth_vertices(convex, "sphere")
    inside = patch_smooth_vertices(concave, "chamber")

    np.testing.assert_allclose(
        convex.vert_kappa[:, 0][outside], 1.0 / SPHERE_RADIUS, rtol=CURVATURE_RTOL
    )
    np.testing.assert_allclose(
        concave.vert_kappa[:, 0][inside], -1.0 / SPHERE_RADIUS, rtol=CURVATURE_RTOL
    )
    assert np.all(convex.vert_kappa[:, 0][outside] > 0.0)
    assert np.all(concave.vert_kappa[:, 0][inside] < 0.0)
    np.testing.assert_allclose(
        convex.vert_kappa[:, 0][outside][0], -concave.vert_kappa[:, 0][inside][0], rtol=CURVATURE_RTOL
    )


def test_cylinder_distinguishes_k1_from_k2_and_from_averaged_curvature(
    goldens: dict[str, SurfaceInput],
) -> None:
    """A cylinder rejects mean and Gaussian curvature, and rejects a swap."""

    surface = goldens["cylinder-external"]
    side = patch_smooth_vertices(surface, "cylinder")
    k1 = surface.vert_kappa[:, 0][side]
    k2 = surface.vert_kappa[:, 1][side]
    radius = 1.5

    np.testing.assert_allclose(k1, 1.0 / radius, rtol=CURVATURE_RTOL)
    np.testing.assert_allclose(k2, 0.0, atol=CURVATURE_ATOL)
    # Mean curvature would be 1/(2R) in both slots and Gaussian curvature 0.
    assert not np.allclose(k1, 0.5 / radius)
    assert not np.allclose(k1, 0.0)


def test_torus_curvature_varies_and_changes_sign(
    goldens: dict[str, SurfaceInput],
) -> None:
    """A constant-returning producer passes sphere and cylinder but not this."""

    surface = goldens["torus-external"]
    torus = patch_smooth_vertices(surface, "torus")
    k1 = surface.vert_kappa[:, 0][torus]
    k2 = surface.vert_kappa[:, 1][torus]

    np.testing.assert_allclose(k1, 1.0 / TORUS_MINOR, rtol=CURVATURE_RTOL)
    assert k2.max() - k2.min() > 0.5, "k2 must vary over the torus"

    outer = int(np.argmin(np.linalg.norm(
        surface.points[torus] - np.array([TORUS_MAJOR + TORUS_MINOR, 0.0, 0.0]), axis=1
    )))
    throat = int(np.argmin(np.linalg.norm(
        surface.points[torus] - np.array([TORUS_MAJOR - TORUS_MINOR, 0.0, 0.0]), axis=1
    )))
    outer_point = surface.points[torus][outer]
    throat_point = surface.points[torus][throat]

    np.testing.assert_allclose(
        k2[outer],
        torus_hoop_curvature(
            (np.hypot(outer_point[0], outer_point[1]) - TORUS_MAJOR) / TORUS_MINOR
        ),
        rtol=CURVATURE_RTOL,
    )
    np.testing.assert_allclose(
        k2[throat],
        torus_hoop_curvature(
            (np.hypot(throat_point[0], throat_point[1]) - TORUS_MAJOR) / TORUS_MINOR
        ),
        rtol=CURVATURE_RTOL,
    )
    assert k2.max() > 0.0 and k2.min() < 0.0, "the torus must be a saddle somewhere"
    np.testing.assert_allclose(
        k2.max(), 1.0 / (TORUS_MAJOR + TORUS_MINOR), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        k2.min(), -1.0 / (TORUS_MAJOR - TORUS_MINOR), rtol=1.0e-3
    )


def test_concave_walls_are_exactly_where_layers_would_collide(
    goldens: dict[str, SurfaceInput],
) -> None:
    """Sanity-check the sign against the physics the convention exists for."""

    pipe = goldens["cylinder-pipe"]
    inner = patch_smooth_vertices(pipe, "pipe")
    assert np.all(pipe.vert_kappa[:, 1][inner] < 0.0), "a pipe wall wraps around the fluid"
    np.testing.assert_allclose(pipe.vert_kappa[:, 0][inner], 0.0, atol=CURVATURE_ATOL)

    chamber = goldens["sphere-chamber"]
    assert np.all(chamber.vert_kappa[:, 0] < 0.0)
    assert check_surface_input(chamber).max_kappa < 0.0


# --------------------------------------------------------------------------- #
# Checker behaviour on deliberately corrupted data
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample(goldens: dict[str, SurfaceInput]) -> SurfaceInput:
    return goldens["box"]


@cache
def _box_golden() -> SurfaceInput:
    """Load the box golden outside the fixture system.

    Hypothesis renders every argument of a failing example, and a whole
    ``SurfaceInput`` passed in as a fixture produces a 46 kB repr.
    """

    return read_surface_input(GOLDEN_DIR / "box.npz")


def _corrupt(surface: SurfaceInput, **overrides: object) -> SurfaceInput:
    return replace(surface, **overrides)


def test_flipped_triangle_is_rejected(sample: SurfaceInput) -> None:
    tris = sample.tris.copy()
    tris[3] = tris[3][[0, 2, 1]]
    with pytest.raises(SurfaceInvariantError, match="winding is inconsistent"):
        check_surface_input(_corrupt(sample, tris=tris))


def test_globally_reversed_winding_is_rejected(sample: SurfaceInput) -> None:
    """Reversing every triangle keeps the topology legal but inverts the normal."""

    with pytest.raises(SurfaceInvariantError, match="wound against vert_normal"):
        check_surface_input(_corrupt(sample, tris=sample.tris[:, [0, 2, 1]].copy()))


def test_flipped_vertex_normal_is_rejected(sample: SurfaceInput) -> None:
    normal = sample.vert_normal.copy()
    normal[10] = -normal[10]
    with pytest.raises(SurfaceInvariantError, match="wound against vert_normal"):
        check_surface_input(_corrupt(sample, vert_normal=normal))


def test_non_unit_normal_is_rejected(sample: SurfaceInput) -> None:
    normal = sample.vert_normal.copy()
    normal[7] *= 1.5
    with pytest.raises(SurfaceInvariantError, match="not unit"):
        check_surface_input(_corrupt(sample, vert_normal=normal))


def test_unsorted_curvatures_are_rejected(sample: SurfaceInput) -> None:
    kappa = sample.vert_kappa.copy()
    kappa[5, 0] = -1.0
    with pytest.raises(SurfaceInvariantError, match="not sorted descending"):
        check_surface_input(_corrupt(sample, vert_kappa=kappa))


def test_hole_in_the_surface_is_rejected(sample: SurfaceInput) -> None:
    keep = np.ones(sample.n_tris, dtype=bool)
    keep[0] = False
    with pytest.raises(SurfaceInvariantError, match="not watertight"):
        check_surface_input(
            _corrupt(
                sample, tris=sample.tris[keep].copy(), tri_patch=sample.tri_patch[keep]
            )
        )


def test_hole_is_accepted_when_closure_is_not_required(sample: SurfaceInput) -> None:
    keep = np.ones(sample.n_tris, dtype=bool)
    keep[0] = False
    open_surface = _corrupt(
        sample, tris=sample.tris[keep].copy(), tri_patch=sample.tri_patch[keep]
    )
    assert check_surface_input(open_surface, require_closed=False).n_tris == int(
        keep.sum()
    )


def test_duplicate_triangle_is_rejected(sample: SurfaceInput) -> None:
    tris = np.concatenate([sample.tris, sample.tris[:1]], axis=0)
    tri_patch = np.concatenate([sample.tri_patch, sample.tri_patch[:1]])
    with pytest.raises(SurfaceInvariantError, match="duplicate triangle"):
        check_surface_input(_corrupt(sample, tris=tris, tri_patch=tri_patch))


def test_out_of_range_patch_index_is_rejected(sample: SurfaceInput) -> None:
    tri_patch = sample.tri_patch.copy()
    tri_patch[0] = 9
    with pytest.raises(SurfaceInvariantError, match="out-of-range patch index"):
        check_surface_input(_corrupt(sample, tri_patch=tri_patch))


def test_dropped_corner_is_rejected(sample: SurfaceInput) -> None:
    """A box corner joins three feature edges and cannot be demoted."""

    assert len(sample.feat_corners), "the box golden must have corners"
    with pytest.raises(SurfaceInvariantError, match="not listed in feat_corners"):
        check_surface_input(_corrupt(sample, feat_corners=sample.feat_corners[1:]))


def test_invented_corner_is_rejected(sample: SurfaceInput) -> None:
    smooth = np.setdiff1d(np.arange(sample.n_verts), sample.feat_edges.reshape(-1))
    corners = np.append(sample.feat_corners, np.int32(smooth[0]))
    with pytest.raises(SurfaceInvariantError, match="lies on no feature edge"):
        check_surface_input(_corrupt(sample, feat_corners=corners))


def test_invented_feature_edge_is_rejected(sample: SurfaceInput) -> None:
    edges = np.concatenate(
        [sample.feat_edges, np.array([[0, sample.n_verts - 1]], dtype=np.int32)]
    )
    with pytest.raises(SurfaceInvariantError, match="not an edge of any triangle"):
        check_surface_input(_corrupt(sample, feat_edges=edges))


def test_nan_coordinates_are_rejected(sample: SurfaceInput) -> None:
    points = sample.points.copy()
    points[2, 1] = np.nan
    with pytest.raises(SurfaceInvariantError, match="NaN or infinity"):
        check_surface_input(_corrupt(sample, points=points))


def test_non_positive_ref_length_is_rejected(sample: SurfaceInput) -> None:
    with pytest.raises(SurfaceInvariantError, match="finite and positive"):
        check_surface_input(_corrupt(sample, ref_length=0.0))


def test_float32_arrays_are_rejected(sample: SurfaceInput) -> None:
    with pytest.raises(SurfaceInvariantError, match="dtype float64"):
        check_surface_input(_corrupt(sample, points=sample.points.astype(np.float32)))


def test_contract_version_mismatch_is_rejected(
    sample: SurfaceInput, tmp_path
) -> None:
    path = write_surface_input(sample, tmp_path / "versioned.npz")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["contract_version"] = np.asarray(99, dtype=np.int32)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="contract version 99"):
        read_surface_input(path)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_any_single_flipped_triangle_is_caught(seed: int) -> None:
    """No triangle in the golden is exempt from the winding invariant."""

    surface = _box_golden()
    tris = surface.tris.copy()
    triangle = seed % len(tris)
    tris[triangle] = tris[triangle][[1, 0, 2]]
    with pytest.raises(SurfaceInvariantError):
        check_surface_input(_corrupt(surface, tris=tris))


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_any_single_reversed_normal_is_caught(seed: int) -> None:
    """No vertex is exempt from the into-fluid normal invariant either."""

    surface = _box_golden()
    normal = surface.vert_normal.copy()
    normal[seed % len(normal)] *= -1.0
    with pytest.raises(SurfaceInvariantError):
        check_surface_input(_corrupt(surface, vert_normal=normal))
