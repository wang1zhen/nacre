"""Live Gmsh bake tests, including the two convention-establishing experiments.

This module is the only test file that imports ``nacre_gmsh``. It is skipped
when the GPL distribution is absent, which is exactly the situation the
core-only CI job creates.

Two of these tests exist to record *why* the conventions in ``nacre.contract``
are what they are. They query Gmsh directly rather than through the bake, so
they fail if a future Gmsh release changes what it reports -- which is the point,
because that change would silently invert convex and concave otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from nacre.check import check_surface_input
from nacre.contract import read_surface_input, write_surface_input
from surface_anchors import (
    ANCHOR_CASES,
    CURVATURE_RTOL,
    GOLDEN_DIR,
    SPHERE_RADIUS,
    AnchorCase,
    patch_smooth_vertices,
)

try:
    import gmsh
    import nacre_gmsh
    from nacre_gmsh import BakeSettings, bake_step, fluid_normal_sign, gmsh_session
    from nacre_gmsh.corpus import CASES_BY_NAME, bake_case
except ImportError as error:
    pytest.skip(
        f"the nacre-gmsh distribution is not installed: {error}",
        allow_module_level=True,
    )
except OSError as error:
    # The gmsh wheel bundles libgmsh.so but not the system libraries it links
    # against, so a plain ImportError is not the only way this can fail. Note
    # that OSError is not an ImportError, which is why pytest.importorskip
    # cannot be used here: it would abort collection instead of skipping.
    pytest.skip(
        "the gmsh wheel is installed but its shared library will not load "
        f"({error}); on Debian or Ubuntu install libglu1-mesa",
        allow_module_level=True,
    )

CASE_IDS = [case.name for case in ANCHOR_CASES]


@pytest.fixture
def session() -> object:
    with gmsh_session():
        yield None


# --------------------------------------------------------------------------- #
# The establishing experiments
# --------------------------------------------------------------------------- #


def test_gmsh_curvature_alone_cannot_tell_convex_from_concave(session) -> None:
    """Why the curvature sign has to be recovered from the fluid side.

    Gmsh measures curvature against the parametric surface normal, which does
    not move when the material swaps sides. A standalone sphere -- which is both
    the solid sphere of the external-flow case and the chamber of the concave
    case, depending only on where the fluid is -- and the same sphere carved out
    of a block therefore report the identical value. A producer that passed the
    number through unchanged would make every wall read as convex.
    """

    reported: dict[str, float] = {}

    gmsh.model.add("solid")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, SPHERE_RADIUS)
    gmsh.model.occ.synchronize()
    reported["solid sphere"] = _sphere_curvature_at_probe(1)
    gmsh.model.remove()

    gmsh.model.add("cavity")
    _cut_sphere_from_block()
    spherical = [
        tag
        for _, tag in gmsh.model.getEntities(2)
        if gmsh.model.getType(2, tag) == "Sphere"
    ]
    assert len(spherical) == 1
    reported["sphere cut from a block"] = _sphere_curvature_at_probe(spherical[0])
    gmsh.model.remove()

    assert set(reported.values()) == {-1.0 / SPHERE_RADIUS}, reported


def test_gmsh_normal_ignores_face_orientation_in_the_solid(session) -> None:
    """Why the into-fluid direction cannot be read off ``getNormal``.

    The same geometric sphere reports the same outward parametric normal
    whether it is the solid or the void, and ``getBoundary(oriented=True)``
    multiplies a box's parametric normals into a single direction per axis
    rather than into outward normals. Neither is an inside/outside flag.
    """

    gmsh.model.add("solid")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, SPHERE_RADIUS)
    gmsh.model.occ.synchronize()
    standalone = _sphere_normal_at_probe(1)
    gmsh.model.remove()

    gmsh.model.add("cavity")
    _cut_sphere_from_block()
    spherical = [
        tag
        for _, tag in gmsh.model.getEntities(2)
        if gmsh.model.getType(2, tag) == "Sphere"
    ]
    carved = _sphere_normal_at_probe(spherical[0])

    np.testing.assert_allclose(standalone, [1.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(carved, standalone, atol=1.0e-12)

    # getBoundary's signs are not outward flags: they fold the box's six
    # parametric normals onto three axis directions.
    oriented = gmsh.model.getBoundary(gmsh.model.getEntities(3), oriented=True)
    folded = set()
    for _, signed in oriented:
        tag = abs(signed)
        if gmsh.model.getType(2, tag) != "Plane":
            continue
        _, uv = gmsh.model.getClosestPoint(2, tag, [0.0, 0.0, 0.0])
        normal = np.asarray(gmsh.model.getNormal(tag, list(uv)))
        folded.add(tuple(np.round(np.sign(signed) * normal + 0.0, 12)))
    assert len(folded) == 3, folded


def test_external_flow_normals_oppose_absolutely_yet_agree_on_the_fluid(
    session,
) -> None:
    """A sphere inside a box: opposite absolute directions, one fluid side.

    Run against Gmsh's own ``isInside`` rather than the analytic predicate used
    by the core suite, so the two checks are independent.
    """

    gmsh.model.add("external")
    volume = _cut_sphere_from_block()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4)
    gmsh.model.mesh.generate(2)

    signs: dict[str, list[int]] = {"Sphere": [], "Plane": []}
    for _, tag in gmsh.model.getEntities(2):
        _, coords, _ = gmsh.model.mesh.getNodes(2, tag, True, True)
        probe = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        sign = fluid_normal_sign(tag, [volume], probe, 17.32, 8)
        signs[gmsh.model.getType(2, tag)].append(sign)

    assert signs["Sphere"] == [1], "the sphere's outward normal enters the fluid"
    assert signs["Plane"] == [-1] * 6, "each box plane's inward normal enters it"


def _cut_sphere_from_block() -> int:
    """Build ``box \\ sphere`` in the current model and return its volume tag."""

    box = gmsh.model.occ.addBox(-5.0, -5.0, -5.0, 10.0, 10.0, 10.0)
    sphere = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, SPHERE_RADIUS)
    volumes, _ = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
    gmsh.model.occ.synchronize()
    assert len(volumes) == 1, volumes
    return int(volumes[0][1])


def _sphere_curvature_at_probe(tag: int) -> float:
    _, uv = gmsh.model.getClosestPoint(2, tag, [SPHERE_RADIUS, 0.0, 0.0])
    maximum, minimum, _, _ = gmsh.model.getPrincipalCurvatures(tag, list(uv))
    assert float(maximum[0]) == float(minimum[0])
    return float(maximum[0])


def _sphere_normal_at_probe(tag: int) -> np.ndarray:
    _, uv = gmsh.model.getClosestPoint(2, tag, [SPHERE_RADIUS, 0.0, 0.0])
    return np.asarray(gmsh.model.getNormal(tag, list(uv)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# A live bake reproduces the committed goldens and the analytic anchors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_live_bake_reproduces_the_committed_golden(
    case: AnchorCase, session
) -> None:
    """Guard against a golden drifting away from the code that produced it."""

    baked = bake_case(CASES_BY_NAME[case.name])
    committed = read_surface_input(GOLDEN_DIR / f"{case.name}.npz")

    assert baked.patch_names == committed.patch_names
    assert baked.patch_types == committed.patch_types
    assert baked.n_verts == committed.n_verts
    assert baked.n_tris == committed.n_tris
    np.testing.assert_allclose(baked.points, committed.points, atol=1.0e-12)
    np.testing.assert_array_equal(baked.tris, committed.tris)
    np.testing.assert_allclose(baked.vert_normal, committed.vert_normal, atol=1.0e-12)
    np.testing.assert_allclose(baked.vert_kappa, committed.vert_kappa, atol=1.0e-12)
    np.testing.assert_allclose(baked.vert_kappa_dir, committed.vert_kappa_dir, atol=1.0e-12)
    np.testing.assert_array_equal(baked.feat_edges, committed.feat_edges)
    np.testing.assert_array_equal(baked.feat_corners, committed.feat_corners)


@pytest.mark.parametrize("case", ANCHOR_CASES, ids=CASE_IDS)
def test_live_bake_matches_analytic_curvature(case: AnchorCase, session) -> None:
    surface = bake_case(CASES_BY_NAME[case.name])
    check_surface_input(surface)
    for patch_name, expected_curvature in case.curvature.items():
        vertices = patch_smooth_vertices(surface, patch_name)
        expected = expected_curvature(surface.points[vertices])
        np.testing.assert_allclose(
            surface.vert_kappa[:, 0][vertices],
            expected[:, 0],
            rtol=CURVATURE_RTOL,
            atol=1.0e-12,
            err_msg=f"{case.name}/{patch_name}: k1",
        )
        np.testing.assert_allclose(
            surface.vert_kappa[:, 1][vertices],
            expected[:, 1],
            rtol=CURVATURE_RTOL,
            atol=1.0e-12,
            err_msg=f"{case.name}/{patch_name}: k2",
        )


@pytest.mark.parametrize("radius", [0.05, 0.5, 5.0])
def test_sphere_curvature_holds_across_two_orders_of_magnitude(
    radius: float, session
) -> None:
    """The M1 exit criterion on analytic spheres, stated in the roadmap.

    Three radii spanning two orders of magnitude, both absolute principal
    curvatures within 1% of ``1/R``. The measured error is far tighter, but the
    gate is the roadmap's.
    """

    gmsh.model.add(f"sphere-{radius}")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
    gmsh.model.occ.synchronize()
    surface = nacre_gmsh.bake_current_model(BakeSettings(size_max=radius / 3.0))
    check_surface_input(surface)

    np.testing.assert_allclose(
        np.abs(surface.vert_kappa), 1.0 / radius, rtol=0.01
    )
    # The fluid fills the chamber, so both curvatures are negative.
    assert np.all(surface.vert_kappa < 0.0)


def test_bake_refuses_a_model_without_a_fluid_volume(session) -> None:
    """Without a volume there is no fluid side, so a bake would be a guess."""

    gmsh.model.add("surface-only")
    gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    with pytest.raises(nacre_gmsh.BakeError, match="no 3D entity"):
        nacre_gmsh.bake_current_model()


# --------------------------------------------------------------------------- #
# The STEP path
# --------------------------------------------------------------------------- #


def test_step_round_trip_bakes_to_the_same_geometry(session, tmp_path) -> None:
    """The documented entry point: a CAD file on disk in, a ``.npz`` out."""

    gmsh.model.add("writer")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, SPHERE_RADIUS)
    gmsh.model.occ.synchronize()
    step = tmp_path / "chamber.step"
    gmsh.write(str(step))
    gmsh.model.remove()

    surface = bake_step(step, BakeSettings(size_max=0.6))
    report = check_surface_input(surface)
    assert report.n_tris > 0
    np.testing.assert_allclose(
        surface.vert_kappa[:, 0], -1.0 / SPHERE_RADIUS, rtol=CURVATURE_RTOL
    )
    written = write_surface_input(surface, tmp_path / "chamber.npz")
    assert read_surface_input(written).n_tris == surface.n_tris


def test_patch_names_come_from_named_cad_faces(session) -> None:
    """Named faces reach patch names through ``gmsh.model.getEntityName``.

    Gmsh's own STEP writer does not emit face labels, so the naming path is
    exercised by setting the entity names directly; importing a STEP that
    carries labels populates the very same resolver, with
    ``Geometry.OCCImportLabels`` enabled by ``gmsh_session``.
    """

    gmsh.model.add("named")
    gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.5)
    gmsh.model.occ.synchronize()
    for _, tag in gmsh.model.getEntities(2):
        label = "pipe_wall" if gmsh.model.getType(2, tag) == "Cylinder" else "pipe end"
        gmsh.model.setEntityName(2, tag, f"Shapes/Solid 1/{label}")

    surface = nacre_gmsh.bake_current_model(
        BakeSettings(size_max=0.8, patch_types={"pipe_wall": "wall"})
    )
    assert set(surface.patch_names) == {"pipe_wall", "pipe_end"}
    assert surface.patch_types == ("wall",) * len(surface.patch_names)


def test_physical_group_names_win_over_cad_labels(session) -> None:
    gmsh.model.add("grouped")
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    tags = [tag for _, tag in gmsh.model.getEntities(2)]
    for tag in tags:
        gmsh.model.setEntityName(2, tag, "ignored")
    gmsh.model.addPhysicalGroup(2, tags[:1], name="inlet")
    gmsh.model.addPhysicalGroup(2, tags[1:], name="walls")

    surface = nacre_gmsh.bake_current_model(BakeSettings(size_max=0.5))
    assert set(surface.patch_names) == {"inlet", "walls"}
