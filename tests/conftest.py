from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pytest

from fixtures import FIXTURE_SPECS, FixtureSpec
from nacre.check import MeshIRCheckResult, check_meshir
from nacre.checkmesh import CheckMeshResult, run_checkmesh
from nacre.io import write_polymesh
from nacre.meshir import PolyMeshIR


@dataclass(frozen=True)
class EvaluatedFixture:
    spec: FixtureSpec
    mesh: PolyMeshIR
    structural: MeshIRCheckResult
    external: CheckMeshResult
    wall_time_s: float


@pytest.fixture(scope="session")
def evaluated_fixtures(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, EvaluatedFixture]:
    root = tmp_path_factory.mktemp("evaluated-fixtures")
    evaluated: dict[str, EvaluatedFixture] = {}
    for spec in FIXTURE_SPECS:
        started = perf_counter()
        mesh = spec.factory()
        structural = check_meshir(mesh)
        mesh_dir = write_polymesh(mesh, Path(root) / spec.name / "polyMesh")
        external = run_checkmesh(mesh_dir, all_geometry=spec.all_geometry)
        wall_time_s = perf_counter() - started
        evaluated[spec.name] = EvaluatedFixture(
            spec=spec,
            mesh=mesh,
            structural=structural,
            external=external,
            wall_time_s=wall_time_s,
        )
    return evaluated
