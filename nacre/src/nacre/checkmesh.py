"""Run OpenFOAM's ``checkMesh`` and parse its textual verdict."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckMeshResult:
    """The machine-readable subset of an OpenFOAM ``checkMesh`` report."""

    n_errors: int
    max_non_ortho: float
    max_skewness: float
    max_aspect_ratio: float
    total_volume: float
    raw_output: str


_FAILED_CHECKS = re.compile(r"Failed\s+(\d+)\s+mesh checks?\.", re.IGNORECASE)
_NON_ORTHO = re.compile(
    r"Mesh non-orthogonality Max:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_SKEWNESS = re.compile(
    r"Max skewness\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_TOTAL_VOLUME = re.compile(
    r"Total volume\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_ASPECT_RATIO = re.compile(
    r"Max aspect ratio\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
def _checkmesh_command() -> list[str]:
    direct = shutil.which("checkMesh")
    if direct is not None:
        return [direct]

    wrapper = shutil.which("openfoam2606")
    if wrapper is not None:
        return [wrapper, "checkMesh"]

    raise FileNotFoundError(
        "checkMesh is unavailable: neither checkMesh nor openfoam2606 is on PATH"
    )


def _docker_checkmesh_command(image: str, case: Path) -> list[str]:
    docker = shutil.which("docker")
    if docker is None:
        raise FileNotFoundError(
            "NACRE_CHECKMESH_IMAGE is set, but docker is not on PATH"
        )
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        raise RuntimeError("Docker-backed checkMesh currently requires a POSIX host")
    return [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--volume",
        f"{case}:/case",
        image,
        "checkMesh",
    ]


def _parse_result(stdout: str, stderr: str, returncode: int) -> CheckMeshResult:
    failed = _FAILED_CHECKS.search(stdout)
    if failed is not None:
        n_errors = int(failed.group(1))
    elif "Mesh OK." in stdout:
        n_errors = 0
    else:
        details = stdout
        if stderr:
            details += "\n--- checkMesh stderr ---\n" + stderr
        raise RuntimeError(
            "checkMesh did not emit a complete mesh verdict; "
            f"process exit code was {returncode}\n{details}"
        )

    non_ortho = _NON_ORTHO.search(stdout)
    skewness = _SKEWNESS.search(stdout)
    total_volume = _TOTAL_VOLUME.search(stdout)
    aspect_ratio = _ASPECT_RATIO.search(stdout)
    max_non_ortho = (
        float(non_ortho.group(1))
        if non_ortho
        else 0.0
        if "Non-orthogonality check OK." in stdout
        else float("nan")
    )

    raw_output = stdout
    if stderr:
        raw_output += "\n--- checkMesh stderr ---\n" + stderr
    if returncode != 0:
        raw_output += f"\n--- checkMesh process exit code: {returncode} ---\n"

    return CheckMeshResult(
        n_errors=n_errors,
        max_non_ortho=max_non_ortho,
        max_skewness=float(skewness.group(1)) if skewness else float("nan"),
        max_aspect_ratio=(
            float(aspect_ratio.group(1)) if aspect_ratio else float("nan")
        ),
        total_volume=(
            float(total_volume.group(1)) if total_volume else float("nan")
        ),
        raw_output=raw_output,
    )


def run_checkmesh(
    mesh_dir: str | Path, *, all_geometry: bool = False
) -> CheckMeshResult:
    """Run local OpenFOAM ``checkMesh`` against a ``polyMesh`` directory.

    The OpenFOAM process exit code is deliberately not used as the mesh verdict.
    OpenFOAM's stdout summary is the source of ``n_errors``.
    """

    source = Path(mesh_dir).resolve(strict=True)
    required = ("points", "faces", "owner", "neighbour", "boundary")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source} is not a polyMesh directory; missing: {', '.join(missing)}"
        )

    with tempfile.TemporaryDirectory(prefix="nacre-checkmesh-") as temporary:
        case = Path(temporary)
        (case / "system").mkdir()
        shutil.copytree(source, case / "constant" / "polyMesh")
        (case / "system" / "controlDict").write_text(
            "FoamFile\n"
            "{\n"
            "    format ascii;\n"
            "    class dictionary;\n"
            "    object controlDict;\n"
            "}\n"
            "application checkMesh;\n"
            "startFrom startTime;\n"
            "startTime 0;\n"
            "stopAt endTime;\n"
            "endTime 0;\n"
            "deltaT 1;\n"
            "writeControl timeStep;\n"
            "writeInterval 1;\n"
            "writePrecision 16;\n",
            encoding="ascii",
        )
        (case / "system" / "fvSchemes").write_text(
            "FoamFile\n"
            "{\n"
            "    format ascii;\n"
            "    class dictionary;\n"
            "    object fvSchemes;\n"
            "}\n"
            "ddtSchemes { default steadyState; }\n"
            "gradSchemes { default Gauss linear; }\n"
            "divSchemes { default none; }\n"
            "laplacianSchemes { default Gauss linear corrected; }\n"
            "interpolationSchemes { default linear; }\n"
            "snGradSchemes { default corrected; }\n",
            encoding="ascii",
        )
        (case / "system" / "fvSolution").write_text(
            "FoamFile\n"
            "{\n"
            "    format ascii;\n"
            "    class dictionary;\n"
            "    object fvSolution;\n"
            "}\n"
            "solvers {}\n",
            encoding="ascii",
        )

        image = os.environ.get("NACRE_CHECKMESH_IMAGE")
        if image:
            command = [
                *_docker_checkmesh_command(image, case),
                "-case",
                "/case",
                "-constant",
            ]
        else:
            command = [
                *_checkmesh_command(),
                "-case",
                str(case),
                "-constant",
            ]
        if all_geometry:
            command.append("-allGeometry")

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    return _parse_result(completed.stdout, completed.stderr, completed.returncode)
