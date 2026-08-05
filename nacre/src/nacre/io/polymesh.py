"""Write-only OpenFOAM ``polyMesh`` serialization."""

from __future__ import annotations

import re
from pathlib import Path

from nacre.check import check_meshir
from nacre.meshir import PolyMeshIR


_FOAM_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_FOOTER = "\n// ************************************************************************* //\n"


def _header(class_name: str, object_name: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        '    arch        "LSB;label=32;scalar=64";\n'
        f"    class       {class_name};\n"
        '    location    "constant/polyMesh";\n'
        f"    object      {object_name};\n"
        "}\n"
    )


def _write(path: Path, contents: str) -> None:
    path.write_text(contents + _FOOTER, encoding="ascii")


def write_polymesh(mesh: PolyMeshIR, mesh_dir: str | Path) -> Path:
    """Validate and write ``mesh`` as an ASCII OpenFOAM ``polyMesh``."""

    check_meshir(mesh)
    destination = Path(mesh_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for name, kind in zip(mesh.patch_names, mesh.patch_types):
        if not _FOAM_WORD.fullmatch(name):
            raise ValueError(f"patch name is not a valid OpenFOAM word: {name!r}")
        if not _FOAM_WORD.fullmatch(kind):
            raise ValueError(f"patch type is not a valid OpenFOAM word: {kind!r}")

    point_lines = "".join(
        f"({x:.17g} {y:.17g} {z:.17g})\n" for x, y, z in mesh.points
    )
    _write(
        destination / "points",
        _header("vectorField", "points")
        + f"\n{len(mesh.points)}\n(\n{point_lines})\n",
    )

    face_lines: list[str] = []
    for face_i in range(mesh.n_faces):
        start = int(mesh.face_offset[face_i])
        stop = int(mesh.face_offset[face_i + 1])
        vertices = mesh.face_verts[start:stop]
        labels = " ".join(str(int(vertex)) for vertex in vertices)
        face_lines.append(f"{len(vertices)}({labels})\n")
    _write(
        destination / "faces",
        _header("faceList", "faces")
        + f"\n{mesh.n_faces}\n(\n{''.join(face_lines)})\n",
    )

    owner_lines = "".join(f"{int(cell)}\n" for cell in mesh.owner)
    _write(
        destination / "owner",
        _header("labelList", "owner")
        + f"\n{mesh.n_faces}\n(\n{owner_lines})\n",
    )

    neighbour_lines = "".join(f"{int(cell)}\n" for cell in mesh.neighbour)
    _write(
        destination / "neighbour",
        _header("labelList", "neighbour")
        + f"\n{mesh.n_internal_faces}\n(\n{neighbour_lines})\n",
    )

    boundary_entries: list[str] = []
    for patch_i, (name, kind) in enumerate(
        zip(mesh.patch_names, mesh.patch_types)
    ):
        relative_start = int(mesh.patch_offset[patch_i])
        relative_stop = int(mesh.patch_offset[patch_i + 1])
        boundary_entries.append(
            f"    {name}\n"
            "    {\n"
            f"        type        {kind};\n"
            f"        nFaces      {relative_stop - relative_start};\n"
            f"        startFace   {mesh.n_internal_faces + relative_start};\n"
            "    }\n"
        )
    _write(
        destination / "boundary",
        _header("polyBoundaryMesh", "boundary")
        + f"\n{len(mesh.patch_names)}\n(\n"
        + "".join(boundary_entries)
        + ")\n",
    )

    return destination


__all__ = ["write_polymesh"]
