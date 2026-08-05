from pathlib import Path

from nacre.checkmesh import run_checkmesh


def _foam_header(object_name: str, class_name: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version 2.0;\n"
        "    format ascii;\n"
        f"    class {class_name};\n"
        '    location "constant/polyMesh";\n'
        f"    object {object_name};\n"
        "}\n"
    )


def _write_inverted_hex(mesh_dir: Path) -> None:
    mesh_dir.mkdir(parents=True)
    points = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    # Every face is reversed from its owner-outward orientation.
    faces = [
        (1, 2, 3, 0),
        (7, 6, 5, 4),
        (4, 5, 1, 0),
        (5, 6, 2, 1),
        (6, 7, 3, 2),
        (7, 4, 0, 3),
    ]

    (mesh_dir / "points").write_text(
        _foam_header("points", "vectorField")
        + f"{len(points)}\n(\n"
        + "".join(f"({x} {y} {z})\n" for x, y, z in points)
        + ")\n",
        encoding="ascii",
    )
    (mesh_dir / "faces").write_text(
        _foam_header("faces", "faceList")
        + f"{len(faces)}\n(\n"
        + "".join(f"{len(face)}({' '.join(map(str, face))})\n" for face in faces)
        + ")\n",
        encoding="ascii",
    )
    (mesh_dir / "owner").write_text(
        _foam_header("owner", "labelList") + "6\n(\n0\n0\n0\n0\n0\n0\n)\n",
        encoding="ascii",
    )
    (mesh_dir / "neighbour").write_text(
        _foam_header("neighbour", "labelList") + "0\n(\n)\n",
        encoding="ascii",
    )
    (mesh_dir / "boundary").write_text(
        _foam_header("boundary", "polyBoundaryMesh")
        + "1\n(\n"
        + "walls\n{\n    type wall;\n    nFaces 6;\n    startFace 0;\n}\n"
        + ")\n",
        encoding="ascii",
    )


def test_checkmesh_detects_an_inverted_cell(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "polyMesh"
    _write_inverted_hex(mesh_dir)

    result = run_checkmesh(mesh_dir)

    assert result.n_errors > 0, result.raw_output
    assert "Failed" in result.raw_output
    assert "negative cell volume" in result.raw_output
