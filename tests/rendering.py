"""Deterministic off-screen fixture rendering for human review."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nacre-mpl"))

import pyvista as pv

from nacre.meshir import PolyMeshIR


WINDOW_SIZE = (640, 480)
BACKGROUND = "#f4f6f8"
PATCH_COLORS = np.asarray(
    [
        (213, 62, 79),
        (50, 136, 189),
        (145, 145, 145),
        (102, 194, 165),
        (252, 141, 98),
        (141, 160, 203),
    ],
    dtype=np.uint8,
)


def _face(mesh: PolyMeshIR, face_i: int) -> np.ndarray:
    return mesh.face_verts[mesh.face_offset[face_i] : mesh.face_offset[face_i + 1]]


def _boundary_surface(mesh: PolyMeshIR) -> pv.PolyData:
    cells: list[int] = []
    colors: list[np.ndarray] = []
    first_boundary = mesh.n_internal_faces
    for patch_i in range(len(mesh.patch_names)):
        start = first_boundary + int(mesh.patch_offset[patch_i])
        stop = first_boundary + int(mesh.patch_offset[patch_i + 1])
        for face_i in range(start, stop):
            vertices = _face(mesh, face_i)
            cells.extend([len(vertices), *(int(vertex) for vertex in vertices)])
            colors.append(PATCH_COLORS[patch_i % len(PATCH_COLORS)])
    surface = pv.PolyData(mesh.points, np.asarray(cells, dtype=np.int64))
    surface.cell_data["patch_rgb"] = np.asarray(colors, dtype=np.uint8)
    return surface


def _volume_grid(mesh: PolyMeshIR) -> pv.UnstructuredGrid:
    cell_faces: list[list[list[int]]] = [[] for _ in range(mesh.n_cells)]
    for face_i in range(mesh.n_faces):
        vertices = [int(vertex) for vertex in _face(mesh, face_i)]
        cell_faces[int(mesh.owner[face_i])].append(vertices)
        if face_i < mesh.n_internal_faces:
            cell_faces[int(mesh.neighbour[face_i])].append(vertices[::-1])

    cells: list[int] = []
    for faces in cell_faces:
        connectivity = [len(faces)]
        for vertices in faces:
            connectivity.extend([len(vertices), *vertices])
        cells.extend([len(connectivity), *connectivity])
    cell_types = np.full(mesh.n_cells, pv.CellType.POLYHEDRON, dtype=np.uint8)
    return pv.UnstructuredGrid(np.asarray(cells), cell_types, mesh.points)


def _plotter(mesh: PolyMeshIR) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.ren_win.SetMultiSamples(0)
    plotter.set_background(BACKGROUND)
    lower = mesh.points.min(axis=0)
    upper = mesh.points.max(axis=0)
    centre = 0.5 * (lower + upper)
    diagonal = float(np.linalg.norm(upper - lower))
    plotter.camera_position = [
        centre + diagonal * np.asarray((1.7, 1.35, 1.15)),
        centre,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.68 * diagonal
    return plotter


def _save(plotter: pv.Plotter, path: Path) -> None:
    plotter.show(screenshot=path, auto_close=True, interactive=False)


def render_fixture(mesh: PolyMeshIR, fixture: str, artifact_dir: Path) -> list[Path]:
    """Render shaded-patch, wireframe, and interior-cut PNGs."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    surface = _boundary_surface(mesh)
    grid = _volume_grid(mesh)
    outputs = [
        artifact_dir / f"{fixture}__shaded.png",
        artifact_dir / f"{fixture}__wireframe.png",
        artifact_dir / f"{fixture}__cut-plane.png",
    ]

    shaded = _plotter(mesh)
    shaded.add_mesh(
        surface,
        scalars="patch_rgb",
        rgb=True,
        lighting=True,
        ambient=0.25,
        diffuse=0.75,
        specular=0.0,
        show_edges=False,
    )
    _save(shaded, outputs[0])

    wireframe = _plotter(mesh)
    wireframe.add_mesh(
        surface,
        style="wireframe",
        color="#25364a",
        line_width=2.0,
        lighting=False,
    )
    _save(wireframe, outputs[1])

    cut = _plotter(mesh)
    cut.add_mesh(surface, color="#9aa6b2", opacity=0.12, lighting=False)
    section = grid.slice(normal=(1.0, 0.37, 0.19), origin=grid.center)
    cut.add_mesh(
        section,
        color="#e76f51",
        show_edges=True,
        edge_color="#25364a",
        line_width=2.0,
        lighting=False,
    )
    _save(cut, outputs[2])

    return outputs
