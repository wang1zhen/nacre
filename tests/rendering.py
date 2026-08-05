"""Deterministic off-screen fixture rendering for human review."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nacre-mpl"))


MIN_CHANNEL_STD = 1.0
MIN_DISTINCT_COLORS = 1
_BACKENDS = (
    ("EGL", "vtkEGLRenderWindow"),
    ("OSMesa", "vtkOSOpenGLRenderWindow"),
    ("Xvfb", "vtkXOpenGLRenderWindow"),
)
_PROBE = f"""
import os
import numpy as np
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import pyvista as pv
p = pv.Plotter(off_screen=True, window_size=(96, 72))
p.ren_win.SetMultiSamples(0)
p.set_background('#f4f6f8')
p.add_mesh(pv.Cube(), color='#e76f51', lighting=True)
p.camera_position = 'iso'
image = np.asarray(p.screenshot(return_img=True))[..., :3]
actual = type(p.ren_win).__name__
p.close()
std = image.reshape(-1, 3).std(axis=0)
colors = np.unique(image.reshape(-1, 3), axis=0).shape[0]
assert np.all(std > {MIN_CHANNEL_STD!r}), (std, colors)
assert colors > {MIN_DISTINCT_COLORS!r}, (std, colors)
print('NACRE_RENDER_PROBE:' + actual)
"""


def _select_backend() -> tuple[str, str]:
    failures: list[str] = []
    for name, vtk_class in _BACKENDS:
        environment = os.environ.copy()
        environment["VTK_DEFAULT_OPENGL_WINDOW"] = vtk_class
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
            check=False,
        )
        marker = f"NACRE_RENDER_PROBE:{vtk_class}"
        if completed.returncode == 0 and marker in completed.stdout:
            os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = vtk_class
            print(f"nacre headless renderer: {name} ({vtk_class})", flush=True)
            return name, vtk_class
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        failures.append(f"{name}: {detail[-1] if detail else 'probe failed'}")
        print(f"nacre headless renderer rejected {name}", flush=True)
    raise RuntimeError("no usable headless renderer; " + "; ".join(failures))


RENDER_BACKEND, RENDER_WINDOW_CLASS = _select_backend()

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


def assert_nonuniform_render(image: np.ndarray, path: Path | str) -> None:
    """Reject blank or near-uniform output from a nominally successful render."""

    pixels = np.asarray(image)[..., :3]
    channel_std = pixels.reshape(-1, 3).std(axis=0)
    distinct_colors = np.unique(pixels.reshape(-1, 3), axis=0).shape[0]
    assert np.all(channel_std > MIN_CHANNEL_STD), (
        f"blank or near-uniform render {path}: channel std={channel_std}"
    )
    assert distinct_colors > MIN_DISTINCT_COLORS, (
        f"blank or near-uniform render {path}: {distinct_colors} distinct colors"
    )


def _save(plotter: pv.Plotter, path: Path) -> None:
    image = plotter.show(
        screenshot=path,
        auto_close=False,
        interactive=False,
        return_img=True,
    )
    plotter.close()
    assert_nonuniform_render(image, path)


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


if __name__ == "__main__":
    print(f"selected renderer backend: {RENDER_BACKEND}")
