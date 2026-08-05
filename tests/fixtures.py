"""Explicit, hand-built meshes for the M0 end-to-end loop."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from nacre.meshir import PolyMeshIR


def _explicit_mesh(
    *,
    points: Sequence[tuple[float, float, float]],
    faces: Sequence[tuple[int, ...]],
    owner: Sequence[int],
    neighbour: Sequence[int],
) -> PolyMeshIR:
    face_sizes = np.asarray([len(face) for face in faces], dtype=np.int32)
    face_offset = np.empty(len(faces) + 1, dtype=np.int32)
    face_offset[0] = 0
    np.cumsum(face_sizes, out=face_offset[1:])
    return PolyMeshIR(
        points=np.asarray(points, dtype=np.float64),
        face_verts=np.asarray(
            [vertex for face in faces for vertex in face], dtype=np.int32
        ),
        face_offset=face_offset,
        owner=np.asarray(owner, dtype=np.int32),
        neighbour=np.asarray(neighbour, dtype=np.int32),
        patch_names=("walls",),
        patch_types=("wall",),
        patch_offset=np.asarray([0, len(faces) - len(neighbour)], dtype=np.int32),
    )


def single_hexahedron() -> PolyMeshIR:
    return _explicit_mesh(
        points=[
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        faces=[
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ],
        owner=[0, 0, 0, 0, 0, 0],
        neighbour=[],
    )


def block_2x2x2() -> PolyMeshIR:
    return _explicit_mesh(
        points=[
            (0, 0, 0), (1, 0, 0), (2, 0, 0),
            (0, 1, 0), (1, 1, 0), (2, 1, 0),
            (0, 2, 0), (1, 2, 0), (2, 2, 0),
            (0, 0, 1), (1, 0, 1), (2, 0, 1),
            (0, 1, 1), (1, 1, 1), (2, 1, 1),
            (0, 2, 1), (1, 2, 1), (2, 2, 1),
            (0, 0, 2), (1, 0, 2), (2, 0, 2),
            (0, 1, 2), (1, 1, 2), (2, 1, 2),
            (0, 2, 2), (1, 2, 2), (2, 2, 2),
        ],
        faces=[
            (1, 4, 13, 10),
            (3, 12, 13, 4),
            (9, 10, 13, 12),
            (4, 13, 14, 5),
            (10, 11, 14, 13),
            (4, 7, 16, 13),
            (12, 13, 16, 15),
            (13, 14, 17, 16),
            (10, 13, 22, 19),
            (12, 21, 22, 13),
            (13, 22, 23, 14),
            (13, 16, 25, 22),
            (9, 12, 3, 0),
            (12, 15, 6, 3),
            (18, 21, 12, 9),
            (21, 24, 15, 12),
            (2, 5, 14, 11),
            (5, 8, 17, 14),
            (11, 14, 23, 20),
            (14, 17, 26, 23),
            (1, 10, 9, 0),
            (2, 11, 10, 1),
            (10, 19, 18, 9),
            (11, 20, 19, 10),
            (6, 15, 16, 7),
            (7, 16, 17, 8),
            (15, 24, 25, 16),
            (16, 25, 26, 17),
            (3, 4, 1, 0),
            (4, 5, 2, 1),
            (6, 7, 4, 3),
            (7, 8, 5, 4),
            (18, 19, 22, 21),
            (19, 20, 23, 22),
            (21, 22, 25, 24),
            (22, 23, 26, 25),
        ],
        owner=[
            0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6,
            0, 2, 4, 6, 1, 3, 5, 7, 0, 1, 4, 5,
            2, 3, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
        ],
        neighbour=[1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7],
    )


def two_cells_with_hanging_node() -> PolyMeshIR:
    return _explicit_mesh(
        points=[
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
            (1, 0, 0.5),
        ],
        faces=[
            (1, 2, 6, 5, 12),
            (0, 4, 7, 3),
            (8, 9, 11, 10),
            (0, 1, 12, 5, 4),
            (1, 8, 10, 5, 12),
            (3, 7, 6, 2),
            (2, 6, 11, 9),
            (0, 3, 2, 1),
            (1, 2, 9, 8),
            (4, 5, 6, 7),
            (5, 10, 11, 6),
        ],
        owner=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        neighbour=[1],
    )


def two_cells_with_warped_face() -> PolyMeshIR:
    return _explicit_mesh(
        points=[
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1.1, 1.05, 1.08),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
        ],
        faces=[
            (1, 2, 6, 5),
            (0, 4, 7, 3),
            (8, 9, 11, 10),
            (0, 1, 5, 4),
            (1, 8, 10, 5),
            (3, 7, 6, 2),
            (2, 6, 11, 9),
            (0, 3, 2, 1),
            (1, 2, 9, 8),
            (4, 5, 6, 7),
            (5, 10, 11, 6),
        ],
        owner=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        neighbour=[1],
    )


def block_2x2x2_multi_patch() -> PolyMeshIR:
    return replace(
        block_2x2x2(),
        patch_names=("inlet", "outlet", "walls"),
        patch_types=("patch", "patch", "wall"),
        patch_offset=np.asarray([0, 4, 8, 24], dtype=np.int32),
    )


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    factory: Callable[[], PolyMeshIR]
    all_geometry: bool


FIXTURE_SPECS = (
    FixtureSpec("single-hexahedron", single_hexahedron, False),
    FixtureSpec("2x2x2-block", block_2x2x2, True),
    FixtureSpec("hanging-node-pentagon", two_cells_with_hanging_node, False),
    FixtureSpec("warped-face", two_cells_with_warped_face, False),
    FixtureSpec("2x2x2-multi-patch", block_2x2x2_multi_patch, True),
)
