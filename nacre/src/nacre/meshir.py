"""Face-based, structure-of-arrays intermediate mesh representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PolyMeshIR:
    """Topology and geometry required to write an OpenFOAM ``polyMesh``.

    Faces use CSR storage: the vertices of face ``i`` are
    ``face_verts[face_offset[i]:face_offset[i + 1]]``. ``neighbour`` contains
    one entry per internal face, so its length is also the index of the first
    boundary face. Patch offsets are relative to that boundary-face range.

    This class is only a data container. Use ``nacre.check.check_meshir`` for
    validation.
    """

    points: NDArray[np.float64]
    face_verts: NDArray[np.int32]
    face_offset: NDArray[np.int32]
    owner: NDArray[np.int32]
    neighbour: NDArray[np.int32]
    patch_names: tuple[str, ...]
    patch_types: tuple[str, ...]
    patch_offset: NDArray[np.int32]

    @property
    def n_faces(self) -> int:
        return len(self.owner)

    @property
    def n_internal_faces(self) -> int:
        return len(self.neighbour)

    @property
    def n_boundary_faces(self) -> int:
        return self.n_faces - self.n_internal_faces

    @property
    def n_cells(self) -> int:
        owner_max = int(np.max(self.owner, initial=-1))
        neighbour_max = int(np.max(self.neighbour, initial=-1))
        return max(owner_max, neighbour_max) + 1


__all__ = ["PolyMeshIR"]
