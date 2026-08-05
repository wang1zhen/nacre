# nacre

<!-- TODO: status badge -->

**nacre** is a modular, embeddable Python library for generating hybrid CFD meshes from CAD geometry: an octree hexahedral core, anisotropic boundary layers, and a conformal polyhedral transition.

The core is licensed under [MPL-2.0](LICENSE). It targets Python 3.11 and later.

## Status

This repository is at the foundation stage. No meshing algorithm, command-line interface, file writer, or validated workflow works yet. The quick start and architecture below describe the intended interface, not released functionality.

M0--M2 are primarily engineering work based on documented algorithms and existing references. Boundary-layer extrusion and conformal transition in M3--M4 are research-grade work. In particular, there is no clean open-source reference implementation for the complete transition planned here. See the [roadmap](ROADMAP.md) for exit criteria and risks.

## Why this exists

The most valuable thing nacre can ship is a standalone, embeddable, MPL-2.0-licensed 3D anisotropic boundary-layer extruder. The rest of the system makes that extruder usable in a complete CFD meshing workflow or useful independently.

The current open-source tools leave a gap:

- **cfMesh** has the only broadly usable open-source 3D boundary-layer extruder, but its public development has been stagnant around the v1.1.2 era. It is GPL-licensed, coupled to OpenFOAM workflows, and accepts triangulated surfaces rather than CAD curvature.
- **snappyHexMesh** is an application rather than an embeddable library. Its layer addition can be fragile, and its triangulated-surface input does not preserve true BREP curvature.
- **Gmsh** provides an excellent CAD kernel interface, surface meshing, and size fields. It does not provide the intended octree core or a production 3D boundary-layer extrusion workflow, and its element-centric representation is not a native representation for arbitrary polyhedral finite-volume cells.

nacre fills the middle: a face-based polyhedral core and embeddable boundary-layer extruder, while delegating CAD import, surface meshing, and BREP queries to Gmsh. The goal is not to replace the parts that Gmsh already does well.

## What it produces

The primary output is an OpenFOAM `polyMesh`. CGNS `NGON_n`/`NFACE_n` is the secondary format, and VTU is the debugging format.

<!-- TODO: gallery -->

No output-quality or performance claims are made yet. Those claims will require recorded corpus, `checkMesh`, solver-validation, and timing results.

## Architecture

Modules are deliberately decoupled. Each should be independently testable and independently useful, and modules communicate only through explicitly frozen interfaces.

```text
nacre/
  contract.py     # planned for M1, when the bake pipeline produces SurfaceInput
  meshir.py       # face-based PolyMeshIR: points/faces/owner/neighbour, SoA arrays
  sizefield.py    # curvature + proximity + BOI -> unified size field
  octree.py       # octree construction, size refinement, and 2:1 balance
  trim.py         # planned split: classification and boundary trimming
  blex.py         # boundary-layer extrusion; the flagship module
  stitch.py       # conformal polyhedral transition from BL cap to octree core
  optimize.py     # polyhedral quality optimization
  io/             # polyMesh, CGNS, and VTU writers
  check/          # one invariant checker per module

nacre_gmsh/       # separate GPL package: STEP -> SurfaceInput
```

`trim.py` is an anticipated split, not a present implementation commitment: classification and trimming may begin beside octree construction, but move into `trim.py` before `octree.py` approaches the roughly 800-line module budget.

### Interface freeze points

The implemented `PolyMeshIR` CSR layout, field names, dtypes, and OpenFOAM ordering conventions are frozen. Changing them requires an explicit project-owner decision.

`SurfaceInput` is deliberately deferred until the start of M1, when the Gmsh bake pipeline supplies a real producer. Its planned conceptual API is:

```python
@dataclass(frozen=True)
class SurfaceInput:
    points:       np.ndarray   # (Np, 3) float64
    tris:         np.ndarray   # (Nt, 3) int32
    tri_patch:    np.ndarray   # (Nt,)   int32
    patch_names:  tuple[str, ...]
    vert_normal:  np.ndarray   # (Np, 3) float64
    vert_kappa:   np.ndarray   # (Np, 2) float64, principal BREP curvatures
    feat_edges:   np.ndarray   # (Ne, 2) int32
    feat_corners: np.ndarray   # (Nc,)   int32

class SizeField(Protocol):
    def __call__(self, xyz: np.ndarray) -> np.ndarray: ...  # (N, 3) -> (N,)
```

`vert_kappa` comes from Gmsh's BREP curvature query, not from a discrete estimate over surface triangles. Retaining CAD-derived curvature is the central reason for using Gmsh as the frontend rather than an STL-only workflow.

Once implemented, changing this contract will require an explicit project-owner decision. It must never happen as a side effect of implementing or refactoring another module.

### Bake once, mesh without Gmsh

Gmsh runs once at the front of the workflow. Its result is baked into a `.npz` file; the core does not import or call Gmsh while meshing.

```text
STEP/BREP -- nacre-gmsh --> SurfaceInput .npz -- nacre core --> volume mesh
```

This boundary exists for three reasons:

1. **License isolation.** The MPL core has no GPL runtime dependency. Only the separate adapter process imports Gmsh.
2. **Performance.** Meshing hot loops do not make Python-to-Gmsh round trips.
3. **Testing.** CI uses committed `.npz` golden files and does not need Gmsh installed.

## Quick start (aspirational)

The commands below document the intended interface. They do not work yet.

Create an environment and install both distributions when CAD baking is required:

```console
uv init nacre-case --python 3.11
uv add --project nacre-case nacre nacre-gmsh
```

Bake once, then mesh and check without a Gmsh call in the meshing stage:

```console
uv run --project nacre-case nacre bake model.step -o model.surf.npz
uv run --project nacre-case nacre mesh model.surf.npz config.yaml -o case/constant/polyMesh/
uv run --project nacre-case nacre check case/constant/polyMesh/
```

For development from this workspace:

```console
uv sync --all-packages
uv run --package nacre pytest
```

Environment and dependencies are managed only with `uv`; project executables and Python commands are invoked with `uv run`.

## Design principles

### JAX for numbers, NumPy for topology

JAX is used where dense numerical work has fixed-shape arrays: size-field and SDF evaluation, batched normals and curvature transforms, size blending, cell-quality metrics, smoothing and optimization inner loops, and batched geometric predicates. `jit` and `grad` are expected to pay off most clearly in `optimize.py`.

Topology remains in NumPy and Python: octree refinement and balancing, face merging, cell construction, stitching, dynamic shapes, data-dependent control flow, and growing arrays. Forcing those operations through `jit` would create recompilation pressure and make topology harder to reason about.

The rule of thumb is: **JAX for numbers over fixed-shape arrays; NumPy for topology.**

JAX x64 is mandatory and must be enabled before other JAX work:

```python
import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.x64_enabled
```

JAX defaults can otherwise permit float32 geometry. Near-degenerate and near-coplanar predicates can then fail silently and lead to inverted cells. Package import will assert x64 mode, and tests will reject float32 core arrays.

### Structure of arrays, never arrays of objects

Points, faces, cells, connectivity, and attributes are NumPy or JAX arrays. They are never represented as per-face or per-cell Python objects. `PolyMeshIR` is face-based, with `points`, `faces`, `owner`, and `neighbour` data in structure-of-arrays form.

### Invariants before implementation

Every module receives a `check_*()` function in `nacre/check/` before its implementation. Functions are pure where practical: arrays in, arrays out, no shared mutable state, inheritance, or class hierarchies. Each module stays near or below 800 lines and is split by responsibility before exceeding that budget.

Every test geometry renders a PNG into `tests/artifacts/`, archived per commit. `meshir.py` and invariant checkers receive property-based tests with Hypothesis. Baked `SurfaceInput` golden files let CI run independently of Gmsh. A 20--30-case corpus, ranging from analytic shapes to hostile CAD, is created before algorithm work; corpus failure rate is the headline robustness metric.

### Three required testing tiers

1. **Structural invariants:** fast checks on every commit. These include unique and correctly shared faces, cell closure, `owner < neighbour`, 2:1 octree balance, positive non-self-intersecting prisms, monotonic layer thickness, and conformal transition faces.
2. **OpenFOAM `checkMesh`:** every mesh-producing test exports a `polyMesh` and must pass with zero errors. Maximum non-orthogonality and skewness are tracked even when the mesh passes. This independent tool is the project's most valuable legal-mesh judge.
3. **Solver validation:** nightly and release-gated OpenFOAM cases compare against numeric references: a laminar flat plate against Blasius profile and skin friction, a cylinder at Re=40 for recirculation length and drag, a backward-facing step at Re=800 for reattachment length, and a curved duct for curvature-driven resolution.

Passing `checkMesh` proves that a mesh is legal, not that it is suitable or correct for CFD. A module is not done until its checker passes on the full applicable corpus.

## Relationship to Gmsh

Gmsh is an optional frontend dependency, isolated in the `nacre-gmsh` distribution. nacre delegates CAD import, surface meshing, and BREP curvature to it permanently. The project will not maintain a Gmsh fork or compete with Gmsh's surface mesher.

Where nacre needs missing frontend capabilities, the preferred route is to contribute enabling patches upstream: batched or vectorized API queries, proximity/local-feature-size fields, and exposed feature-edge classification.

## Non-goals

- nacre is not a flow solver.
- It is not a CAD kernel. It will not implement geometry healing or its own Boolean engine.
- It is not a GUI.
- It does not compete with Gmsh on surface meshing; that work is delegated entirely and permanently.
- v1 will not provide MPI or distributed meshing. The target is shared-memory execution.
- It is not a general-purpose FEM mesher. CFD finite-volume meshes are the target.

## Licensing

The workspace, documentation, and core `nacre` distribution are MPL-2.0. The optional `nacre-gmsh` distribution is GPL-3.0-or-later because it depends on Gmsh. Each distribution has its own `pyproject.toml` and `LICENSE`; the core has no dependency on the adapter or on GPL code.

Distribution names are `nacre` and `nacre-gmsh`. Their Python import names are `nacre` and `nacre_gmsh`.

## Contributing

The project is designed for machine-verifiable development, including AI-assisted contributions. Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change; checker-first development, frozen interfaces, and all three test tiers are project rules rather than suggestions.
