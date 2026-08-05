# Contributing to nacre

nacre is designed for machine-verifiable development. This is especially important because substantial work may be AI-assisted and may not receive line-by-line human review. Invariants and independent validation are the source of truth.

This document is an initial policy stub. Workflow details will be expanded during M0 without weakening the rules below.

## Development environment

Python 3.11 or later is required. Use `uv` exclusively for environment and dependency management. Do not use `pip`, Conda, Poetry, or direct `venv` commands. Add or remove dependencies with `uv add` and `uv remove`; never hand-edit dependency lists in `pyproject.toml`, so `uv.lock` stays synchronized.

Run project tools through `uv run`:

```console
uv sync --all-packages
uv run --package nacre pytest
uv run --package nacre python -c "import jax; print(jax.config.x64_enabled)"
```

## Required order of work

1. Define the module's input/output invariants and failure diagnostics.
2. Add its `nacre/check/check_*()` checker before implementation.
3. Add focused examples, property-based coverage where applicable, and deterministic PNG rendering.
4. Implement with arrays in and arrays out.
5. Run structural checks on the complete applicable corpus.
6. For every mesh-producing case, export `polyMesh` and run OpenFOAM `checkMesh`. Zero errors is a hard gate.
7. Run applicable solver-validation gates before a release.

A module is not done until its checker passes on the full applicable corpus. Tests must not duplicate an implementation in a way that lets the same mistake validate itself.

## Frozen interfaces

`PolyMeshIR` is frozen. Its CSR layout; field names (`points`, `face_verts`, `face_offset`, `owner`, `neighbour`, `patch_names`, `patch_types`, and `patch_offset`); dtypes; internal-before-boundary face layout; upper-triangular internal-face ordering; and contiguous patch ordering require an explicit decision from the project owner to change. Never alter them as a side effect of implementing another module.

`SurfaceInput` is frozen as of M1, when the Gmsh bake pipeline supplied a real producer to verify the contract against. `nacre/contract.py` carries the same explicit-approval protection as `PolyMeshIR`: its field names, dtypes, and both sign conventions — `vert_normal` points into the fluid, and curvature is positive where the wall is convex toward the fluid — require an explicit decision from the project owner to change. The module docstring records the experiment that established each convention; change the convention and you invalidate every committed golden `.npz` and every boundary-layer extrusion direction.

Modules must not reach into one another's private data or share mutable state. Do not perform cross-module refactoring to make an implementation more convenient.

## Data and module design

- Use structure-of-arrays NumPy/JAX data. Faces and cells are arrays, never Python objects.
- Prefer pure functions, with arrays in and arrays out.
- Do not introduce shared mutable state, class hierarchies, or inheritance.
- Keep each module near or below 800 lines. Split it by responsibility before it grows beyond that budget.
- Use JAX for fixed-shape dense numerical work. Use NumPy and Python for topology, dynamic shapes, and data-dependent construction.
- Enable `jax_enable_x64` at package import before other JAX work and assert that it is active. Core floating-point arrays must never be float32.

## Required validation

### Tier 1: structural invariants

Run fast checks on every commit. Applicable checks include exactly two cells per internal face, cell closure from area-weighted face normals, `owner < neighbour`, no duplicate faces, 2:1 octree balance, positive non-self-intersecting prisms, monotonic layer thickness, and a conformal transition interface.

Use Hypothesis for `meshir.py` and checker behavior. Checker tests must include deliberately corrupted data and verify that the correct invariant fails loudly.

### Tier 2: OpenFOAM `checkMesh`

Every mesh-producing test exports a `polyMesh` and runs `checkMesh`. Zero errors is mandatory. Archive maximum non-orthogonality, maximum skewness, and maximum aspect ratio rather than reducing the result to pass/fail.

Select `-allGeometry` per fixture, never as a suite-global switch. The current small-fixture exclusion boundary is **fewer than 8 cells**: those deliberately tiny meshes may use standard `checkMesh` because their cell-determinant stencil can be underdetermined. Fixtures with 8 or more cells should enable `-allGeometry`. Any mesh with more than 100 cells **must** run `-allGeometry`; opting out above that threshold is not permitted.

### Tier 3: solver validation

Nightly and release-gated OpenFOAM cases compare against committed numeric references and tolerances: Blasius flat plate, cylinder at Re=40, backward-facing step at Re=800, and a curved duct. `checkMesh` legality never substitutes for solver validation.

## Corpus, golden files, and visual evidence

- Commit baked `SurfaceInput` `.npz` golden files so core CI never requires Gmsh. Regenerate them with `uv run python -m nacre_gmsh goldens tests/goldens` and never let a golden become the only definition of correctness: every committed golden must also be checked against closed-form geometry in `tests/surface_anchors.py`.
- Build and retain a 20--30-geometry corpus from analytic cases through hostile CAD.
- Track corpus failure rate as the headline robustness metric.
- Render every test case deterministically to `tests/artifacts/` and archive PNGs per commit.
- Record the fixed metrics defined in [ROADMAP.md](ROADMAP.md) for every mesh-producing milestone.

## Licensing boundary

Core `nacre` code is MPL-2.0 and must never contain, link, import, or acquire a runtime dependency on GPL code. Gmsh integration belongs only in the separate GPL-3.0-or-later `nacre-gmsh` distribution. The core consumes baked contract data and does not call Gmsh while meshing.

This is enforced by machine, not by review. `tests/test_license_boundary.py` parses every core module and rejects any `gmsh` or `nacre_gmsh` import, and checks in a subprocess that importing the core loads neither. The `core-license-boundary` CI job then installs the core distribution alone, asserts `import gmsh` raises `ImportError`, and runs the full suite against the committed golden `.npz` files. Only `tests/test_gmsh_bake.py` may import the Gmsh frontend, and it skips itself when the frontend is absent.

## Names and trademarks

Do not use `Mosaic`, `poly-hexcore`, `hexcore`, or names containing `Foam` as a project name, distribution name, CLI command, or public-facing brand. Descriptive prose may compare capabilities with named products or approaches where accurate.

This restriction does **not** apply to accurate internal implementation identifiers. An internal name such as `hexcore` is not itself treated as a branding violation, although this project uses `octree.py` because it describes the mechanism more precisely. OpenFOAM and `polyMesh` may be named descriptively when discussing compatibility, output, or validation.
