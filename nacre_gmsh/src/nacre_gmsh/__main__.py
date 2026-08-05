"""Command line for the bake pipeline: ``python -m nacre_gmsh --help``."""

from __future__ import annotations

import argparse
from pathlib import Path

from nacre.check import check_surface_input
from nacre.contract import write_surface_input

from nacre_gmsh.bake import BakeSettings, bake_step
from nacre_gmsh.corpus import write_goldens
from nacre_gmsh.session import gmsh_session


def _bake(arguments: argparse.Namespace) -> int:
    settings = BakeSettings(
        size_max=arguments.size_max,
        size_min=arguments.size_min,
        size_from_curvature=arguments.size_from_curvature,
        feature_angle_deg=arguments.feature_angle,
    )
    with gmsh_session(verbose=arguments.verbose):
        surface = bake_step(arguments.source, settings)

    written = write_surface_input(surface, arguments.destination)
    report = check_surface_input(surface)
    print(
        f"{written}: {report.n_verts} vertices, {report.n_tris} triangles, "
        f"{report.n_patches} patches {surface.patch_names}, "
        f"{report.n_feat_edges} feature edges, "
        f"{report.n_feat_corners} corners, "
        f"curvature in [{report.min_kappa:.6g}, {report.max_kappa:.6g}]"
    )
    return 0


def _goldens(arguments: argparse.Namespace) -> int:
    with gmsh_session(verbose=arguments.verbose):
        written = write_goldens(arguments.destination)
    for path in written:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nacre_gmsh", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="show Gmsh output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bake = subparsers.add_parser(
        "bake", help="bake a STEP/IGES/BREP file into a SurfaceInput .npz"
    )
    bake.add_argument("source", type=Path, help="CAD file; its solids are the fluid")
    bake.add_argument("destination", type=Path, help="output .npz path")
    bake.add_argument("--size-max", type=float, default=None)
    bake.add_argument("--size-min", type=float, default=None)
    bake.add_argument(
        "--size-from-curvature",
        type=float,
        default=0.0,
        help="elements per 2*pi of curvature radius; 0 disables",
    )
    bake.add_argument("--feature-angle", type=float, default=30.0)
    bake.set_defaults(handler=_bake)

    goldens = subparsers.add_parser(
        "goldens", help="rebake the committed analytic golden .npz corpus"
    )
    goldens.add_argument("destination", type=Path, help="output directory")
    goldens.set_defaults(handler=_goldens)

    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
