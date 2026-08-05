import json
from pathlib import Path

import numpy as np
import pytest

from conftest import EvaluatedFixture
from fixtures import FIXTURE_SPECS
from rendering import assert_nonuniform_render, render_fixture


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
METRIC_COLUMNS = (
    "fixture",
    "n_cells",
    "n_faces",
    "n_points",
    "max_non_ortho",
    "max_skewness",
    "max_aspect_ratio",
    "volume_rel_error",
    "checkmesh_errors",
    "layer_coverage",
    "wall_time_s",
)


def test_blank_render_is_rejected() -> None:
    with pytest.raises(AssertionError, match="blank or near-uniform"):
        assert_nonuniform_render(np.zeros((32, 32, 3), dtype=np.uint8), "blank.png")


def test_emit_visual_and_metrics_artifacts(
    evaluated_fixtures: dict[str, EvaluatedFixture],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for spec in FIXTURE_SPECS:
        evaluated = evaluated_fixtures[spec.name]
        structural = evaluated.structural
        external = evaluated.external
        volume_rel_error = (
            abs(structural.total_volume - external.total_volume)
            / abs(external.total_volume)
        )
        record = {
            "fixture": spec.name,
            "n_cells": structural.n_cells,
            "n_faces": structural.n_faces,
            "n_points": structural.n_points,
            "max_non_ortho": external.max_non_ortho,
            "max_skewness": external.max_skewness,
            "max_aspect_ratio": external.max_aspect_ratio,
            "volume_rel_error": volume_rel_error,
            "checkmesh_errors": external.n_errors,
            "layer_coverage": None,
            "wall_time_s": evaluated.wall_time_s,
        }
        assert tuple(record) == METRIC_COLUMNS
        assert all(
            np.isfinite(record[column])
            for column in (
                "max_non_ortho",
                "max_skewness",
                "max_aspect_ratio",
                "volume_rel_error",
                "wall_time_s",
            )
        )
        records.append(record)

    metrics_path = ARTIFACT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    rendered: list[Path] = []
    for spec in FIXTURE_SPECS:
        rendered.extend(
            render_fixture(
                evaluated_fixtures[spec.name].mesh,
                spec.name,
                ARTIFACT_DIR,
            )
        )

    assert len(records) == len(FIXTURE_SPECS)
    assert len(rendered) == 3 * len(FIXTURE_SPECS)
    assert all(path.is_file() and path.stat().st_size > 0 for path in rendered)
