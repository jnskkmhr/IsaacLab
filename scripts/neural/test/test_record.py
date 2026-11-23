# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of how ``record.py`` splits a terrain into processes, and of what it refuses before it simulates."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import record  # noqa: E402
import train  # noqa: E402


def options(arguments: list[str]) -> dict:
    """Command-line arguments of one process, as ``{option: values}``."""
    parsed, key = {}, None
    for token in arguments:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            parsed[key] = []
        else:
            parsed[key].append(token)
    return parsed


@pytest.mark.parametrize("terrain", dataset.PRODUCTION_TERRAINS)
def test_processes_of_a_terrain_record_every_episode_once(terrain):
    """Each process must hold one scene, and together they must leave no episode out or record one twice."""
    recorded = []
    for unit in record.build_units(terrain, ["all"], dataset.PER_MOTION):
        given = options(unit["args"])
        records = record.select_records(
            given["terrain"][0],
            given["category"][0],
            int(given["per_motion"][0]),
            given.get("shape", [None])[0],
            int(given["size_level"][0]) if "size_level" in given else None,
            given["variant"][0],
            [int(i) for i in given["indices"]] if "indices" in given else None,
        )
        assert records, unit["label"]
        assert len({r["terrain_variant"] for r in records}) == 1, unit["label"]
        if given["category"][0] == "freefall":
            assert len(records) == 1, unit["label"]  # the block fixes the particle count of the scene
        recorded += [r["id"] for r in records]

    expected = [r["id"] for category in dataset.CATEGORY_MOTIONS for r in dataset.episodes(terrain, category)]
    assert sorted(recorded) == sorted(expected)


def test_object_category_without_a_shape_stops_before_the_simulation(tmp_path, capsys):
    with pytest.raises(SystemExit):
        record.main(["--dataset", str(tmp_path), "--terrain", "flat", "--category", "impact"])

    assert "--shape and --size-level are required" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_training_does_not_overwrite_a_finished_run(tmp_path, capsys):
    (tmp_path / "last.pt").touch()

    with pytest.raises(SystemExit):
        train.main(["--dataset", str(tmp_path), "--output", str(tmp_path)])

    assert "already holds a checkpoint" in capsys.readouterr().err
