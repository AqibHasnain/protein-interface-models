import json

import numpy as np

from cla.utils.logging import JsonlLogger
from cla.utils.manifest import build_manifest, config_hash


def test_config_hash_ignores_key_order_and_tracks_values():
    config = {"lr": 1e-4, "model": {"name": "esm", "layers": 6}, "seed": 0}
    reordered = {"seed": 0, "model": {"layers": 6, "name": "esm"}, "lr": 1e-4}

    assert config_hash(config) == config_hash(reordered)
    assert config_hash(config) != config_hash({**config, "seed": 1})
    assert len(config_hash(config)) == 12


def test_build_manifest_outside_git_repo_records_unknown_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manifest = build_manifest({"seed": 0})

    assert manifest.git_sha == "unknown"
    assert manifest.git_dirty is False


def test_jsonl_logger_writes_one_flushed_line_per_call_with_numpy_and_path(tmp_path):
    logger = JsonlLogger(tmp_path)
    events = tmp_path / "events.jsonl"

    logger.log("start", loss=np.float32(0.5), run_dir=tmp_path)
    # Read before close: a killed run must already have this line on disk.
    assert len(events.read_text().splitlines()) == 1
    logger.log("step", loss=np.float64(0.25), grads=np.array([1.0, 2.0]))
    logger.close()

    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert [r["event"] for r in records] == ["start", "step"]
    assert records[0]["loss"] == 0.5
    assert records[0]["run_dir"] == str(tmp_path)
    assert records[1]["grads"] == [1.0, 2.0]
    assert all("ts" in r for r in records)
