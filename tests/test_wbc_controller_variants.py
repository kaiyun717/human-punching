"""Named controller variants preserve force caps/assets and reject stale hashes."""
import json
from pathlib import Path

import pytest

from human_punching.wbc.controller_variants import gain_variant
from human_punching.wbc.provenance import PROJECT_ROOT, sha256_file
from human_punching.wbc.runtime.bundle import load_runtime_bundle


def test_gain_variants_preserve_source_and_reopen_qualification(tmp_path, task10_common_bundle):
    source = task10_common_bundle.root
    before = sha256_file(source/"manifest.json")
    original = load_runtime_bundle(source)
    result = gain_variant(source, tmp_path/"variant", {"LeftFoot_x": {"kp": 2., "kd": 2**.5}},
                          reason="CPU-only engineering fixture")
    changed = load_runtime_bundle(result["path"])
    index = original.joint_names.index("LeftFoot_x")
    assert changed.body["kp"][index] == original.body["kp"][index]*2
    assert changed.body["effort"] == original.body["effort"]
    assert changed.controller_sha256() != original.controller_sha256()
    assert sha256_file(changed.model_path) == sha256_file(original.model_path)
    assert sha256_file(source/"manifest.json") == before
    with pytest.raises(ValueError, match="unqualified"):
        changed.require_training_qualification()
    metadata = changed.path(changed.manifest["body_metadata_path"])
    body = json.loads(metadata.read_text()); body["kp"][index] += 1
    metadata.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="asset hash mismatch"):
        load_runtime_bundle(changed.root)


def test_cannot_change_effort_limits_disguised_as_gains(tmp_path, task10_common_bundle):
    with pytest.raises(ValueError, match="kp and kd"):
        gain_variant(task10_common_bundle.root, tmp_path/"invalid",
                     {"LeftFoot_x": {"kp": 2., "effort": 2.}}, reason="invalid fixture")
    assert not (tmp_path/"invalid").exists()
