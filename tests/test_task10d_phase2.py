"""Focused family-input checks; accepted Phase 1 mechanics are not requalified."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from human_punching.task10d.g1 import PROJECT
from human_punching.task10d.phase2 import load_library, load_primitive, landmark_frames


PATHS = {
    "jab": "outputs/task10c/jab/nominal_reference_adaptation_20261022_01/final/candidate/jab.npz",
    "cross": "outputs/task10c/references/cross.npz",
    "left_hook": "outputs/task10c/reference_correction/hooks_final/initial_guard_return/selected/left_hook.npz",
    "right_hook": "outputs/task10c/reference_correction/hooks_final/right_compact_return/selected/right_hook.npz",
}


@pytest.fixture(scope="module")
def records():
    result = []
    for family, path in PATHS.items():
        metadata = json.loads((PROJECT / path).with_suffix(".json").read_text())
        result.append(dict(primitive_id=family + "_input_check", family=family,
            active_hand=metadata["active_hand"], role="nominal", reference_npz=path,
            reference_sha256=hashlib.sha256((PROJECT / path).read_bytes()).hexdigest(),
            placement=dict(xy=[0., 0.], yaw=0.),
            target=dict(snapshot_id=53874, head_world=[1.061174750, .260344207, 1.228431463]),
            annotations=dict(extension_center_s=metadata.get("strike_center_time_s", metadata["strike_time_s"]))))
    return result


def test_all_family_references_share_native_binding_without_jab_landmarks(records):
    primitives = [load_primitive(record) for record in records]
    model = primitives[0].reference.spec().compile()
    binding = primitives[0].reference.bind(model, prefix="")
    data = mujoco.MjData(model)
    for primitive in primitives:
        for source_s in (0., primitive.landmarks["extension_center_s"], primitive.reference.end_time_s):
            sample = primitive.reference.sample(source_s, primitive.placement)
            binding.write(sample, data)
            mujoco.mj_kinematics(model, data)
            np.testing.assert_allclose(data.geom_xpos[binding.glove_geom_ids], sample.glove_pos, atol=1e-14)
        expected = int(np.rint(primitive.landmarks["extension_center_s"]/.8/.02))
        assert expected in landmark_frames(primitive, SimpleNamespace(rate=.8))
    assert len({primitive.landmarks["extension_center_s"] for primitive in primitives}) > 1


def test_manifest_must_match_anatomical_hand_family_and_identity(records):
    for key, wrong_value, message in (("family", "cross", "family mismatch"),
            ("active_hand", "right", "active_hand mismatch"),
            ("reference_sha256", "0"*64, "hash mismatch"),
            ("reference_json_sha256", "0"*64, "JSON hash mismatch")):
        record = deepcopy(records[0])
        record[key] = wrong_value
        with pytest.raises(ValueError, match=message):
            load_primitive(record)
    record = deepcopy(records[1])
    record["placement"]["z"] = -.2
    with pytest.raises(ValueError, match="no vertical shift"):
        load_primitive(record)


def test_family_landmark_required_and_nominal_target_cannot_change(records):
    record = deepcopy(records[1])
    record["annotations"] = {}
    with pytest.raises(ValueError, match="own explicit extension_center_s"):
        load_primitive(record)
    record = deepcopy(records[1])
    record["annotations"]["extension_center_s"] = 900.
    with pytest.raises(ValueError, match="Invalid source landmark"):
        load_primitive(record)
    record = deepcopy(records[1])
    record["target"]["snapshot_id"] = 62759
    with pytest.raises(ValueError, match="only TRAIN snapshot 53874"):
        load_primitive(record)


def test_default_load_requires_four_nominals_and_variants_are_explicit(records, tmp_path):
    path = tmp_path / "library.json"
    variant = deepcopy(records[1])
    variant.update(primitive_id="cross_style_variant", role="variant")
    path.write_text(json.dumps(dict(primitives=[*records, variant])))
    assert len(load_library(path)) == 4
    selected = load_library(path, ["cross_style_variant"])
    assert [primitive.primitive_id for primitive in selected] == ["cross_style_variant"]
    path.write_text(json.dumps(dict(primitives=records[:-1])))
    with pytest.raises(ValueError, match="exactly one nominal per family"):
        load_library(path)

