import mujoco
import numpy as np
import pytest

from human_punching.wbc.references import PROJECT_ROOT
from human_punching.wbc.snapshots import SnapshotBank, head_point_state, load_verified_g1_model

BANK = PROJECT_ROOT.parent / "robust_sav/constants/boxing/snapshots/v1_20260921"


def test_actual_snapshot_payloads_keep_original_groups_and_eligibility():
    bank = SnapshotBank(BANK)
    assert len(bank.locations["slot_id"]) == 317520
    assert [len(bank.eligible_ids(split)) for split in ("train", "validation", "test")] == [23135, 23331, 23335]
    for split in ("train", "validation", "test"):
        ids = bank.eligible_ids(split)
        state = bank.state(int(ids[0]))
        assert len(state.qpos) == 36 and len(state.qvel) == 35
        assert state.original_split == split
        assert state.source_group_id == f"command:{state.command_id}"
        assert state.original_source_group_id >= 0 and state.rollout_id >= 0 and state.seed > 0
        assert state.eligible and len(state.payload_sha256) == 64
    with pytest.raises(ValueError):
        bank.state(-1)
    with pytest.raises(ValueError):
        bank.state(317520)


def test_head_point_velocity_includes_rotating_body_offset():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root" pos="0 0 1"><freejoint/>
        <geom name="head_collision" type="sphere" size=".06" pos=".2 0 .4" mass="1"/>
      </body></worldbody></mujoco>''')
    qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
    qvel[0] = .3
    qvel[5] = 2
    head, velocity = head_point_state(model, qpos, qvel)
    np.testing.assert_allclose(head, [.2, 0, 1.4])
    np.testing.assert_allclose(velocity, [.3, .4, 0])
    dt = 1e-6
    moved = qpos.copy()
    mujoco.mj_integratePos(model, moved, qvel, dt)
    next_head, _ = head_point_state(model, moved, qvel)
    np.testing.assert_allclose((next_head-head)/dt, velocity, atol=1e-6)


def test_actual_g1_assets_and_snapshot_head_geometry():
    assets = PROJECT_ROOT / ".cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1"
    if not assets.exists():
        pytest.skip("Run project-local setup to export pinned mjlab assets")
    bank = SnapshotBank(BANK)
    plant = PROJECT_ROOT.parent / "robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json"
    model, provenance = load_verified_g1_model(assets, bank, plant)
    state = bank.state(int(bank.eligible_ids("train")[0]))
    qpos, qvel = np.array(state.qpos), np.array(state.qvel)
    head, velocity = head_point_state(model, qpos, qvel)
    assert np.isfinite(head).all() and np.isfinite(velocity).all()
    assert provenance["asset_count"] == 36
    moved = qpos.copy()
    dt = 1e-6
    mujoco.mj_integratePos(model, moved, qvel, dt)
    next_head, _ = head_point_state(model, moved, qvel)
    np.testing.assert_allclose((next_head-head)/dt, velocity, atol=1e-5)


def test_missing_payload_fails_instead_of_schema_only_success(tmp_path):
    # A genuine manifest/index without its location NPZ cannot masquerade as a bank.
    import shutil
    shutil.copyfile(BANK / "manifest.json", tmp_path / "manifest.json")
    shutil.copyfile(BANK / "bank_index.json", tmp_path / "bank_index.json")
    with pytest.raises(FileNotFoundError):
        SnapshotBank(tmp_path)


def test_hash_tampering_is_rejected(tmp_path):
    import shutil
    shutil.copyfile(BANK / "manifest.json", tmp_path / "manifest.json")
    (tmp_path / "bank_index.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        SnapshotBank(tmp_path)
