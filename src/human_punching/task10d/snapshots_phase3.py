"""Phase 3A: the entire accepted velocity domain, FK and a train-only design set.

This module performs no physics integration or placement/coverage search. The
existing derived reset's five mechanical/action-history arrays retain their
original bytes; the full source context remains available through shard/row IDs.
Only the accepted scene registration is reused from Task 10C's target pool; none
of that pool's state selection, height window or command exclusions is reused.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .g1 import BANK, BUNDLE, PROJECT, RESET_KEYS, TARGET_POOL, _imports, transform_snapshot
from human_punching.task10c.targets import TargetPool, canonical_from_collection, sha256

COUNTS = {"train": 23135, "validation": 23331, "test": 23335}
SPLITS = tuple(COUNTS)
DEFAULT_OUTPUT = PROJECT / "outputs/task10d/phase3/snapshots"
SCHEMA = "task10d_phase3_snapshot_geometry_v1"
FEATURE_NAMES = (
    "head_height_m", "head_forward_offset_m", "head_left_offset_m",
    "torso_pitch_rad", "torso_roll_rad", "sin_waist_heading", "cos_waist_heading",
    "camera_head_forward_offset_m", "camera_head_left_offset_m", "camera_head_up_offset_m",
    "camera_optical_pitch_rad", "camera_optical_roll_rad", "pelvis_pitch_rad", "pelvis_roll_rad",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _verified_json(root: Path, name: str, record: dict) -> dict:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
        raise ValueError(f"Source identity mismatch: {name}")
    return json.loads(path.read_text())


def _rotation_rpy(rotation: np.ndarray) -> np.ndarray:
    """ZYX roll/pitch/yaw coordinates; axes remain full 3D in saved matrices."""
    r = np.asarray(rotation)
    return np.stack((np.arctan2(r[..., 2, 1], r[..., 2, 2]),
                     np.arctan2(-r[..., 2, 0], np.hypot(r[..., 0, 0], r[..., 1, 0])),
                     np.arctan2(r[..., 1, 0], r[..., 0, 0])), -1)


def selection_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Posture/camera-sensitive features in the G1 pelvis-heading frame."""
    yaw = arrays["pelvis_heading_rad"]
    c, s = np.cos(yaw), np.sin(yaw)
    heading = np.zeros((len(yaw), 3, 3))
    heading[:, 0, 0] = c; heading[:, 1, 0] = s
    heading[:, 0, 1] = -s; heading[:, 1, 1] = c; heading[:, 2, 2] = 1
    to_local = lambda vector: np.einsum("nji,nj->ni", heading, vector)
    head_offset = to_local(arrays["head_world"] - arrays["pelvis_world"])
    camera_head = to_local(arrays["head_world"] - arrays["camera_position_world"])
    torso = _rotation_rpy(np.einsum("nji,njk->nik", heading, arrays["torso_rotation"]))
    pelvis = _rotation_rpy(np.einsum("nji,njk->nik", heading, arrays["pelvis_rotation"]))
    optical = np.einsum("nji,njk->nik", heading, arrays["baseline_camera_rotation"])
    # Forward direction gives pitch; right axis roll is explicitly camera-sensitive.
    camera_pitch = np.arctan2(optical[:, 2, 2], np.hypot(optical[:, 0, 2], optical[:, 1, 2]))
    camera_roll = np.arctan2(optical[:, 2, 0], np.hypot(optical[:, 0, 0], optical[:, 1, 0]))
    return np.column_stack((arrays["head_world"][:, 2], head_offset[:, :2],
        torso[:, 1], torso[:, 0], np.sin(torso[:, 2]), np.cos(torso[:, 2]),
        camera_head, camera_pitch, camera_roll, pelvis[:, 1], pelvis[:, 0]))


def deterministic_training_subset(ids: np.ndarray, features: np.ndarray, *,
                                  count: int = 1000, nominal_id: int = 53874) -> np.ndarray:
    """Nominal + feature extrema + deterministic maximin posture-space design.

    Features are robustly scaled by train-only p95-p05. Ties use ascending original
    snapshot ID. Every TRAIN state participates; outcomes and primitive reach do
    not affect selection. Repeated physical states are retained in the domain.
    """
    ids, features = np.asarray(ids), np.asarray(features, dtype=np.float64)
    if ids.ndim != 1 or features.ndim != 2 or len(ids) != len(features):
        raise ValueError("Expected original IDs and feature matrix with matching rows")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(features).all():
        raise ValueError("Selection needs distinct IDs and finite features")
    if not 0 < count <= len(ids) or nominal_id not in ids:
        raise ValueError("Selection count/nominal snapshot is outside TRAIN")
    order = np.argsort(ids, kind="stable")
    ordered = features[order]
    span = np.quantile(ordered, .95, axis=0) - np.quantile(ordered, .05, axis=0)
    full_span = np.ptp(ordered, axis=0)
    # Sparse but real extremes still participate when both central quantiles
    # coincide; only actually constant features contribute zero.
    varying = full_span > 1e-7
    span = np.where(span > 1e-7, span, full_span)
    normalized = np.zeros_like(ordered)
    normalized[:, varying] = ((ordered[:, varying] - np.median(ordered[:, varying], axis=0))
                              / span[varying])
    seeds = [int(np.flatnonzero(ids[order] == nominal_id)[0])]
    for column in np.flatnonzero(varying):
        seeds.extend((int(np.argmin(normalized[:, column])), int(np.argmax(normalized[:, column]))))
    chosen: list[int] = []
    minimum = np.full(len(ids), np.inf)
    for row in seeds:
        if row not in chosen and len(chosen) < count:
            chosen.append(row)
            minimum = np.minimum(minimum, np.sum((normalized - normalized[row]) ** 2, axis=1))
            minimum[chosen] = -np.inf
    while len(chosen) < count:
        row = int(np.argmax(minimum))
        chosen.append(row)
        minimum = np.minimum(minimum, np.sum((normalized - normalized[row]) ** 2, axis=1))
        minimum[chosen] = -np.inf
    return order[np.asarray(chosen)]


def _domain_and_outcomes(bank) -> tuple[dict[str, np.ndarray], dict]:
    """Recompute eligibility from all 317,520 accepted source outcome rows."""
    from boxing.snapshots.contracts import velocity_eligibility, VELOCITY_SCHEMA
    ids = bank.compact_ids
    fields = ("replica_id", "source_group_id", "rollout_id", "seed", "split_id", "command_id",
              "batch_id", "batch_seed", "time_index", "time_seconds", "actual_velocity",
              "requested_command", "control_step")
    retained: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    source_records = []
    for split_id, split in enumerate(SPLITS):
        if len(bank.eligible[split]) != COUNTS[split]:
            raise ValueError(f"Accepted velocity-domain count changed for {split}")
        name = f"replicas/{split_id:03d}/index/shards/00000000.json"
        desc = _verified_json(bank.root, name, bank.manifest["files"][name])
        payload_name = str(Path(name).parent / desc["payload"]["file"])
        payload = (bank.root / payload_name).resolve(strict=True)
        if (not payload.is_relative_to(bank.root)
                or sha256(payload) != desc["payload"]["sha256"]
                or sha256(payload) != bank.manifest["files"][payload_name]["sha256"]):
            raise ValueError("Accepted source outcome payload changed")
        with np.load(payload, allow_pickle=False) as source:
            rows = {key: source[key].copy() for key in source.files}
        mask = velocity_eligibility(rows["actual_velocity"],
            valid=(rows["status"] == 1) & (rows["rejection_reason"] == 0))
        if not np.array_equal(mask, rows["eligible"]):
            raise ValueError("Stored eligibility differs from accepted measured-velocity rule")
        if not np.array_equal(rows["slot_id"][mask], bank.eligible[split]):
            raise ValueError("Source eligibility IDs differ from original bank split")
        if not np.all(rows["split_id"] == split_id) or not np.all(rows["source_group_id"] == split_id):
            raise ValueError("Original split/group provenance changed")
        for key in fields:
            retained[key].append(rows[key][mask])
        source_records.append(dict(split=split, path=payload_name,
            sha256=desc["payload"]["sha256"], stored_rows=len(mask), eligible_rows=int(mask.sum())))
    arrays = {key: np.concatenate(parts) for key, parts in retained.items()}
    expected = np.concatenate([bank.eligible[split] for split in SPLITS])
    if not np.array_equal(ids, expected):
        raise ValueError("Compact reset rows do not preserve original ordered domain IDs")
    arrays.update(snapshot_id=ids.copy(),
        shard_ordinal=bank.locations["shard_ordinal"][ids].copy(),
        row_in_shard=bank.locations["row_in_shard"][ids].copy())
    return arrays, dict(counts=COUNTS, total=len(ids), source_outcomes=source_records,
        eligibility=VELOCITY_SCHEMA,
        rule="stored valid source row and each clean [vx_body, vy_body, native pelvis gyro z] within inclusive [-0.5,+0.5]; no arrival/command/Task10B filter")


def _model_and_calibration(bank):
    """Load actual installed G1 geometry and accepted torso-mounted camera."""
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    from boxing.perception.rectified_camera import CAMERA
    # The existing verifier checks selected G1 assets/model/action identity. It
    # does not construct a CUDA environment or change any physics settings.
    from .g1 import execution_dependencies
    dependencies = execution_dependencies()
    fp = dependencies["model_action_fingerprint"]
    asset_root = G1_XML.parent.parent
    for relative, expected in fp["g1_asset_sha256"].items():
        if sha256(asset_root / relative) != expected:
            raise ValueError(f"Installed G1 asset differs from selected runtime: {relative}")
    if bank.manifest["dependencies"]["g1_assets_sha256"] != dependencies["original_dependencies"]["g1_assets_sha256"]:
        raise ValueError("Snapshot and runtime G1 geometry differ")
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.body("torso_link").add_camera(name="task10d_fk_camera", pos=CAMERA.mount_position,
        quat=CAMERA.mujoco_quaternion_wxyz, fovy=CAMERA.vertical_fov_deg,
        resolution=(CAMERA.width, CAMERA.height))
    model = spec.compile()
    if (model.nq, model.nv) != (36, 35):
        raise ValueError("Expected selected G1 qpos36/qvel35")
    if [model.joint(i).name for i in range(1, model.njnt)] != fp["joint_order"]:
        raise ValueError("G1 joint order differs from accepted snapshot bank")
    if model.geom_size[model.geom("head_collision").id, 0] != .06:
        raise ValueError("Head sphere radius changed")
    pool = TargetPool.load(TARGET_POOL)
    scene = pool.arrays["scene_from_canonical"][0].copy()
    return model, scene, dict(g1_xml=str(G1_XML), g1_xml_sha256=sha256(G1_XML),
        g1_assets_sha256=bank.manifest["dependencies"]["g1_assets_sha256"],
        source_calibration=str(TARGET_POOL), source_calibration_sha256=sha256(TARGET_POOL),
        calibration_family="jab; shared accepted Phase1/2 combined scene", scene_from_canonical=scene.tolist(),
        calibration_scope="Only existing whole-body planar registration; none of the Task10C selected IDs or central-region eligibility is used",
        camera=CAMERA.record(), nominal_bundle=str(BUNDLE))


def _duplicate_summary(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Exact source derived-reset duplicates, without claiming independence."""
    state = np.concatenate([arrays[key] for key in RESET_KEYS], axis=1)
    hashes = np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in state], dtype="U64")
    _, first, inverse, counts = np.unique(hashes, return_index=True, return_inverse=True, return_counts=True)
    group = arrays["snapshot_id"][first][inverse]
    return group, dict(exact_reset_state_unique=int(len(first)),
        exact_reset_state_duplicate_rows=int(len(state) - len(first)),
        exact_reset_state_repeated_groups=int(np.count_nonzero(counts > 1)),
        comparison="byte-identical original qpos, qvel and all three raw action history fields",
        trajectories_by_split={split: int(len(np.unique(arrays["rollout_id"][arrays["split_id"] == i])))
                               for i, split in enumerate(SPLITS)},
        independence="Stored snapshots are repeated times within command rollouts and seed-grouped replicas; no row-level independence claim")


def prepare(output: Path = DEFAULT_OUTPUT, *, subset_count: int = 1000) -> dict:
    """Verify the entire bank domain and compute static FK for every eligible row."""
    _imports()
    import mujoco
    from boxing.artifacts import artifact_path
    from boxing.single_punch.artifacts import CompactBank
    from boxing.snapshots.contracts import actual_base_velocities
    start = time.monotonic()
    output = output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase3").resolve()):
        raise ValueError("Snapshot preparation outputs must stay in Phase3")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("geometry.npz", "geometry.json", "training_subset.json")):
        raise FileExistsError("Snapshot preparation cannot overwrite an existing result")
    bank = CompactBank(artifact_path("reset_states"), bank_root=BANK)
    arrays, domain = _domain_and_outcomes(bank)
    arrays.update({key: value.copy() for key, value in bank.reset_arrays.items()})
    measured = actual_base_velocities(arrays[RESET_KEYS[0]], arrays[RESET_KEYS[1]])
    velocity_error = float(np.max(np.abs(measured - arrays["actual_velocity"])))
    if velocity_error > 2e-6:
        raise ValueError(f"Actual-state velocity and accepted outcome disagree: {velocity_error}")
    model, scene, identity = _model_and_calibration(bank)
    data = mujoco.MjData(model)
    head_id, pelvis_id, torso_id, camera_id = (model.geom("head_collision").id,
        model.body("pelvis").id, model.body("torso_link").id, model.camera("task10d_fk_camera").id)
    n = len(arrays["snapshot_id"])
    for name in ("head_world", "pelvis_world", "torso_world", "camera_position_world"):
        arrays[name] = np.empty((n, 3))
    for name in ("pelvis_rotation", "torso_rotation", "baseline_camera_rotation"):
        arrays[name] = np.empty((n, 3, 3))
    arrays["world_from_collection"] = np.empty((n, 4, 4))
    for row, (qpos, qvel) in enumerate(zip(arrays[RESET_KEYS[0]], arrays[RESET_KEYS[1]], strict=True)):
        transform = scene @ canonical_from_collection(qpos)
        q, v = transform_snapshot(qpos, qvel, transform)
        data.qpos[:] = q; data.qvel[:] = v
        # No forward dynamics, integration, settling, velocity zeroing or history.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        arrays["world_from_collection"][row] = transform
        arrays["head_world"][row] = data.geom_xpos[head_id]
        arrays["pelvis_world"][row] = data.xpos[pelvis_id]
        arrays["torso_world"][row] = data.xpos[torso_id]
        arrays["camera_position_world"][row] = data.cam_xpos[camera_id]
        arrays["pelvis_rotation"][row] = data.xmat[pelvis_id].reshape(3, 3)
        arrays["torso_rotation"][row] = data.xmat[torso_id].reshape(3, 3)
        optical = data.cam_xmat[camera_id].reshape(3, 3).copy(); optical[:, 1:] *= -1
        arrays["baseline_camera_rotation"][row] = optical
        if data.time != 0:
            raise AssertionError("FK advanced simulation time")
    arrays["pelvis_heading_rad"] = np.arctan2(arrays["pelvis_rotation"][:, 1, 0], arrays["pelvis_rotation"][:, 0, 0])
    arrays["features"] = selection_features(arrays)
    arrays["duplicate_reset_group_snapshot_id"], duplicates = _duplicate_summary(arrays)
    training_rows = np.flatnonzero(arrays["split_id"] == 0)
    chosen = training_rows[deterministic_training_subset(arrays["snapshot_id"][training_rows],
        arrays["features"][training_rows], count=subset_count)]
    arrays["training_subset_indices"] = chosen
    nominal = int(np.flatnonzero(arrays["snapshot_id"] == 53874)[0])
    np.testing.assert_allclose(arrays["head_world"][nominal], [1.061174750, .260344207, 1.228431463], atol=5e-6, rtol=0)
    payload = output / "geometry.npz"
    np.savez_compressed(payload, **arrays)
    subset = dict(schema="task10d_phase3_training_design_v1", count=len(chosen), nominal_snapshot_id=53874,
        snapshot_ids=arrays["snapshot_id"][chosen].tolist(), geometry_row_indices=chosen.tolist(),
        feature_names=list(FEATURE_NAMES),
        selection="nominal + every varying feature's min/max + deterministic maximin on train-only p95-p05-scaled posture/camera features; ID-stable tie break",
        training_feature_min=arrays["features"][training_rows].min(0).tolist(),
        training_feature_max=arrays["features"][training_rows].max(0).tolist(),
        selected_feature_min=arrays["features"][chosen].min(0).tolist(),
        selected_feature_max=arrays["features"][chosen].max(0).tolist(),
        limitation="Deliberately posture-covering design sample, not an iid or frequency-representative sample")
    _write_json(output / "training_subset.json", subset)
    metadata = dict(schema=SCHEMA, payload_sha256=sha256(payload), domain=domain, duplicates=duplicates,
        bank_manifest_sha256=bank.manifest_sha256, bank_index_sha256=bank.index_sha256,
        reset_cache=str(artifact_path("reset_states")), reset_cache_sha256=bank.reset_meta["payload"]["sha256"],
        model_and_calibration=identity, source_state_keys=list(RESET_KEYS),
        full_source_context="Every original full-context shard remains unchanged: resolve shard_ordinal/row_in_shard through the identity-pinned bank_index.json; five exact arrays here are all fields consumed by the accepted derived Task08/Fresh96 reset, not an exact source simulator continuation",
        state_restoration="Call load_snapshot_rows from this module; it returns original qpos/qvel/raw histories and whole-body planar transform; real qvel is rotated only for world linear velocity, never zeroed",
        snapshot_count=n, subset_count=len(chosen), feature_names=list(FEATURE_NAMES),
        actual_velocity_reconstruction_max_abs_error=velocity_error,
        nominal_snapshot_id=53874, nominal_head_world=arrays["head_world"][nominal].tolist(),
        calibration_heading_rad=float(arrays["pelvis_heading_rad"][nominal]),
        head_world_min=arrays["head_world"].min(0).tolist(), head_world_max=arrays["head_world"].max(0).tolist(),
        physics_steps=0, simulation_time_s=0.0, coverage_search_performed=False,
        application_wall_s=time.monotonic() - start)
    _write_json(output / "geometry.json", metadata)
    return metadata


@lru_cache(maxsize=2)
def load_geometry(path: str | Path = DEFAULT_OUTPUT / "geometry.npz") -> tuple[dict, dict]:
    path = Path(path).resolve(strict=True)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["schema"] != SCHEMA or sha256(path) != metadata["payload_sha256"]:
        raise ValueError("Phase3 snapshot geometry identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    for value in arrays.values():
        value.setflags(write=False)
    return arrays, metadata


def load_snapshot_rows(snapshot_ids, *, path: str | Path = DEFAULT_OUTPUT / "geometry.npz") -> dict:
    """Drop-in accepted derived reset rows with the full-domain calibration."""
    arrays, metadata = load_geometry(path)
    ids = np.asarray(snapshot_ids, dtype=np.int64)
    rows = np.searchsorted(arrays["snapshot_id"], ids)
    if np.any(rows >= len(arrays["snapshot_id"])) or not np.array_equal(arrays["snapshot_id"][rows], ids):
        raise ValueError("Requested snapshot outside accepted velocity domain")
    return dict(snapshot_ids=ids.copy(), arrays={key: arrays[key][rows].copy() for key in RESET_KEYS},
        transforms=arrays["world_from_collection"][rows].copy(), split_id=arrays["split_id"][rows].copy(),
        bank_manifest_sha256=metadata["bank_manifest_sha256"], geometry=str(Path(path).resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset-count", type=int, default=1000)
    args = parser.parse_args()
    result = prepare(args.output, subset_count=args.subset_count)
    print(json.dumps({key: result[key] for key in ("domain", "duplicates", "snapshot_count", "subset_count", "application_wall_s")}, indent=2))


if __name__ == "__main__":
    main()
