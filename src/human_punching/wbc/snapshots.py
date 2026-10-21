"""Read-only Task 05 payload adapter for frozen launch geometry, not restoration.

Only bank-index members are selectable. Pilot/capacity/validation trajectories
are not samples. Original IDs, replica splits, and rollout groups are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .references import sha256


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Snapshot manifest path escapes the selected bank")
    return path


def _verified(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Snapshot hash mismatch: {path.name}")


def _verify_array(array: np.ndarray, spec: dict[str, Any], name: str) -> None:
    if list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]:
        raise ValueError(f"Snapshot array schema mismatch: {name}")
    if "sha256" in spec and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Snapshot array hash mismatch: {name}")


@dataclass(frozen=True)
class SnapshotState:
    slot_id: int
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    replica_id: int
    command_id: int
    time_index: int
    original_split: str
    source_group_id: str
    payload_sha256: str
    eligible: bool
    original_source_group_id: int = -1
    rollout_id: int = -1
    seed: int = -1


class SnapshotBank:
    """Lazy per-shard verification of a selected bank with actual NPZ payloads."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if not self.manifest.get("selected") or not self.manifest.get("promoted"):
            raise ValueError("Expected a selected, promoted snapshot bank")
        index_path = _inside(self.directory, self.manifest["bank_index"])
        _verified(index_path, self.manifest["bank_index_sha256"])
        self.index = json.loads(index_path.read_text())
        locations = self.index["locations"]
        location_path = _inside(self.directory, locations["path"])
        _verified(location_path, locations["sha256"])
        with np.load(location_path, allow_pickle=False) as source:
            self.locations = {name: source[name].copy() for name in locations["arrays"]}
        for name, spec in locations["arrays"].items():
            _verify_array(self.locations[name], spec, name)
            self.locations[name].setflags(write=False)
        slot_ids = self.locations["slot_id"]
        if not np.array_equal(slot_ids, np.arange(len(slot_ids))):
            raise ValueError("Expected the original contiguous slot-id addressing")
        self._cache: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}

    def eligible_ids(self, split: str) -> np.ndarray:
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown original snapshot split")
        return self.locations[f"eligible_slot_ids_{split}"].copy()

    def _shard(self, ordinal: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if ordinal in self._cache:
            return self._cache[ordinal]
        entry = self.index["shards"][ordinal]
        if entry["ordinal"] != ordinal:
            raise ValueError("Shard ordinal mismatch")
        manifest_path = _inside(self.directory, entry["manifest"])
        _verified(manifest_path, entry["manifest_sha256"])
        description = json.loads(manifest_path.read_text())
        path = _inside(self.directory, entry["payload"])
        if description["payload"]["sha256"] != entry["payload_sha256"]:
            raise ValueError("Conflicting snapshot payload hashes")
        if path.name != description["payload"]["file"]:
            raise ValueError("Conflicting snapshot payload paths")
        _verified(path, entry["payload_sha256"])
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in ("sim/data/qpos", "sim/data/qvel")}
        for name, values in arrays.items():
            _verify_array(values, description["arrays"][name], name)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite snapshot physical state")
            values.setflags(write=False)
        # Bound memory: selected rows need only one full-context shard at a time.
        self._cache.clear()
        self._cache[ordinal] = arrays, description
        return arrays, description

    def state(self, slot_id: int, *, require_eligible: bool = True) -> SnapshotState:
        if isinstance(slot_id, bool) or not isinstance(slot_id, (int, np.integer)):
            raise ValueError("slot_id must be an integer")
        if not 0 <= slot_id < len(self.locations["slot_id"]):
            raise ValueError("Snapshot slot is outside the selected bank")
        ordinal = int(self.locations["shard_ordinal"][slot_id])
        row = int(self.locations["row_in_shard"][slot_id])
        entry = self.index["shards"][ordinal]
        split = entry["split"]
        eligible = bool(np.any(self.locations[f"eligible_slot_ids_{split}"] == slot_id))
        if require_eligible and not eligible:
            raise ValueError("Snapshot is outside the bank's original measured-velocity eligibility")
        arrays, description = self._shard(ordinal)
        if not 0 <= row < len(arrays["sim/data/qpos"]):
            raise ValueError("Snapshot row is outside its shard")
        replica = int(self.locations["replica_id"][slot_id])
        command = int(self.locations["command_id"][slot_id])
        if replica != entry["replica_id"] or command != description["metadata"]["command_ids"][row]:
            raise ValueError("Snapshot index disagrees with original replica/command")
        outcomes = description["metadata"].get("slot_outcomes", {})
        if "slot_id" in outcomes and int(outcomes["slot_id"]["values"][row]) != slot_id:
            raise ValueError("Snapshot row maps to a different original slot")
        # Keep every replica and neighboring time sample of one command grouped.
        # Preserve the bank's own coarser source-group/rollout IDs separately.
        group = f"command:{command}"
        def original_id(name: str) -> int:
            return int(outcomes[name]["values"][row]) if name in outcomes else -1
        return SnapshotState(int(slot_id), tuple(arrays["sim/data/qpos"][row].tolist()),
                             tuple(arrays["sim/data/qvel"][row].tolist()), replica,
                             command, int(self.locations["time_index"][slot_id]),
                             split, group, entry["payload_sha256"], eligible,
                             original_id("source_group_id"), original_id("rollout_id"),
                             original_id("seed"))


def head_point_state(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray,
                     head_geom: str = "head_collision") -> tuple[np.ndarray, np.ndarray]:
    """Head sphere center and point velocity, including omega cross offset.

    This performs FK and a point Jacobian only. It does not step physics or claim
    exact Task05 continuation from qpos/qvel alone.
    """
    qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
        raise ValueError("Snapshot state does not match supplied G1 model")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Snapshot state must be finite")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, head_geom)
    if geom_id < 0:
        raise ValueError(f"Missing declared head geom: {head_geom}")
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE or not np.isclose(model.geom_size[geom_id, 0], .06):
        raise ValueError("Expected the accepted .06 m spherical G1 head proxy")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    point = data.geom_xpos[geom_id].copy()
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, point, int(model.geom_bodyid[geom_id]))
    return point, jp @ qvel


def load_verified_g1_model(asset_root: Path, bank: SnapshotBank,
                           nominal_plant_path: Path) -> tuple[mujoco.MjModel, dict[str, Any]]:
    """Bind raw G1 geometry to the accepted bank asset fingerprint before FK.

    The standalone asset is adequate for geometric FK/Jacobians, whose root and
    scalar coordinates are checked by name. It is deliberately not described as
    an exact restored simulation with controller caches and per-world constants.
    """
    plant = json.loads(Path(nominal_plant_path).read_text())
    fingerprint = plant["fingerprint"]
    unsigned = dict(fingerprint)
    recorded = unsigned.pop("fingerprint_sha256")
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical(unsigned)).hexdigest() != recorded:
        raise ValueError("Accepted model/action fingerprint digest mismatch")
    assets = fingerprint["g1_asset_sha256"]
    if hashlib.sha256(canonical(assets)).hexdigest() != bank.manifest["dependencies"]["g1_assets_sha256"]:
        raise ValueError("G1 assets do not match the snapshot bank's model provenance")
    root = Path(asset_root).resolve()
    for relative, expected in assets.items():
        _verified(_inside(root, relative), expected)
    model_path = _inside(root, "xmls/g1.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv) != (36, 35):
        raise ValueError("Accepted G1 geometry must have nq=36, nv=35")
    joints = [model.joint(i).name for i in range(1, model.njnt)]
    if joints != fingerprint["joint_order"]:
        raise ValueError("G1 coordinate order differs from the bank model")
    return model, {"asset_fingerprint_sha256": bank.manifest["dependencies"]["g1_assets_sha256"],
                   "model_sha256": sha256(model_path), "asset_count": len(assets),
                   "nominal_plant_sha256": sha256(Path(nominal_plant_path)),
                   "usage": "FK and point Jacobian only; not simulation restoration"}


def launch_snapshot(state: SnapshotState, model_path: Path, *,
                    expected_model_sha256: str, human_origin_w: tuple[float, float, float],
                    human_quat_wxyz: tuple[float, float, float, float],
                    head_geom: str = "head_collision"):
    """Freeze full G1 launch pose/velocity and derived geometry in a contract."""
    from .contracts import LaunchSnapshot

    model_path = Path(model_path).resolve()
    _verified(model_path, expected_model_sha256)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    head, velocity = head_point_state(model, np.asarray(state.qpos),
                                      np.asarray(state.qvel), head_geom)
    return LaunchSnapshot(snapshot_id=str(state.slot_id), g1_qpos=state.qpos,
                          g1_qvel=state.qvel, head_position_w=tuple(head),
                          head_velocity_w=tuple(velocity), human_origin_w=human_origin_w,
                          human_quat_wxyz=human_quat_wxyz,
                          source_group_id=state.source_group_id)
