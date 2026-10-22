"""Small frozen-head target pool extracted from the selected G1 snapshot bank.

This is geometric FK, never restoration or stepping of the G1 controller. Scene
transforms remove only collection yaw and ground-plane translation; root tilt,
posture offsets, and the original head height are preserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SPLITS = ("train", "validation", "test")
SCHEMA = "task10c_snapshot_head_pool_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def target_subset_indices(points: np.ndarray, fraction: float) -> np.ndarray:
    """Central real snapshots ranked by XY distance, without moving any point."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('Expected nonempty target points [N,3]')
    if not 0 < fraction <= 1:
        raise ValueError('Target fraction must lie in (0,1]')
    distances = np.linalg.norm(points[:, :2] - np.median(points[:, :2], axis=0), axis=1)
    return np.argsort(distances, kind='stable')[:max(1, int(np.ceil(len(points) * fraction)))]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Source hash mismatch: {path}")


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes the selected snapshot bank")
    return path


def _output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("Task 10C output must remain inside human-punching")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def yaw_transform(yaw: float, translation: np.ndarray) -> np.ndarray:
    """Homogeneous rigid transform with yaw only and an explicit translation."""
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def canonical_from_collection(qpos: np.ndarray) -> np.ndarray:
    """Remove projected root heading and XY, preserving roll/pitch and height.

    The origin is the saved root's projection onto the collection z=0 plane.
    Heading is the azimuth of the root's forward X axis, not its full rotation.
    """
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 1 or len(qpos) < 7 or not np.isfinite(qpos).all():
        raise ValueError("Expected one finite free-root qpos")
    quat = qpos[3:7]
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Root quaternion has zero norm")
    w, x, y, z = quat / norm
    forward_xy = np.array((1 - 2 * (y*y + z*z), 2 * (x*y + w*z)))
    if np.linalg.norm(forward_xy) < 1e-8:
        raise ValueError("Vertical root-forward axis has no ground-plane heading")
    yaw = np.arctan2(forward_xy[1], forward_xy[0])
    transform = yaw_transform(-yaw, np.zeros(3))
    transform[:3, 3] = -transform[:3, :3] @ np.array((*qpos[:2], 0.0))
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", transform[..., :3, :3], points) + transform[..., :3, 3]


@dataclass(frozen=True)
class TargetPool:
    """Numeric pool and scene transforms, aligned with original snapshot IDs."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> TargetPool:
        path = Path(path).resolve()
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["schema"] != SCHEMA:
            raise ValueError("Unsupported target-pool schema")
        _verified(path, metadata["payload_sha256"])
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        n = len(arrays["snapshot_id"])
        if arrays["canonical_head"].shape != (n, 3):
            raise ValueError("Malformed target pool")
        if arrays["canonical_from_collection"].shape != (n, 4, 4):
            raise ValueError("Malformed source scene transforms")
        if arrays["scene_from_canonical"].shape != (len(FAMILIES), 4, 4):
            raise ValueError("Malformed family scene transforms")
        if len(np.unique(arrays["snapshot_id"])) != n:
            raise ValueError("Duplicate source snapshot IDs")
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite target pool")
            value.setflags(write=False)
        return cls(arrays, metadata)

    def _mask(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(f"Unknown snapshot split: {split}")
        return self.arrays["split_id"] == SPLITS.index(split)

    def points(self, split: str, family: str) -> np.ndarray:
        """World XYZ before adding vector-environment origins; returns a copy."""
        if family not in FAMILIES:
            raise ValueError(f"Unknown punch family: {family}")
        return transform_points(self.arrays["scene_from_canonical"][FAMILIES.index(family)],
                                self.arrays["canonical_head"][self._mask(split)])

    def snapshot_ids(self, split: str) -> np.ndarray:
        """IDs aligned with ``points(split, family)`` for sampling provenance."""
        return self.arrays["snapshot_id"][self._mask(split)].copy()

    def transforms(self, split: str, family: str) -> np.ndarray:
        """Exact world-from-original-collection transforms for sampled targets."""
        return (self.arrays["scene_from_canonical"][FAMILIES.index(family)]
                @ self.arrays["canonical_from_collection"][self._mask(split)])


def _read_sources(bank_dir: Path, candidates: dict[str, np.ndarray]):
    """Read only selected qpos rows, grouping access by full-context source shard."""
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    index_path = _inside(bank_dir, manifest["bank_index"])
    _verified(index_path, manifest["bank_index_sha256"])
    index = json.loads(index_path.read_text())
    location_path = _inside(bank_dir, index["locations"]["path"])
    _verified(location_path, index["locations"]["sha256"])
    with np.load(location_path, allow_pickle=False) as source:
        locations = {key: source[key].copy() for key in source.files}
    rows: dict[int, np.ndarray] = {}
    sources = {}
    selected = np.concatenate(list(candidates.values()))
    for ordinal in np.unique(locations["shard_ordinal"][selected]):
        entry = index["shards"][int(ordinal)]
        description_path = _inside(bank_dir, entry["manifest"])
        _verified(description_path, entry["manifest_sha256"])
        description = json.loads(description_path.read_text())
        payload_path = _inside(bank_dir, entry["payload"])
        _verified(payload_path, entry["payload_sha256"])
        if description["payload"]["sha256"] != entry["payload_sha256"]:
            raise ValueError("Conflicting source shard identities")
        with np.load(payload_path, allow_pickle=False) as source:
            qpos = source["sim/data/qpos"]
        spec = description["arrays"]["sim/data/qpos"]
        if list(qpos.shape) != spec["shape"] or qpos.dtype.str != spec["dtype"]:
            raise ValueError("Snapshot qpos schema mismatch")
        slots = selected[locations["shard_ordinal"][selected] == ordinal]
        for slot in slots:
            row = int(locations["row_in_shard"][slot])
            if slot not in candidates[entry["split"]]:
                raise ValueError("Source snapshot split mismatch")
            rows[int(slot)] = qpos[row].astype(float, copy=True)
        sources[str(int(ordinal))] = {
            "path": entry["payload"], "sha256": entry["payload_sha256"],
            "split": entry["split"], "manifest_sha256": entry["manifest_sha256"],
        }
    return rows, locations, manifest, sources


def _load_g1(asset_root: Path, nominal_plant: Path, manifest: dict):
    import mujoco

    fingerprint = json.loads(nominal_plant.read_text())["fingerprint"]
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    unsigned = dict(fingerprint)
    recorded = unsigned.pop("fingerprint_sha256")
    if hashlib.sha256(canonical(unsigned)).hexdigest() != recorded:
        raise ValueError("Nominal G1 fingerprint digest mismatch")
    assets = fingerprint["g1_asset_sha256"]
    if hashlib.sha256(canonical(assets)).hexdigest() != manifest["dependencies"]["g1_assets_sha256"]:
        raise ValueError("G1 geometry does not match the snapshot bank fingerprint")
    for relative, expected in assets.items():
        _verified(_inside(asset_root, relative), expected)
    model = mujoco.MjModel.from_xml_path(str(asset_root / "xmls/g1.xml"))
    if (model.nq, model.nv) != (36, 35):
        raise ValueError("Expected selected 29-joint G1 geometry for snapshot FK")
    if [model.joint(i).name for i in range(1, model.njnt)] != fingerprint["joint_order"]:
        raise ValueError("G1 qpos joint order differs from the saved snapshot bank")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_collision")
    if geom < 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("Expected G1 head_collision sphere")
    if not np.isclose(model.geom_size[geom, 0], .06):
        raise ValueError("G1 head-region radius differs from 0.06 m")
    return model, geom, {"g1_assets_sha256": manifest["dependencies"]["g1_assets_sha256"],
                         "nominal_plant_sha256": sha256(nominal_plant),
                         "g1_xml_sha256": sha256(asset_root / "xmls/g1.xml")}


def build_target_pool(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Extract a reproducible modest pool; no simulation, hit filter, or IK search."""
    import mujoco

    bank_dir = Path(config["snapshot_bank"]).resolve()
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    index_path = _inside(bank_dir, manifest["bank_index"])
    _verified(index_path, manifest["bank_index_sha256"])
    index = json.loads(index_path.read_text())
    location_path = _inside(bank_dir, index["locations"]["path"])
    _verified(location_path, index["locations"]["sha256"])
    rng = np.random.default_rng(config["seed"])
    candidates = {}
    with np.load(location_path, allow_pickle=False) as source:
        for split in SPLITS:
            eligible = source[f"eligible_slot_ids_{split}"]
            count = min(len(eligible), int(config["counts"][split]) * 4)
            candidates[split] = rng.choice(eligible, count, replace=False)
    rows, locations, manifest, sources = _read_sources(bank_dir, candidates)
    model, geom, identity = _load_g1(Path(config["g1_asset_root"]).resolve(),
                                    Path(config["nominal_plant"]).resolve(), manifest)
    data = mujoco.MjData(model)
    extracted = {}
    for split, slots in candidates.items():
        collection, canonical, transforms = [], [], []
        for slot in slots:
            qpos = rows[int(slot)]
            if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
                raise ValueError("Invalid saved G1 qpos")
            data.qpos[:] = qpos
            # mj_kinematics refreshes geom transforms without solving contacts.
            mujoco.mj_kinematics(model, data)
            point = data.geom_xpos[geom].copy()
            transform = canonical_from_collection(qpos)
            collection.append(point)
            canonical.append(transform_points(transform, point))
            transforms.append(transform)
        extracted[split] = (np.asarray(collection), np.asarray(canonical), np.asarray(transforms))
    center = np.median(extracted["train"][1], axis=0)
    selected = {}
    statistics = {}
    for split, slots in candidates.items():
        collection, canonical, transforms = extracted[split]
        delta = canonical - center
        keep = np.flatnonzero((np.linalg.norm(delta[:, :2], axis=1) <= config["horizontal_radius"])
                              & (np.abs(delta[:, 2]) <= config["vertical_half_width"]))
        count = min(len(keep), int(config["counts"][split]))
        if not count:
            raise ValueError(f"No actual snapshot heads support requested central region: {split}")
        chosen = rng.choice(keep, count, replace=False)
        selected[split] = (slots[chosen], collection[chosen], canonical[chosen], transforms[chosen])
        statistics[split] = {"candidate_count": len(slots), "central_count": len(keep),
                             "retained_count": count, "canonical_min": canonical[chosen].min(0).tolist(),
                             "canonical_max": canonical[chosen].max(0).tolist()}
    scene_transforms = []
    for family in FAMILIES:
        scene = config["families"][family]
        transform = yaw_transform(scene["g1_yaw"], np.zeros(3))
        # Only XY placement is chosen; z=0 leaves all real snapshot heights intact.
        transform[:2, 3] = np.asarray(scene["head_center_xy"]) - (transform[:3, :3] @ center)[:2]
        scene_transforms.append(transform)
    slots = np.concatenate([selected[s][0] for s in SPLITS])
    arrays = {
        "snapshot_id": slots.astype(np.int64),
        "split_id": np.concatenate([np.full(len(selected[s][0]), i, dtype=np.int8)
                                     for i, s in enumerate(SPLITS)]),
        "collection_head": np.concatenate([selected[s][1] for s in SPLITS]),
        "canonical_head": np.concatenate([selected[s][2] for s in SPLITS]),
        "canonical_from_collection": np.concatenate([selected[s][3] for s in SPLITS]),
        "scene_from_canonical": np.asarray(scene_transforms),
        "source_qpos": np.asarray([rows[int(slot)] for slot in slots]),
        "shard_ordinal": locations["shard_ordinal"][slots],
        "row_in_shard": locations["row_in_shard"][slots],
        "replica_id": locations["replica_id"][slots],
        "command_id": locations["command_id"][slots],
        "time_index": locations["time_index"][slots],
    }
    output = _output(output)
    np.savez_compressed(output, **arrays)
    used_shards = {str(int(value)) for value in arrays["shard_ordinal"]}
    metadata = {
        "schema": SCHEMA, "payload_sha256": sha256(output), "config": config,
        "bank_index_sha256": sha256(index_path), "bank_manifest_sha256": sha256(bank_dir / "manifest.json"),
        "model_identity": identity, "families": list(FAMILIES), "splits": list(SPLITS),
        "training_candidate_median": center.tolist(), "statistics": statistics,
        "source_shards": {key: value for key, value in sources.items() if key in used_shards},
        "transform_rule": "collection head -> subtract root XY at z=0 -> inverse root heading yaw -> family scene yaw and XY translation; no Z translation",
        "geometry": "actual saved qpos and matching head_collision geom; 0.06 m radius; no G1 simulation",
        "limitations": ["Measured-velocity bank eligibility is not H2 reachability.",
                         "Central subset selection is independent of eventual policy hit or miss.",
                         "Head heights and posture offsets are never scaled or clipped.",
                         "Targets are non-solid frozen world points; no impact force is modeled."],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def place_from_references(pool_path: Path, reference_dir: Path, config_path: Path,
                          families: tuple[str, ...] = FAMILIES) -> dict[str, Any]:
    """Choose family separation from nominal glove strike XY without changing Z.

    Reuses the already-extracted actual G1 heads; no reachability or hit filtering.
    The remaining vertical offset to the demonstration is reported explicitly.
    """
    pool = TargetPool.load(pool_path)
    arrays = {key: value.copy() for key, value in pool.arrays.items()}
    metadata = pool.metadata
    config = metadata["config"]
    center = np.asarray(metadata["training_candidate_median"])
    placement = metadata.get("nominal_reference_placement", {})
    for family in families:
        index = FAMILIES.index(family)
        reference_path = reference_dir / f"{family}.npz"
        description = json.loads(reference_path.with_suffix(".json").read_text())
        active_hand = description["active_hand"]
        if active_hand not in ("left", "right"):
            raise ValueError(f"Invalid active hand for {family}")
        with np.load(reference_path, allow_pickle=False) as source:
            times = source["time_s"]
            glove = source["glove_pos_w"][:, 0 if active_hand == "left" else 1]
            strike = np.array([np.interp(description["strike_time_s"], times, glove[:, axis])
                               for axis in range(3)])
        if not np.isfinite(strike).all():
            raise ValueError(f"Nonfinite reference strike position for {family}")
        config["families"][family]["head_center_xy"] = strike[:2].tolist()
        transform = yaw_transform(config["families"][family]["g1_yaw"], np.zeros(3))
        transform[:2, 3] = strike[:2] - (transform[:3, :3] @ center)[:2]
        arrays["scene_from_canonical"][index] = transform
        placement[family] = {
            "reference_path": str(reference_path.resolve()),
            "reference_sha256": sha256(reference_path),
            "strike_time_s": description["strike_time_s"], "active_hand": active_hand,
            "reference_strike_glove_xyz": strike.tolist(),
            "target_candidate_center_xyz": transform_points(transform, center).tolist(),
            "reference_minus_target_height_m": float(strike[2] - center[2]),
        }
    metadata["nominal_reference_placement"] = placement
    metadata["pending_reference_placement"] = [family for family in FAMILIES if family not in placement]
    metadata["placement_rule"] = "Target candidate median XY coincides with family reference active-glove strike XY; real G1 head Z is unchanged."
    pool_path = _output(pool_path)
    np.savez_compressed(pool_path, **arrays)
    metadata["payload_sha256"] = sha256(pool_path)
    pool_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    _output(config_path).write_text(json.dumps(config, indent=2) + "\n")
    return placement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/task10c/targets.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/task10c/setup/target_pool.npz")
    parser.add_argument("--place-from-references", type=Path,
                        help="Update cached pool/config scene XY from four references; preserve actual head Z")
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=list(FAMILIES))
    args = parser.parse_args()
    if args.place_from_references:
        placement = place_from_references(args.output, args.place_from_references, args.config,
                                          tuple(args.families))
        print(json.dumps(placement, indent=2))
        return
    config = json.loads(args.config.read_text())
    result = build_target_pool(config, args.output)
    print(json.dumps({"path": str(args.output.resolve()), "statistics": result["statistics"]}, indent=2))


if __name__ == "__main__":
    main()
