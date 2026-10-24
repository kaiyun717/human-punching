"""Convert explicit reviewed source events with the established GMR/H2 adapter.

This offline CPU path performs no policy training, physics rollout, balance
qualification, target-bank expansion, or learned scheduling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from human_punching.task10c.references import arrays_from_qpos, quaternion_interpolate
from human_punching.task10c.retargeting import DuoBoxAdapter, ExplicitH2Retargeter
from human_punching.task10c.robot import GLOVE_OFFSETS, H2_XML, ROOT
from .source_survey import load_event


def convert_event(event: dict, manifest: dict, output: Path) -> dict:
    began = time.monotonic()
    destination = (output / event["event_id"]).resolve()
    destination.relative_to((ROOT / "outputs/task10d/phase2").resolve())
    if destination.exists():
        raise FileExistsError(f"Preserving existing conversion: {destination}")
    clip = load_event(event, Path(manifest["dataset_root"]))
    config_path = Path(manifest["retarget_config"])
    config = json.loads(config_path.read_text())
    family = event["family"]
    adapter = DuoBoxAdapter(clip, source_target_offsets_torso_m=config.get(
        "family_source_target_offsets_torso_m", {}).get(family, {}))
    solver = ExplicitH2Retargeter(config_path=config_path)
    for _ in range(12):
        solver.retarget(adapter.frame(0))
    heads = [int(solver.model.joint(name).qposadr[0]) for name in ("head_pitch_joint", "head_yaw_joint")]
    raw = []
    for i in range(len(clip.time_s)):
        q = solver.retarget(adapter.frame(i))
        q[heads] = 0
        solver.configuration.update(q)
        raw.append(q)
    raw = np.asarray(raw)
    processing = manifest["processing"]
    smooth = gaussian_filter1d(raw, sigma=processing["smoothing_sigma_source_frames"], axis=0, mode="nearest")
    smooth[:, 3:7] = raw[:, 3:7]
    t = np.arange(0., clip.time_s[-1] + 1e-8, 1 / processing["output_hz"])
    # Use the same uniform 50 Hz convention as the accepted selected references;
    # source event end may be <20 ms after the final retained output sample.
    qpos = np.stack([np.interp(t, clip.time_s, smooth[:, j]) for j in range(38)], axis=-1)
    qpos[:, 3:7] = quaternion_interpolate(clip.time_s, raw[:, 3:7], t)
    data = mujoco.MjData(solver.model)
    shifts = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(solver.model, data)
        sole_low = []
        for side in ("left", "right"):
            gid = solver.model.geom(side + "_foot_sole_collision").id
            rotation = data.geom_xmat[gid].reshape(3, 3)
            sole_low.append(data.geom_xpos[gid, 2] - np.dot(np.abs(rotation[2]), solver.model.geom_size[gid]))
        shift = processing["sole_clearance_m"] - min(sole_low)
        q[2] += shift
        shifts.append(shift)
    arrays = arrays_from_qpos(t, qpos)
    arrays["source_frames"] = clip.source_frames[0] + t * 120
    strike = (event["extension_frame"]-clip.source_frames[0]) / 120
    metadata = {"schema": "task10c_h2_reference_v1", "primitive_id": event["event_id"],
                "family": family, "active_hand": event["anatomical_active_hand"], "stance": event["stance"],
                "strike_time_s": strike, "end_time_s": float(t[-1]),
                "recovery_duration_s": float(t[-1]-strike), "terminal_guard_hold_s": processing["terminal_guard_hold_s"],
                "guard_start_time_s": 0., "guard_preparation_interval_s": [0., strike],
                "withdrawal_interval_s": [strike, float(t[-1])],
                "terminal_guard_interval_s": [max(strike, float(t[-1]-.08)), float(t[-1])],
                "annotation_rationale": "Explicit recorded event bounds; source reach maximum is an extension descriptor, never a frozen-target first-entry annotation. Output terminal guard is a source candidate, subject to full H2 review.",
                "source": clip.metadata, "source_event": event, "adapter": adapter.metadata,
                "provenance_category": "recorded_style_variant_with_H2_morphology_retargeting",
                "diversity_rationale": event["reason"],
                "asset_sha256": hashlib.sha256(H2_XML.read_bytes()).hexdigest(),
                "retarget_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "retarget_config_path": str(config_path.resolve()),
                "source_revisions": {"GMR": "bb1bbe40774794fceb2a7c579a3464a28e68c844", "h2_extension": "3d9a05d951b49e803d9d2c3555aed29a4b1e4b2b"},
                "solver": "GMR inherited two-stage Mink proxqp", "ik_dt_s": 1/120,
                "output_hz": processing["output_hz"], "quaternion_order": "wxyz",
                "qvel_root_angular_frame": "MuJoCo local root frame", "glove_radius_m": .08,
                "glove_attachment_offsets_m": GLOVE_OFFSETS,
                "smoothing_sigma_source_frames": processing["smoothing_sigma_source_frames"],
                "sole_ground_shift_range_m": [float(min(shifts)), float(max(shifts))],
                "head_coordinates": "Existing selected H2 adapter convention: pitch/yaw explicitly neutral and zero velocity.",
                "kinematic_scope": "Source-derived prescribed motion; no inverse dynamics, torque/contact feasibility or learned balance claim.",
                "generation_wall_s": time.monotonic()-began}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / f"{family}.npz", **arrays)
    (destination / f"{family}.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return {"event_id": event["event_id"], "path": str(destination/f"{family}.npz"),
            "sha256": hashlib.sha256((destination/f"{family}.npz").read_bytes()).hexdigest(),
            "wall_s": metadata["generation_wall_s"], "frames": len(t)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/task10d/phase2/survey/selected_events.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event", nargs="+")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = [x for x in manifest["events"] if args.event is None or x["event_id"] in args.event]
    if not selected or args.event and len(selected) != len(args.event):
        raise ValueError("Explicit event selection is empty or unknown")
    for event in selected:
        print(json.dumps(convert_event(event, manifest, args.output.resolve())), flush=True)


if __name__ == "__main__":
    main()
