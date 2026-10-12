"""Generate provenance-checked human punch meshes, trajectories, and movies."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .motion import CROP_SPECS, load_crop

PROJECT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT.parent / "DuoBox")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/preview")
    parser.add_argument(
        "--height", type=float, default=1.80, help="Neutral crown-to-sole height in metres"
    )
    parser.add_argument(
        "--padding", type=float, default=0.25, help="Recorded context on each side, seconds"
    )
    parser.add_argument(
        "--punches",
        nargs="+",
        choices=list(CROP_SPECS),
        default=["jab", "right_hook", "left_hook", "cross"],
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Export model/trajectories without OpenGL"
    )
    parser.add_argument(
        "--backend", choices=["osmesa", "egl"], default=os.environ.get("MUJOCO_GL", "osmesa")
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.height) or not 1.2 <= args.height <= 2.3:
        parser.error("height must be in [1.2, 2.3] metres")
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("outputs must stay inside the human-punching project")
    os.environ["MUJOCO_GL"] = args.backend
    # Set the OpenGL backend before importing MuJoCo or its renderer.
    import mujoco

    from .human import build_human
    from .retarget import retarget_clip, sha256

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    clips = {p: load_crop(args.dataset, p, args.padding) for p in args.punches}
    # One fixed morphology shared by all clips, independent of selection order.
    bind_clip = load_crop(args.dataset, "jab")
    human = build_human(output / "model", bind_clip.rest_offsets, height_m=args.height)
    model = mujoco.MjModel.from_xml_path(str(human["xml_path"]))
    result = {
        "schema": "human_punching_preview_v1",
        "height_m": args.height,
        "morphology": human,
        "model_sha256": sha256(Path(human["xml_path"])),
        "backend": args.backend if not args.no_render else "none",
        "runtime": {
            key: importlib.metadata.version(key)
            for key in ("mujoco", "numpy", "scipy", "imageio", "pillow")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_assets_sha256": {
            str(path.relative_to(output / "model")): sha256(path)
            for path in sorted((output / "model").rglob("*"))
            if path.is_file() and path.suffix in (".xml", ".obj")
        },
        "joint_layout": [
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                "type": int(model.jnt_type[j]),
                "qpos_address": int(model.jnt_qposadr[j]),
                "dof_address": int(model.jnt_dofadr[j]),
            }
            for j in range(model.njnt)
        ],
        "trajectory_contract": "MuJoCo free-root + ball-joint qpos/qvel; not WBC scalar-hinge joint_pos. Body arrays omit world body; use body_names.",
        "punches": {},
    }
    stills = []
    for punch, clip in clips.items():
        print(f"Retargeting {punch}: {len(clip.time_s)} source frames", flush=True)
        clip_dir = output / punch
        clip_dir.mkdir(parents=True, exist_ok=True)
        retargeted = retarget_clip(clip, model, float(human["scale"]))
        trajectory, metrics = retargeted["trajectory"], retargeted["metrics"]
        np.savez_compressed(clip_dir / "trajectory.npz", **trajectory)
        metrics["model_relative_path"] = "../model/human.xml"
        metrics["trajectory_sha256"] = sha256(clip_dir / "trajectory.npz")
        if not args.no_render:
            from .render import render_clip

            print(f"Rendering {punch} with {args.backend}", flush=True)
            metrics["media"] = render_clip(model, trajectory, metrics, clip_dir, args.height)
            stills.append(Path(metrics["media"]["still"]))
        (clip_dir / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
        result["punches"][punch] = metrics
        print(f"Finished {punch}; max FK error {metrics['joint_fk_max_error_m']:.3g} m", flush=True)
    if len(stills) == 4:
        from .render import make_overview

        make_overview(stills, output / "overview.png")
    result["elapsed_s"] = time.monotonic() - started
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed preview: {output} ({result['elapsed_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
