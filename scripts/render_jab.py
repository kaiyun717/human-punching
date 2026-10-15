#!/usr/bin/env python3
"""Render a scalar human reference or a recorded learned/classical controller trial."""

import argparse
import json
from pathlib import Path

import numpy as np

from human_punching.jab_evaluation import evaluate_rollout, render_rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/scalar-body/human.xml"))
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--scene-model", type=Path, help="Optional fixed-G1 scene with identical human state order, used for rendering only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--head-center", type=float, nargs=3)
    parser.add_argument("--physics-timestep", type=float, help="Explicit simulator timestep override when not saved in rollout")
    parser.add_argument("--attack-window", type=float, nargs=2)
    parser.add_argument("--family", choices=("jab", "cross", "left_hook", "right_hook"), help="Display label; otherwise infer reference metadata")
    parser.add_argument("--artifact-prefix", choices=("jab", "cross", "left_hook", "right_hook", "punch"), default="jab", help="Filename prefix; legacy default preserves previous artifact paths")
    parser.add_argument("--require-physics-timestep", type=float)
    parser.add_argument("--require-policy-timestep", type=float)
    parser.add_argument("--kind", choices=("reference", "learned", "classical"), required=True)
    args = parser.parse_args()
    family = args.family
    if family is None:
        for source in (args.reference, args.rollout):
            if source is not None:
                with np.load(source, allow_pickle=False) as arrays:
                    key = next((key for key in ("punch_family", "family") if key in arrays), None)
                    if key is not None:
                        family = str(arrays[key].item())
                        break
    family = (family or "jab").replace("_", " ").upper()
    labels = {
        "reference": f"KINEMATIC {family} REFERENCE",
        "learned": f"LEARNED {family} POLICY — RECORDED PHYSICS",
        "classical": f"REFERENCE {family} PD BASELINE — RECORDED PHYSICS",
    }
    head = args.head_center
    if head is None:
        for source in (args.rollout, args.reference):
            if source is None:
                continue
            with np.load(source, allow_pickle=False) as arrays:
                for key in ("head_center_w", "target_head_center_w", "head_center"):
                    if key in arrays:
                        head = arrays[key].copy()
                        break
            if head is not None:
                break
    if args.kind != "reference":
        if args.reference is None:
            parser.error("--reference is required when evaluating a physics rollout")
        evaluate_rollout(args.model, args.rollout, args.reference,
                         args.output / "evaluation.json", head_center=head,
                         physics_timestep_s=args.physics_timestep,
                         attack_window_reference_s=args.attack_window,
                         required_physics_timestep_s=args.require_physics_timestep,
                         required_policy_timestep_s=args.require_policy_timestep)
    result = render_rollout(args.scene_model or args.model, args.rollout, args.output,
                            head_center=head, label=labels[args.kind], artifact_prefix=args.artifact_prefix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
