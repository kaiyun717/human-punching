"""Explicit Task 10 audit/conversion/configuration entry points.

No command submits Slurm jobs. Training is delegated to the pinned WBC/mjlab
runner only after separate native/reference qualification and compute checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .budget import TrainingScale
from .provenance import PROJECT_ROOT, configure_local_caches, runtime_record, source_record, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Record source and installed-package identity; no simulation")
    audit.add_argument("--output", type=Path, required=True)
    budget = commands.add_parser("budget", help="Print global-versus-per-rank PPO accounting")
    budget.add_argument("--gpus", type=int, choices=(1, 2, 4), default=2)
    budget.add_argument("--environments", type=int, default=8192)
    budget.add_argument("--iterations", type=int, default=10000)
    convert = commands.add_parser("convert-release", help="Convert preserved references using FK only")
    convert.add_argument("--release", type=Path,
                         default=PROJECT_ROOT / "outputs/four-punch-milestone")
    convert.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect-task", help="Build WBC task config; no physics/learning")
    inspect.add_argument("--bundle", type=Path, required=True)
    inspect.add_argument("--environments", type=int, default=1, help="Per rank")
    inspect.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-trace", help="Evaluate complete recorded 200 Hz evidence")
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--trace", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--recipe", required=True)
    evaluate.add_argument("--spatial-cell", required=True)
    evaluate.add_argument("--ttc-cell", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Run qualified WBC learning inside an existing allocation")
    train.add_argument("--bundle", type=Path, required=True)
    train.add_argument("--compute-plan", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    probe = commands.add_parser("native-probe", help="Bounded diagnostic inside an existing Slurm GPU allocation")
    probe.add_argument("--bundle", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--environments", type=int, default=16)
    probe.add_argument("--policy-steps", type=int, default=50)
    candidates = commands.add_parser("generate-candidates", help="Offline FK/IK candidates; never qualifies or trains")
    candidates.add_argument("--recipes", type=Path, required=True, help="JSON array of CandidateSpec objects")
    candidates.add_argument("--release", type=Path, required=True)
    candidates.add_argument("--snapshot-bank", type=Path, required=True)
    candidates.add_argument("--g1-assets", type=Path, required=True)
    candidates.add_argument("--nominal-plant", type=Path, required=True)
    candidates.add_argument("--backend", choices=("original", "refined_v1", "refined_v2", "refined_v3"), default="refined_v3")
    candidates.add_argument("--ik-iterations", type=int, default=35)
    candidates.add_argument("--guard-iterations", type=int, default=120)
    candidates.add_argument("--output", type=Path, required=True)
    effort = commands.add_parser("diagnose-effort", help="Offline exact-path support/effort diagnostic; no physics integration")
    effort.add_argument("--bundle", type=Path, required=True)
    effort.add_argument("--phases", type=Path)
    effort.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command != "budget":
        configure_local_caches()
    if args.command == "budget":
        report = TrainingScale(global_environments=args.environments,
                               world_size=args.gpus).report(args.iterations)
    elif args.command == "audit":
        report = {"schema": "human_wbc_environment_audit_v1", "sources": source_record(),
                  "runtime": runtime_record(), "simulation_executed": False,
                  "training_executed": False}
        write_json(args.output, report)
    elif args.command == "convert-release":
        from .references import convert_release
        report = convert_release(args.release, args.output)
    elif args.command == "inspect-task":
        from .runtime.task import build_human_env_cfg, human_ppo_cfg
        cfg = build_human_env_cfg(args.bundle, args.environments, evaluation=True)
        agent = human_ppo_cfg()
        report = {"schema": "human_wbc_config_inspection_v1",
                  "runtime": runtime_record(), "simulation_executed": False,
                  "environments_per_rank": cfg.scene.num_envs,
                  "physics_dt_s": cfg.sim.mujoco.timestep, "decimation": cfg.decimation,
                  "solver": asdict(cfg.sim.mujoco), "ppo": asdict(agent),
                  "commands": {k: type(v).__name__ for k, v in cfg.commands.items()},
                  "actions": {k: type(v).__name__ for k, v in cfg.actions.items()},
                  "observations": {k: list(v.terms) for k, v in cfg.observations.items()}}
        write_json(args.output, report)
    elif args.command == "evaluate-trace":
        import numpy as np
        from .contracts import LaunchSnapshot, PunchRequest
        from .evaluation import evaluate_episode
        specification = json.loads(args.request.read_text())
        specification["snapshot"] = LaunchSnapshot(**specification["snapshot"])
        request = PunchRequest(**specification)
        with np.load(args.trace, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        result = evaluate_episode(request, arrays, training_seed=args.seed,
                                  recipe_id=args.recipe, spatial_cell_id=args.spatial_cell,
                                  ttc_cell_id=args.ttc_cell)
        report = asdict(result)
        write_json(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if result.complete_cycle_success and result.acceptance_eligible else 1
    elif args.command == "train":
        from .training import launch_qualified_training
        launch_qualified_training(args.bundle, args.compute_plan, args.output, args.seed)
        return 0
    elif args.command == "native-probe":
        from .native_probe import run_native_probe
        report = run_native_probe(args.bundle, args.output,
                                  envs=args.environments, policy_steps=args.policy_steps)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report.get("status") == "passed" else 1
    elif args.command == "generate-candidates":
        from .candidates import CandidateSpec, build_candidate_bundle
        rows = json.loads(args.recipes.read_text())
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Recipes must be a nonempty JSON array of explicit candidate objects")
        report = build_candidate_bundle([CandidateSpec(**row) for row in rows],
            release_root=args.release, bank_root=args.snapshot_bank,
            g1_asset_root=args.g1_assets, nominal_plant_path=args.nominal_plant,
            output_root=args.output, backend=args.backend,
            ik_iterations=args.ik_iterations, guard_iterations=args.guard_iterations)
    elif args.command == "diagnose-effort":
        from .effort_diagnostics import diagnose_bundle
        report = diagnose_bundle(args.bundle, args.output, phase_sidecar=args.phases)
    else:
        raise AssertionError("unreachable command")
    # The saved manifest remains authoritative; do not print large array payloads.
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
