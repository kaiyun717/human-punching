"""Phase 2 jab retention/evaluation hooks around mjlab's existing PPO runner.

Collection, return calculation, normalization and PPO updates remain upstream.
Only the local logger subclass adds an after-update hook and bounded stopping.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import difflib
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.utils.logger import Logger

from .rl_cfg import runner_cfg
from .train import ROOT, identity, json_write, state_equal

PHASE = "task10c_phase2_jab"
JAB_SHA256 = "f4c710d2d6839d8b86486d141f68d307a268a999aed1ecdbe36b49713878f8f6"


def resolve_reward_scales(parent: dict | None = None, **overrides) -> dict[str, float]:
    """Retain a resumed lineage's budgets unless an explicit override is given."""
    scales = {"balance": 1.0, "guard": 1.0, "style": 1.0}
    if parent is not None:
        scales.update(parent)
    scales.update({key: value for key, value in overrides.items() if value is not None})
    if set(scales) != {"balance", "guard", "style"}:
        raise ValueError("Only balance, guard and style reward budgets are configurable")
    scales = {key: float(value) for key, value in scales.items()}
    if any(not math.isfinite(value) or value < 0 for value in scales.values()):
        raise ValueError("Reward budget scales must be finite and nonnegative")
    return scales


def resolve_guard_speed_profile(parent: str | None = None, override: str | None = None) -> str:
    profile = override if override is not None else parent if parent is not None else "gaussian"
    if profile not in ("gaussian", "rational"):
        raise ValueError("Guard speed profile must be gaussian or rational")
    return profile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_resume(infos: dict, reference_hash: str, target_hash: str, seed: int) -> None:
    """Reject setup/other-family weights before mutating any learner state."""
    if infos.get("phase") != PHASE or infos.get("family") != "jab" or not infos.get("lineage_id"):
        raise ValueError("Resume requires this Phase 2 fresh-jab lineage, never Phase 1/another family")
    for key, expected in (("reference_sha256", reference_hash), ("target_pool_sha256", target_hash),
                          ("seed", seed)):
        if infos.get(key) != expected:
            raise ValueError(f"Resume identity mismatch: {key}")
    if not isinstance(infos.get("completed_iterations"), int) or infos["completed_iterations"] < 0:
        raise ValueError("Resume is missing its completed iteration count")
    if not isinstance(infos.get("curriculum"), dict):
        raise ValueError("Resume is missing its curriculum state")


class BoundedStop(Exception):
    """A normal boundary stop after a complete PPO update."""


class UpdateLogger(Logger):
    def __init__(self, *args, after_update, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_update = after_update

    def log(self, **kwargs):
        super().log(**kwargs, print_minimal=True)
        self.after_update(kwargs)


class Phase2Runner(MjlabOnPolicyRunner):
    """Atomic rolling checkpoint adapter; no custom PPO implementation."""
    def __init__(self, *args, metadata: dict, curriculum: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_metadata = metadata
        self.curriculum = deepcopy(curriculum)
        self.completed_iterations = 0
        self.total_transitions = 0
        self.best_score: float | None = None
        self.best_iteration: int | None = None
        self.best_checkpoint: str | None = None
        self.output = Path(self.logger.log_dir)

    def save(self, path: str, infos=None) -> None:
        # Upstream's periodic/final model names both map to rolling latest.
        name = "best.pt" if Path(path).name == "best.pt" else "latest.pt"
        destination = self.output / name
        details = {**self.task_metadata, **(infos or {}),
                   "completed_iterations": self.completed_iterations,
                   "total_transitions": self.total_transitions,
                   "curriculum": deepcopy(self.curriculum),
                   "best_score": self.best_score, "best_iteration": self.best_iteration,
                   "best_checkpoint": self.best_checkpoint,
                   "adaptive_learning_rate": self.alg.learning_rate}
        temporary = destination.with_suffix(".tmp")
        super().save(str(temporary), details)
        temporary.replace(destination)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        expected = self.task_metadata
        validate_resume(loaded.get("infos", {}), expected["reference_sha256"],
                        expected["target_pool_sha256"], expected["seed"])
        old_contract = loaded["infos"].get("observation_contract")
        if old_contract != expected.get("observation_contract"):
            raise ValueError("Resume observation/action contract changed")
        infos = super().load(path, load_cfg, strict, map_location)
        self.completed_iterations = infos["completed_iterations"]
        self.current_learning_iteration = self.completed_iterations
        self.total_transitions = infos["total_transitions"]
        self.curriculum = deepcopy(infos["curriculum"])
        self.best_score = infos.get("best_score")
        self.best_iteration = infos.get("best_iteration")
        self.best_checkpoint = infos.get("best_checkpoint")
        self.task_metadata["lineage_id"] = infos["lineage_id"]
        # Keep the adaptive scalar explicit in the checkpoint contract alongside
        # the optimizer parameter groups restored by RSL-RL.
        if "adaptive_learning_rate" in infos:
            self.alg.learning_rate = float(infos["adaptive_learning_rate"])
        self.env.unwrapped.command_manager.get_term("punch").set_curriculum(self.curriculum)
        return infos


class EpisodeStream:
    """Drain one rollout at a time; retain all raw records on disk, not in RAM."""
    def __init__(self, output: Path):
        self.file = gzip.open(output / "episodes.jsonl.gz", "at", compresslevel=3)
        self.totals: dict[str, dict] = {}
        self.count = 0

    def drain(self, command, iteration: int) -> dict:
        records = command.episode_records
        batch = {"episodes": len(records), "full_hits": 0, "full_attempts": 0,
                 "falls": 0, "component_sums": defaultdict(float)}
        for record in records:
            record["completed_iteration"] = iteration
            record["episode_record_id"] = self.count
            self.file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            self.count += 1
            kind = record["reset_kind"]
            total = self.totals.setdefault(kind, {"episodes": 0, "hits": 0, "falls": 0,
                "completed_horizon": 0, "interrupted": 0, "component_sums": {}})
            total["episodes"] += 1
            total["hits"] += int(record.get("hit") is True)
            total["falls"] += int(record["fall"])
            total["completed_horizon"] += int(record["complete_horizon"])
            total["interrupted"] += int(not record["complete_horizon"] and not record["fall"])
            batch["falls"] += int(record["fall"])
            if kind == "full":
                batch["full_attempts"] += 1
                batch["full_hits"] += int(record.get("hit") is True)
            for key, value in record["raw_component_totals"].items():
                total["component_sums"][key] = total["component_sums"].get(key, 0.0) + value
                batch["component_sums"][key] += value
        records.clear()
        self.file.flush()
        return batch

    def close(self):
        self.file.close()


def initialize_guard_bias(runner: Phase2Runner, command) -> dict:
    env = runner.env.unwrapped
    action = env.action_manager.get_term("joint_pos")
    guard = command.reference.arrays["joint_pos"][0]
    bias = (guard[action.target_ids] - action.offset[0]) / action.scale[0]
    layer = [module for module in runner.alg.actor.mlp.modules()
             if isinstance(module, torch.nn.Linear)][-1]
    with torch.no_grad():
        layer.bias.copy_(bias)
    return {"guard_bias": bias.cpu().tolist(), "fresh_weights_normalizers_optimizer": True}


def checkpoint_roundtrip(runner: Phase2Runner) -> dict:
    """Native saved-state reload, including deterministic actions and optimizer."""
    runner.save(str(runner.output / "latest.pt"))
    observations = runner.env.get_observations().clone()
    runner.alg.eval_mode()
    with torch.inference_mode():
        expected = runner.alg.actor(observations).clone()
    saved = torch.load(runner.output / "latest.pt", weights_only=False, map_location=runner.device)
    expected_lr = runner.alg.learning_rate
    restored = runner.load(str(runner.output / "latest.pt"), map_location=runner.device)
    runner.alg.eval_mode()
    with torch.inference_mode():
        actual = runner.alg.actor(observations)
    result = {
        "deterministic_action_exact": bool(torch.equal(expected, actual)),
        "actor_normalizer_exact": state_equal(saved["actor_state_dict"], runner.alg.actor.state_dict()),
        "critic_normalizer_exact": state_equal(saved["critic_state_dict"], runner.alg.critic.state_dict()),
        "optimizer_exact": state_equal(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict()),
        "adaptive_learning_rate_exact": expected_lr == runner.alg.learning_rate,
        "completed_iterations": restored["completed_iterations"],
        "parameters_finite": all(bool(torch.isfinite(p).all())
                                 for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()),
    }
    json_write(runner.output / "checkpoint_check.json", result)
    if not all(value for key, value in result.items() if key != "completed_iterations"):
        raise AssertionError(result)
    runner.alg.train_mode()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("jab",), default="jab")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=2000, help="Additional PPO updates this invocation")
    parser.add_argument("--seed", type=int, default=102202)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pool", type=Path, default=ROOT / "outputs/task10c/setup/target_pool.npz")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--guard-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wall-seconds", type=float, default=20400,
                        help="Application budget including initialization; reserve time before Slurm ends")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-attempts", type=int, default=64)
    parser.add_argument("--eval-videos", type=int, default=0)
    parser.add_argument("--target-fraction", type=float)
    parser.add_argument("--ttc-min", type=float)
    parser.add_argument("--ttc-max", type=float)
    parser.add_argument("--curriculum-stage", type=int)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--init-std", type=float)
    parser.add_argument("--balance-scale", type=float, help="Balance episode-budget multiplier; fresh default 1")
    parser.add_argument("--guard-scale", type=float, help="Recovery episode-budget multiplier; fresh default 1")
    parser.add_argument("--style-scale", type=float, help="Style episode-budget multiplier; fresh default 1")
    parser.add_argument("--guard-speed-profile", choices=("gaussian", "rational"),
                        help="Soft speed reward profile; fresh default gaussian, resume inherits")
    args = parser.parse_args()
    started = time.monotonic()
    if args.updates <= 0 or args.num_envs <= 0 or args.wall_seconds <= 0 or args.wall_seconds > 21600:
        parser.error("Positive exposure/budget required, with wall seconds <=21600")
    if args.eval_every <= 0 or args.checkpoint_every <= 0 or args.eval_attempts <= 0:
        parser.error("Checkpoint/evaluation cadences and attempt count must be positive")
    if (args.ttc_min is None) != (args.ttc_max is None):
        parser.error("Specify both --ttc-min and --ttc-max")
    if args.ttc_min is not None and not 0.5 <= args.ttc_min < args.ttc_max:
        parser.error("Requested TTC must have 0.5 <= min < max")
    if args.target_fraction is not None and not 0 < args.target_fraction <= 1:
        parser.error("Target fraction must lie in (0,1]")
    try:
        resolve_reward_scales(balance=args.balance_scale, guard=args.guard_scale, style=args.style_scale)
    except ValueError as error:
        parser.error(str(error))
    # Slurm may backfill a shorter allocation when submitted with --time-min.
    # Honor its actual end time and retain shutdown/evaluation reserve.
    if os.environ.get('SLURM_JOB_END_TIME'):
        allocation_remaining = int(os.environ['SLURM_JOB_END_TIME']) - time.time()
        args.wall_seconds = min(args.wall_seconds, allocation_remaining - 180)
        if args.wall_seconds <= 0:
            raise RuntimeError('Allocation has no usable training time after shutdown reserve')
    out = args.output.resolve()
    out.relative_to((ROOT / "outputs/task10c/jab").resolve())
    if (out / "identity.json").exists():
        raise FileExistsError("Use a new invocation directory; never overwrite historical run identity")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Substantive jab PPO requires an allocated CUDA GPU")
    import warp as wp
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    from .env_cfg import make_env_cfg
    from .environment import PunchEnv
    from .mdp import OBSERVATION_SCHEMA
    configure_torch_backends()
    torch.manual_seed(args.seed)
    cfg = make_env_cfg(family="jab", target_pool_path=args.target_pool, num_envs=args.num_envs,
                       seed=args.seed, recovery_fraction=0.2)
    cfg.sim.nan_guard.output_dir = str(out / "numerical_diagnostics")
    reference_path = Path(cfg.commands["punch"].reference_path)
    if sha256(reference_path) != JAB_SHA256:
        raise ValueError("Selected jab reference differs from the Phase 2 accepted input")
    metadata = {**identity(), "phase": PHASE, "family": "jab", "seed": args.seed,
                "lineage_id": str(uuid.uuid4()), "reference_sha256": sha256(reference_path),
                "target_pool_sha256": sha256(args.target_pool), "argv": os.sys.argv,
                "physics_dt": 0.005, "actor_dt": 0.02,
                "initialization": "fresh_weights_normalizers_optimizer",
                "resume_semantics": "learner/normalizer/optimizer/iteration/curriculum; fresh simulator episodes, not bitwise RNG resume"}
    resume_infos = None
    if args.resume:
        loaded = torch.load(args.resume, weights_only=False, map_location="cpu")
        resume_infos = loaded.get("infos", {})
        validate_resume(resume_infos, metadata["reference_sha256"], metadata["target_pool_sha256"], args.seed)
        metadata.update(lineage_id=resume_infos["lineage_id"], initialization="phase2_lineage_resume",
                        resume_checkpoint=str(args.resume.resolve()), resume_checkpoint_sha256=sha256(args.resume))
        del loaded
    reward_scales = resolve_reward_scales(resume_infos.get("reward_scales") if resume_infos else None,
                                         balance=args.balance_scale, guard=args.guard_scale, style=args.style_scale)
    cfg.commands["punch"].reward_scales = reward_scales.copy()
    metadata["reward_scales"] = reward_scales.copy()
    guard_speed_profile = resolve_guard_speed_profile(
        resume_infos.get("guard_speed_profile") if resume_infos else None, args.guard_speed_profile)
    cfg.commands["punch"].guard_speed_profile = guard_speed_profile
    metadata["guard_speed_profile"] = guard_speed_profile
    metadata["reward_component_totals_semantics"] = "scaled reward rates integrated once over actor dt; RewardManager weights remain 1"
    if resume_infos:
        metadata["parent_reward_scales"] = resolve_reward_scales(resume_infos.get("reward_scales"))
        metadata["parent_guard_speed_profile"] = resolve_guard_speed_profile(resume_infos.get("guard_speed_profile"))
    ref_metadata = json.loads(reference_path.with_suffix(".json").read_text())
    nominal_ttc = max(0.5, ref_metadata["strike_time_s"])
    curriculum = deepcopy(resume_infos["curriculum"]) if resume_infos else {
        "stage": 0, "target_fraction": 0.25, "ttc_range": [max(0.5, 0.9 * nominal_ttc), 1.1 * nominal_ttc],
        "easy_fraction": 0.5}
    for key, value in (("target_fraction", args.target_fraction), ("stage", args.curriculum_stage)):
        if value is not None:
            curriculum[key] = value
    if args.ttc_min is not None:
        curriculum["ttc_range"] = [args.ttc_min, args.ttc_max]
    cfg.commands["punch"].target_fraction = curriculum["target_fraction"]
    cfg.commands["punch"].ttc_range = tuple(curriculum["ttc_range"])
    # New invocations inherit established PPO hyperparameters. Older Phase 2
    # checkpoints predate this metadata field and used the original runner_cfg.
    agent_cfg = (deepcopy(resume_infos["agent_config"])
                 if resume_infos and "agent_config" in resume_infos else asdict(runner_cfg(args.seed)))
    agent_cfg.update(experiment_name=PHASE, max_iterations=args.updates, save_interval=args.checkpoint_every)
    if args.entropy_coef is not None:
        agent_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
    if args.init_std is not None:
        if args.resume:
            parser.error("--init-std only applies to fresh weights")
        agent_cfg["actor"]["distribution_cfg"]["init_std"] = args.init_std
    metadata["agent_config"] = deepcopy(agent_cfg)
    if resume_infos:
        metadata["parent_agent_config"] = deepcopy(resume_infos.get("agent_config"))
    tracked_paths = ["src/human_punching/task10c", "configs/task10c", "scripts"]
    source_diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD", "--", *tracked_paths], text=True)
    untracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--others",
                                         "--exclude-standard", "--", *tracked_paths], text=True).splitlines()
    for relative in untracked:
        source_path = (ROOT / relative).resolve()
        source_path.relative_to(ROOT)
        source_diff += "".join(difflib.unified_diff([], source_path.read_text().splitlines(keepends=True),
                                                    fromfile="/dev/null", tofile=f"b/{relative}"))
    (out / "source.diff").write_text(source_diff)
    metadata["source_diff_sha256"] = sha256(out / "source.diff")
    json_write(out / "config.json", {"arguments": vars(args), "agent": agent_cfg,
                                    "environment": asdict(cfg), "curriculum": curriculum,
                                    "reward_scales": reward_scales, "guard_speed_profile": guard_speed_profile})
    env = PunchEnv(cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)
    command = env.command_manager.get_term("punch")
    command.set_curriculum(curriculum)
    action = env.action_manager.get_term("joint_pos")
    metadata["observation_contract"] = {"order": OBSERVATION_SCHEMA,
        "action_joint_names": [env.scene["robot"].joint_names[i] for i in action.target_ids],
        "action_scale": action.scale[0].cpu().tolist(), "action_offset": action.offset[0].cpu().tolist()}
    runner = Phase2Runner(wrapped, deepcopy(agent_cfg), str(out), "cuda:0",
                          metadata=metadata, curriculum=curriculum)
    if args.resume:
        runner.load(str(args.resume), map_location="cuda:0")
        runner.curriculum = deepcopy(curriculum)
        command.set_curriculum(curriculum)
        metadata["curriculum_parent"] = resume_infos["curriculum"]
        if curriculum != resume_infos["curriculum"]:
            metadata["parent_best"] = {key: resume_infos.get(key)
                                       for key in ("best_score", "best_iteration", "best_checkpoint")}
            runner.best_score = runner.best_iteration = runner.best_checkpoint = None
    elif args.guard_bias:
        json_write(out / "initial_action.json", initialize_guard_bias(runner, command))
    json_write(out / "identity.json", metadata)
    # Commands/curriculum are installed before the first collection reset.
    wrapped.reset()
    command.episode_records.clear()
    stream = EpisodeStream(out)
    curve = (out / "learning.jsonl").open("a", buffering=1)
    initial_iteration, initial_transitions = runner.completed_iterations, runner.total_transitions
    initial_parameters = {key: value.detach().cpu().clone() for key, value in runner.alg.actor.named_parameters()}
    stop_reason = "updates_completed"
    stop_signal = None
    eval_env = None
    latest_eval = None
    last_eval_iteration = None
    previous_signals = {}

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signal.Signals(signum).name

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        previous_signals[signum] = signal.signal(signum, request_stop)

    def evaluate_current():
        nonlocal eval_env, latest_eval, last_eval_iteration
        from .evaluate import create_recipes, evaluate_policy, make_evaluation_env
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            if eval_env is None:
                eval_env = make_evaluation_env(cfg, split="validation", num_envs=args.eval_attempts,
                                               output=out / "development")
            recipe_path = out / "development_recipes.json"
            if recipe_path.exists():
                recipes = json.loads(recipe_path.read_text())
            else:
                recipes = create_recipes(args.target_pool, "validation", args.eval_attempts,
                                         args.seed + 10000, tuple(curriculum["ttc_range"]),
                                         target_fraction=curriculum["target_fraction"])
                json_write(recipe_path, recipes)
            destination = out / "development" / f"update_{runner.completed_iterations:06d}"
            latest_eval = evaluate_policy(eval_env, runner.get_inference_policy(), recipes, destination,
                                          video_count=args.eval_videos)
        runner.alg.train_mode()
        score = float(latest_eval["selection_score"])
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite development selection score")
        if runner.best_score is None or score > runner.best_score:
            runner.best_score, runner.best_iteration = score, runner.completed_iterations
            runner.best_checkpoint = str(out / "best.pt")
            runner.save(str(out / "best.pt"), {"selection_evaluation": str(destination)})
        runner.save(str(out / "latest.pt"))
        last_eval_iteration = runner.completed_iterations

    def after_update(values):
        nonlocal stop_reason
        runner.completed_iterations = values["it"] + 1
        runner.total_transitions += args.num_envs * agent_cfg["num_steps_per_env"]
        episodes = stream.drain(command, runner.completed_iterations)
        row = {"completed_iterations": runner.completed_iterations,
               "total_transitions": runner.total_transitions, "elapsed_seconds": time.monotonic() - started,
               "collect_seconds": values["collect_time"], "learn_seconds": values["learn_time"],
               "learning_rate": runner.alg.learning_rate,
               "mean_action_std": float(values["action_std"].detach().mean()),
               "losses": {key: float(value) for key, value in values["loss_dict"].items()},
               "episodes": episodes, "curriculum": runner.curriculum}
        curve.write(json.dumps(row, allow_nan=False) + "\n")
        if runner.completed_iterations % args.checkpoint_every == 0:
            runner.save(str(out / "latest.pt"))
            json_write(out / "component_totals.json", stream.totals)
        # Wall/signal stops occur at completed updates so optimizer/checkpoint
        # counts agree; never attempt a long evaluation inside the final reserve.
        remaining = args.wall_seconds - (time.monotonic() - started)
        if stop_signal or remaining <= 60:
            stop_reason = stop_signal or "wall_budget"
            raise BoundedStop(stop_reason)
        if runner.completed_iterations % args.eval_every == 0 and remaining > 180:
            evaluate_current()

    runner.logger = UpdateLogger(log_dir=str(out), cfg=runner.cfg, env_cfg=wrapped.cfg,
        num_envs=args.num_envs, is_distributed=False, gpu_world_size=1, gpu_global_rank=0,
        device="cuda:0", after_update=after_update)
    # Our identity/source diff already covers the local source. Avoid duplicate
    # source bundles emitted by the upstream logger's default repository list.
    runner.logger.git_status_repos = []
    runner.save(str(out / "latest.pt"))
    json_write(out / "initial_learner.json", {
        "lineage_id": runner.task_metadata["lineage_id"], "initialization": metadata["initialization"],
        "completed_iterations": runner.completed_iterations,
        "optimizer_state_entries": len(runner.alg.optimizer.state),
        "initial_checkpoint_sha256": sha256(out / "latest.pt"), "guard_bias_requested": args.guard_bias,
    })
    try:
        evaluate_current()
        try:
            runner.learn(args.updates, init_at_random_ep_len=False)
        except BoundedStop:
            pass
        runner.save(str(out / "latest.pt"))
        if not stop_signal and args.wall_seconds - (time.monotonic() - started) > 180 and last_eval_iteration != runner.completed_iterations:
            evaluate_current()
        parity = checkpoint_roundtrip(runner)
        # This final reset records unfinished collection episodes as interrupted,
        # explicitly separate from complete full-cycle or fall outcomes.
        wrapped.reset()
        stream.drain(command, runner.completed_iterations)
        json_write(out / "component_totals.json", stream.totals)
        summary = {"phase": PHASE, "status": "completed" if stop_reason == "updates_completed" else "bounded_stop",
            "stop_reason": stop_reason, "seed": args.seed, "lineage_id": runner.task_metadata["lineage_id"],
            "starting_completed_iterations": initial_iteration, "completed_iterations": runner.completed_iterations,
            "new_updates": runner.completed_iterations - initial_iteration,
            "new_transitions": runner.total_transitions - initial_transitions, "total_transitions": runner.total_transitions,
            "num_envs": args.num_envs, "rollout_steps": agent_cfg["num_steps_per_env"],
            "elapsed_seconds": time.monotonic() - started, "curriculum": runner.curriculum,
            "reward_scales": reward_scales,
            "guard_speed_profile": guard_speed_profile,
            "reset_counts": command.reset_counts, "transition_counts": command.transition_counts,
            "episode_records": stream.count, "component_totals": stream.totals,
            "latest_checkpoint": str(out / "latest.pt"), "best_checkpoint": runner.best_checkpoint,
            "best_iteration": runner.best_iteration, "best_score": runner.best_score,
            "latest_development_evaluation": latest_eval, "checkpoint_check": parity,
            "actor_changed_after_updates": any(not torch.equal(value, dict(runner.alg.actor.named_parameters())[key].detach().cpu())
                                               for key, value in initial_parameters.items()),
            "heldout_evaluation": "not run by training; separate frozen-policy evaluation required"}
        json_write(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        # Preserve the most recently completed update. A numerical fault is not
        # relabeled a healthy bounded stop, and the process retains failure status.
        stream.drain(command, runner.completed_iterations)
        json_write(out / "failure.json", {"type": type(error).__name__, "error": str(error),
            "completed_iterations": runner.completed_iterations, "total_transitions": runner.total_transitions})
        raise
    finally:
        curve.close()
        stream.close()
        if runner.logger.writer is not None and hasattr(runner.logger.writer, "close"):
            runner.logger.writer.close()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        if eval_env is not None:
            eval_env.close()
        env.close()


if __name__ == "__main__":
    main()
