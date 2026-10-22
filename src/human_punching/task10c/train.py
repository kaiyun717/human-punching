"""Small Phase 1 collection/update exercise using the unmodified PPO learner.

Train/play construction follows TaskNPoint scripts/train.py (b8faaf2); only
local task configuration and checkpoint metadata/retention are adapted here.
No substantive specialist learning is implied by this setup entry point.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from human_punching.task10c.rl_cfg import runner_cfg

ROOT = Path(__file__).resolve().parents[3]


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def identity() -> dict:
    sources = {}
    for name in ("human-punching", "tasknpoint", "mjlab", "GMR", "wbc-mjlab-extension-h2"):
        sources[name] = subprocess.check_output(
            ["git", "-C", str(ROOT.parent / name), "rev-parse", "HEAD"], text=True).strip()
    files = list((ROOT / "src/human_punching/task10c").rglob("*.py"))
    files += [p for p in (ROOT / "configs/task10c").rglob("*") if p.is_file()]
    files += list((ROOT / "scripts").glob("task10c*"))
    files += list((ROOT / "assets/task10c").rglob("*.xml"))
    files += list((ROOT / "outputs/task10c/references").glob("*.npz"))
    files += list((ROOT / "outputs/task10c/references").glob("*.json"))
    return {
        "source_revisions": sources,
        "file_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(files)},
        "python": platform.python_version(), "executable": str(Path(os.sys.executable).resolve()),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("mjlab", "rsl-rl-lib", "mujoco", "mujoco-warp", "warp-lang", "torch")},
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
    }


class SetupRunner(MjlabOnPolicyRunner):
    """Metadata/rolling-checkpoint adapter; collection and PPO remain upstream."""

    task_metadata: dict

    def save(self, path: str, infos=None) -> None:
        # RSL-RL calls save with the index of the update just completed. Record
        # the next index explicitly so a resumed learn() does not repeat it.
        details = {**getattr(self, "task_metadata", {}), **(infos or {}),
                   "completed_iterations": self.current_learning_iteration + 1,
                   "curriculum": {"stage": 0, "scope": "phase1_setup"}}
        destination = Path(self.logger.log_dir) / "latest.pt"
        temporary = destination.with_suffix(".tmp")
        super().save(str(temporary), details)
        temporary.replace(destination)
        # Retain collected evidence even if a later reset/replay fails.
        env = self.env.unwrapped
        command = env.command_manager.get_term('punch')
        json_write(destination.parent / 'episodes.json', command.episode_records)
        json_write(destination.parent / 'collection_exposure.json', {
            'completed_iterations': details['completed_iterations'],
            'reset_counts': dict(command.reset_counts),
            'transition_counts': dict(command.transition_counts)})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        self.current_learning_iteration = infos["completed_iterations"]
        return infos


def state_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("jab", "cross", "left_hook", "right_hook"), default="jab")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--resume-updates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=102100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pool", type=Path, default=ROOT / "outputs/task10c/setup/target_pool.npz")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--guard-bias", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT / "outputs/task10c")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Phase 1 native exercise requires an allocated CUDA GPU")
    import warp as wp
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    from human_punching.task10c.env_cfg import make_env_cfg
    from human_punching.task10c.environment import PunchEnv
    configure_torch_backends()
    torch.manual_seed(args.seed)
    cfg = make_env_cfg(family=args.family,
                       target_pool_path=args.target_pool, num_envs=args.num_envs, seed=args.seed)
    reference_path = Path(cfg.commands['punch'].reference_path)
    agent_cfg = asdict(runner_cfg(args.seed))
    metadata = {**identity(), "family": args.family, "seed": args.seed,
                "argv": os.sys.argv, "actor_dt": 0.02, "physics_dt": 0.005,
                "resume_semantics": "learner/normalizer/optimizer/iteration; fresh simulator episodes"}
    metadata["target_pool_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.target_pool, args.target_pool.with_suffix('.json'))
    }
    metadata["selected_reference_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (reference_path, reference_path.with_suffix('.json'))
    }
    json_write(out / "identity.json", metadata)
    json_write(out / "config.json", {"arguments": vars(args), "agent": agent_cfg, "environment": asdict(cfg)})
    env = PunchEnv(cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)
    runner = SetupRunner(wrapped, deepcopy(agent_cfg), str(out), "cuda:0")
    cmd = env.command_manager.get_term("punch")
    from human_punching.task10c.mdp import OBSERVATION_SCHEMA
    term = env.action_manager.get_term('joint_pos')
    metadata["observation_contract"] = {"order": OBSERVATION_SCHEMA,
        "action_joint_names": [env.scene["robot"].joint_names[i] for i in term.target_ids],
        "action_scale": term.scale[0].cpu().tolist(), "action_offset": term.offset[0].cpu().tolist()}
    json_write(out / "identity.json", metadata)
    runner.task_metadata = metadata
    initial_action_info = {}
    if args.resume:
        runner.load(str(args.resume))
        metadata['resume_checkpoint'] = str(args.resume.resolve())
        metadata['resume_checkpoint_sha256'] = hashlib.sha256(args.resume.read_bytes()).hexdigest()
    elif args.guard_bias:
        from human_punching.task10c.references import StoredReference
        ref = StoredReference.load(reference_path)
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        indices = [list(ref.arrays["joint_names"]).index(n) for n in names]
        guard = torch.as_tensor(ref.arrays["joint_pos"][0, indices], device=env.device)
        action = env.action_manager.get_term("joint_pos")
        bias = (guard[action.target_ids] - action.offset[0]) / action.scale[0]
        layer = [m for m in runner.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            layer.bias.copy_(bias)
        initial_action_info = {"guard_bias": bias.cpu().tolist(),
            "default_guard_max_delta_rad": float((guard - robot.data.default_joint_pos[0]).abs().max())}
    json_write(out / "initial_action.json", initial_action_info)
    starting_iteration = runner.current_learning_iteration
    json_write(out / "identity.json", metadata)
    before = {k: v.clone() for k, v in runner.alg.actor.named_parameters()}
    started = time.monotonic()
    runner.learn(args.updates, init_at_random_ep_len=False)
    fixed_obs = wrapped.get_observations().clone()
    runner.alg.eval_mode()
    with torch.inference_mode():
        expected = runner.alg.actor(fixed_obs).clone()
    saved = torch.load(out / "latest.pt", weights_only=False, map_location="cuda:0")
    clone = SetupRunner(wrapped, deepcopy(agent_cfg), str(out / "resume"), "cuda:0")
    clone.task_metadata = metadata
    restored = clone.load(str(out / "latest.pt"))
    clone.alg.eval_mode()
    with torch.inference_mode():
        actual = clone.alg.actor(fixed_obs)
    parity = {
        "deterministic_action_exact": bool(torch.equal(expected, actual)),
        "max_action_difference": float((expected - actual).abs().max()),
        "actor_including_normalizer_exact": state_equal(saved["actor_state_dict"], clone.alg.actor.state_dict()),
        "critic_including_normalizer_exact": state_equal(saved["critic_state_dict"], clone.alg.critic.state_dict()),
        "optimizer_exact": state_equal(runner.alg.optimizer.state_dict(), clone.alg.optimizer.state_dict()),
        "completed_iterations": restored["completed_iterations"],
        "common_step_counter_restored": env.common_step_counter == restored["env_state"]["common_step_counter"],
        "actor_changed_after_updates": any(not torch.equal(v, dict(runner.alg.actor.named_parameters())[k]) for k, v in before.items()),
        "parameters_finite": all(bool(torch.isfinite(v).all()) for v in runner.alg.actor.parameters()),
    }
    json_write(out / "checkpoint_check.json", parity)
    if not all(parity[k] for k in ("deterministic_action_exact", "actor_including_normalizer_exact",
                                   "critic_including_normalizer_exact", "optimizer_exact",
                                   "common_step_counter_restored", "actor_changed_after_updates", "parameters_finite")):
        raise AssertionError(parity)
    resumed_before = {k: v.clone() for k, v in clone.alg.actor.named_parameters()}
    optimizer_before = max(float(s['step']) for s in clone.alg.optimizer.state.values() if 'step' in s)
    if args.resume_updates:
        wrapped.reset()
        clone.learn(args.resume_updates, init_at_random_ep_len=False)
        parity["resume_parameters_changed"] = any(
            not torch.equal(v, dict(clone.alg.actor.named_parameters())[k])
            for k, v in resumed_before.items())
        parity["resume_optimizer_step_increased"] = max(
            float(s['step']) for s in clone.alg.optimizer.state.values() if 'step' in s) > optimizer_before
        parity["resume_parameters_finite"] = all(bool(torch.isfinite(v).all()) for v in clone.alg.actor.parameters())
        parity["resume_iteration_correct"] = clone.current_learning_iteration + 1 == restored["completed_iterations"] + args.resume_updates
        json_write(out / "checkpoint_check.json", parity)
        assert all(parity[k] for k in ("resume_parameters_changed", "resume_optimizer_step_increased",
                                      "resume_parameters_finite", "resume_iteration_correct")), parity
    summary = {"status": "completed", "family": args.family, "seed": args.seed,
               "num_envs": args.num_envs, "rollout_steps": 24,
               "initial_updates": args.updates, "resumed_updates": args.resume_updates,
               "starting_completed_iterations": starting_iteration,
               "ending_completed_iterations": starting_iteration + args.updates + args.resume_updates,
               "new_transitions": args.num_envs * 24 * (args.updates + args.resume_updates),
               "elapsed_seconds": time.monotonic() - started,
               "checkpoint_check": parity,
               "latest_checkpoint": str(out / "resume/latest.pt" if args.resume_updates else out / "latest.pt")}
    # Snapshot collection exposure before the separate illustrative replay.
    if hasattr(cmd, "exposure"):
        summary["reset_transition_exposure"] = deepcopy(cmd.exposure)
    else:
        summary["reset_counts"] = dict(cmd.reset_counts)
        summary["transition_counts"] = dict(cmd.transition_counts)
    # Close unfinished collection episodes before saving their component ledger.
    cmd.cfg.recovery_fraction = 0.0
    wrapped.reset()
    # Environment owns raw per-component returns, distinct from RSL's normalized logger.
    if hasattr(cmd, "episode_records"):
        json_write(out / "episodes.json", cmd.episode_records)
        summary["episode_records"] = len(cmd.episode_records)
        summary["episode_dispositions"] = {
            kind: {
                "completed_horizon": sum(r['reset_kind'] == kind and r['complete_horizon'] for r in cmd.episode_records),
                "fall": sum(r['reset_kind'] == kind and r['fall'] for r in cmd.episode_records),
                "collection_interrupted": sum(r['reset_kind'] == kind and not r['complete_horizon'] and not r['fall'] for r in cmd.episode_records),
            } for kind in ('full', 'recovery_snippet')}
    # One illustrative deterministic replay, using guard starts only. This is a
    # setup diagnostic on the training target pool, not held-out competence.
    from human_punching.task10c.native_checks import video
    import numpy as np
    eval_start = len(cmd.episode_records)
    policy = clone.get_inference_policy() if args.resume_updates else runner.get_inference_policy()
    poses = []
    replay_trace = []
    replay_steps = 0
    env0_done = False
    obs = wrapped.get_observations()
    with torch.inference_mode():
        for _ in range(150):
            act = policy(obs)
            obs, reward, done, _ = wrapped.step(act)
            replay_steps += 1
            if bool(done[0]):
                env0_done = True
                break  # auto-reset already happened; do not render a teleport.
            poses.append(env.sim.data.qpos[0].cpu().numpy().copy())
            replay_trace.append({'elapsed_s': float(cmd.state.elapsed[0]),
                'requested_ttc_s': float(cmd.state.requested_ttc[0]),
                'target_world': cmd.state.target_world[0].cpu().tolist(),
                'encountered': bool(cmd.state.encountered[0])})
    if poses:
        video(env, poses, out / 'setup_policy.mp4', 'Phase 1 setup PPO: unqualified policy, non-solid target',
              targets=[r['target_world'] for r in replay_trace])
        np.savez_compressed(out / 'setup_policy.npz', qpos=np.asarray(poses))
    json_write(out / 'setup_policy_trace.json', replay_trace)
    json_write(out / 'setup_policy_episodes.json', cmd.episode_records[eval_start:])
    summary['setup_replay'] = {'frames_before_first_env0_terminal': len(poses),
        'actor_steps': replay_steps, 'transitions': replay_steps * args.num_envs,
        'env0_terminated': env0_done,
        'scope': 'deterministic illustration, 100% guard starts, training target pool'}
    summary['elapsed_seconds'] = time.monotonic() - started
    json_write(out / "summary.json", summary)
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "--phase2" in os.sys.argv:
        os.sys.argv.remove("--phase2")
        from human_punching.task10c.runner import main as phase2_main
        phase2_main()
    else:
        main()
