"""Evaluate unchanged WBC tracking kernels against fresh physical geometry.

Pinned mjlab computes rewards before its final live forward refresh. This local
adapter supplies the already computed private endpoint FK to WBC's existing
reward functions. It neither copies their formulas nor steps/forwards/writes
simulation. Weights, kernels, body selection, and reference conditioning remain
the configured WBC choices. Actor observations keep their normal post-forward
path. This is an explicit new objective variant, requiring scratch training.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping

import torch

from human_punching.wbc.telemetry.runtime_telemetry import get_observer

CACHE_ATTRIBUTE = "_human_fresh_imitation_cache"
UPSTREAM_NAMES = (
    "motion_global_anchor_position_error_exp", "motion_global_anchor_orientation_error_exp",
    "motion_anchor_linear_velocity_body_error_exp", "motion_anchor_angular_velocity_body_error_exp",
    "motion_relative_body_position_error_exp", "motion_relative_body_orientation_error_exp",
    "motion_global_body_linear_velocity_error_exp", "motion_global_body_angular_velocity_error_exp",
)


def body_endpoint(arrays: Mapping[str, torch.Tensor], body_ids: torch.Tensor,
                  tree_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Named body origins and world twists; cvel's linear part is at tree COM."""
    position = arrays["xpos"].index_select(1, body_ids)
    quaternion = arrays["xquat"].index_select(1, body_ids)
    velocity = arrays["cvel"].index_select(1, body_ids)
    com = arrays["subtree_com"].index_select(1, tree_ids)
    return {"position": position, "quaternion": quaternion,
        "linear_velocity": velocity[..., 3:] + torch.linalg.cross(velocity[..., :3], position - com),
        "angular_velocity": velocity[..., :3]}


class FreshCommand:
    def __init__(self, command, endpoint, anchor_index):
        self._command = command
        for suffix, key in (("pos", "position"), ("quat", "quaternion"),
                            ("lin_vel", "linear_velocity"), ("ang_vel", "angular_velocity")):
            setattr(self, f"robot_body_{suffix}_w", endpoint[key])
            setattr(self, f"robot_anchor_{suffix}_w", endpoint[key][:, anchor_index])

    def __getattr__(self, name):
        return getattr(self._command, name)


class FreshEnvironment:
    def __init__(self, env, command):
        self._env, self._command = env, command
        self.command_manager = SimpleNamespace(get_term=self.get_term)

    def get_term(self, name):
        if name != "motion":
            raise ValueError("Fresh human imitation currently binds the motion command only")
        return self._command

    def __getattr__(self, name):
        return getattr(self._env, name)


class FreshImitationCache:
    """One named extraction per policy window; never refreshes native state."""
    def __init__(self, env, observer):
        self.env, self.observer = env, observer
        command = env.command_manager.get_term("motion")
        names = tuple(command.cfg.body_names)
        if len(set(names)) != len(names) or command.cfg.anchor_body_name not in names:
            raise ValueError("Distinct tracked body names including the anchor are required")
        prefix = observer.fk.bindings.spec.prefix
        model = env.sim.mj_model
        ids = [int(model.body(prefix + name).id) for name in names]
        self.body_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
        self.tree_ids = torch.tensor([int(model.body_rootid[i]) for i in ids], dtype=torch.long, device=env.device)
        self.anchor_index = names.index(command.cfg.anchor_body_name)
        self.command = command
        self.serial = -1
        self.view = None
        self.raw_values = {}
        self.observed_windows = 0
        self.term_calls = {name: 0 for name in UPSTREAM_NAMES}
        self.nonfinite_counts = torch.zeros(len(UPSTREAM_NAMES), dtype=torch.int64, device=env.device)

    def get(self):
        # Fail closed if called before a complete physical 20 ms reward window.
        self.observer.window_summary()
        serial = self.observer.actual_samples
        if serial != self.serial:
            # _arrays belong to our human extension's private FK workspace.
            # Index selection copies named outputs; no live simulator field is
            # overwritten or refreshed by these reward accessors.
            values = body_endpoint(self.observer.fk._arrays, self.body_ids, self.tree_ids)
            command = FreshCommand(self.command, values, self.anchor_index)
            self.view = FreshEnvironment(self.env, command)
            self.serial = serial
            self.raw_values = {}
            self.observed_windows += 1
        return self.view


    def record(self, name, value):
        # Retain the actual invocation result before RewardManager sanitizes it.
        # A diagnostic read must not reevaluate against a later command frame.
        if name in self.raw_values:
            raise RuntimeError("Fresh imitation term evaluated twice in one physical reward window")
        if value.ndim != 1:
            raise ValueError("Fresh imitation kernel must return one value per environment")
        self.raw_values[name] = value.detach().clone()
        self.term_calls[name] += 1
        self.nonfinite_counts[UPSTREAM_NAMES.index(name)].add_((~torch.isfinite(value)).sum())


def fresh_imitation_health(env) -> dict:
    """Read accumulated raw-output evidence; called only at phase boundaries."""
    cache = getattr(env, CACHE_ATTRIBUTE, None)
    counts = ([0] * len(UPSTREAM_NAMES) if cache is None else cache.nonfinite_counts.cpu().tolist())
    calls = ({name: 0 for name in UPSTREAM_NAMES} if cache is None else dict(cache.term_calls))
    windows = 0 if cache is None else cache.observed_windows
    return {"configured_terms": list(UPSTREAM_NAMES), "observed_windows": windows,
        "term_call_counts": calls, "term_nonfinite_counts": dict(zip(UPSTREAM_NAMES, counts)),
        "all_raw_terms_finite": not any(counts),
        "every_term_each_window": all(value == windows for value in calls.values()),
        "scope": "Actual unweighted WBC outputs before RewardManager sanitization"}


def fresh_imitation_reward(env, *, upstream_reward, **params):
    from wbc_mjlab.env.mdp import rewards
    allowed = tuple(getattr(rewards, name) for name in UPSTREAM_NAMES)
    if upstream_reward not in allowed or params.get("command_name") != "motion":
        raise ValueError("Only the pinned WBC human tracking kernels are supported")
    observer = get_observer(env)
    cache = getattr(env, CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = FreshImitationCache(env, observer)
        setattr(env, CACHE_ATTRIBUTE, cache)
    value = upstream_reward(cache.get(), **params)
    cache.record(upstream_reward.__name__, value)
    return value


def configure_fresh_imitation_rewards(cfg) -> dict:
    """Explicit opt-in; leave all upstream code, parameters and weights intact."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from wbc_mjlab.env.mdp import rewards
    observer = cfg.metrics.get("human_native_substep_valid")
    if (cfg.sim.mujoco.timestep != .005 or cfg.decimation != 4 or observer is None
            or not observer.per_substep or not observer.params["options"].enabled):
        raise ValueError("Fresh imitation requires the qualified 5 ms native observer")
    allowed = tuple(getattr(rewards, name) for name in UPSTREAM_NAMES)
    selected = {name: term for name, term in cfg.rewards.items() if term.func in allowed}
    if len(selected) != len(UPSTREAM_NAMES) or any(term.params.get("command_name") != "motion" for term in selected.values()):
        raise ValueError("Expected all eight configured WBC body/anchor tracking kernels")
    for name, term in selected.items():
        cfg.rewards[name] = RewardTermCfg(func=fresh_imitation_reward, weight=term.weight,
            params={"upstream_reward": term.func, **term.params})
    return {"schema": "human_wbc_fresh_imitation_v1", "terms": list(selected),
        "geometry_timestamp": "final_native_endpoint", "reference_timestamp": "final_native_endpoint",
        "formulas": "unchanged_pinned_WBC_functions", "extra_integration_or_forward": False,
        "native_variant_qualified": False}


def require_fresh_imitation_rewards(cfg) -> None:
    from wbc_mjlab.env.mdp import rewards
    allowed = tuple(getattr(rewards, name) for name in UPSTREAM_NAMES)
    selected = [term for term in cfg.rewards.values() if term.func is fresh_imitation_reward]
    observer = cfg.metrics.get("human_native_substep_valid")
    if (cfg.sim.mujoco.timestep != .005 or cfg.decimation != 4 or observer is None
            or not observer.per_substep or not observer.params["options"].enabled
            or len(selected) != len(allowed)
            or {term.params.get("upstream_reward") for term in selected} != set(allowed)
            or any(term.params.get("command_name") != "motion" for term in selected)
            or any(term.func in allowed for term in cfg.rewards.values())):
        raise ValueError("All eight WBC body/anchor kernels must use fresh endpoint geometry")
