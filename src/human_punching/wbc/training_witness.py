"""Read-only, explicitly scoped training-state witnesses around separate panels.

No forward, observation computation, integration or reset is called. Native arrays
and mutable human/manager buffers are hashed one at a time on CPU; GPU clones are
not retained during evaluation. Immutable reference/model assets are separately
bound by the session contract. RNG preservation belongs to
``validation.isolated_validation_actor`` (rank-local CUDA only).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import math

from .measurement_contract import digest


def _summary(value, depth=0):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        data = value.detach().contiguous().cpu()
        raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"kind": "tensor", "dtype": str(data.dtype), "shape": list(data.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"kind": "numpy", "dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": value.hex()}
    if depth >= 5:
        return None
    if isinstance(value, Mapping):
        return {str(key): _summary(item, depth+1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summary(item, depth+1) for item in value]
    return None


def _buffers(obj, *, omit=()):
    """Capture direct buffers and their containers, never walk back into env."""
    result = {}
    for name, value in vars(obj).items():
        if name in omit or name in {"_env", "env", "cfg", "config"} or callable(value):
            continue
        item = _summary(value)
        if item is not None or value is None:
            result[name] = item
    return result


def training_state_witness(env, *, runner=None) -> dict:
    """Hash live state without changing its history, clocks, sampler or RNG.

    The listed witness coverage is inspectable. It is stronger than an asserted
    nonmutation Boolean, but is not a serialization of every simulator workspace.
    """
    import warp as wp
    from .telemetry.native_qualification import PHYSICAL_FIELDS
    from .telemetry.cycle_runtime import get_cycle_runtime
    from .telemetry.runtime_telemetry import get_observer, native_tensor
    native = {}
    for name in (*PHYSICAL_FIELDS, "qacc_warmstart", "act", "mocap_pos", "mocap_quat"):
        if hasattr(env.sim.data, name):
            native[name] = _summary(native_tensor(getattr(env.sim.data, name)))
    if not {"qpos", "qvel", "ctrl", "time", "qacc_warmstart"} <= native.keys():
        raise ValueError("Training-state witness lacks native state or solver warmstart")
    native["efc.force"] = _summary(wp.to_torch(env.sim.wp_data.efc.force))
    for name in ("geom", "worldid", "pos", "dist", "dim", "efc_address"):
        native["contact."+name] = _summary(wp.to_torch(getattr(env.sim.wp_data.contact, name)))
    for name in ("nacon", "overflow"):
        native[name] = _summary(wp.to_torch(getattr(env.sim.wp_data, name)))
    command = env.command_manager.get_term("motion")
    action = env.action_manager.get_term("joint_pos")
    observer, bridge = get_observer(env), get_cycle_runtime(env)
    fields = {"native": native,
        "environment": {name: _summary(getattr(env, name)) for name in (
            "common_step_counter", "_sim_step_counter", "episode_length_buf", "reset_buf",
            "reset_terminated", "reset_time_outs", "reward_buf") if hasattr(env, name)},
        "command": _buffers(command, omit=("master", "motion", "clips", "_family_clips")),
        "action": _buffers(action), "action_manager": _buffers(env.action_manager),
        "observations": _buffers(env.observation_manager),
        "observer": _buffers(observer), "observer_window": _buffers(observer.window),
        "bridge": _buffers(bridge), "readiness": _buffers(bridge.readiness),
        "events": _buffers(bridge.events),
        "training_initializer": {"buffers": _buffers(bridge.training_initializer),
            "configuration": asdict(bridge.training_initializer.config)}}
    # History implementation differs by manager version; include nested buffer
    # objects explicitly, without invoking their compute/accessor methods.
    for kind in ("history", "delay"):
        histories = getattr(env.observation_manager, f"_group_obs_term_{kind}_buffer", {})
        fields["term_"+kind] = {str(group): {str(term): {
            "buffers": _buffers(buffer),
            "nested": {name: _buffers(value) for name, value in vars(buffer).items()
                       if name in ("_circular_buffer", "_buffer") and hasattr(value, "__dict__")}}
            for term, buffer in terms.items()} for group, terms in histories.items()}
    if runner is not None:
        fields["learner"] = _summary(runner.alg.save())
        fields["runner_iteration"] = runner.current_learning_iteration
        fields["logger_counters"] = {name: getattr(runner.logger, name) for name in
                                     ("tot_timesteps", "tot_time")}
    payload = {"schema": "human_wbc_training_state_witness_v1", "fields": fields,
               "no_environment_methods_called": True, "gpu_state_clones_retained": False}
    return {**payload, "sha256": digest(payload)}


def assert_training_state_unchanged(before: dict, after: dict) -> None:
    for item in (before, after):
        if (item.get("schema") != "human_wbc_training_state_witness_v1"
                or item.get("sha256") != digest({k: v for k, v in item.items() if k != "sha256"})):
            raise ValueError("Invalid training-state witness identity")
    if before["sha256"] != after["sha256"]:
        changed = [name for name in before["fields"]
                   if before["fields"].get(name) != after["fields"].get(name)]
        raise RuntimeError("Independent validation changed training state: "+", ".join(changed))
