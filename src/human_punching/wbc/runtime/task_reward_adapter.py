"""Opt-in reward-manager adapter for the qualified native observer."""
from human_punching.wbc.runtime.task_rewards import (
    DEFAULT_WEIGHTS, task_reward_rates, validate_guard_joint_posture_metric,
)


class RewardCache:
    """Compute the task terms once per completed four-tick policy window."""
    def __init__(self, observer, context_provider, *, guard_joint_posture_metric="rms"):
        self._guard_joint_posture_metric = validate_guard_joint_posture_metric(guard_joint_posture_metric)
        self.observer = observer
        self.context_provider = context_provider
        self.serial = -1
        self.cache = None

    @property
    def guard_joint_posture_metric(self):
        return self._guard_joint_posture_metric

    def values(self):
        observer = self.observer
        window = observer.window_summary()  # validates four actual samples
        serial = observer.actual_samples
        if type(serial) is not int or serial <= 0 or serial % 4:
            raise RuntimeError("Rewards require a completed native policy window")
        if serial != self.serial:
            context = self.context_provider()
            self.cache = task_reward_rates(observer.endpoint_state(), context["goal"],
                window, context, observer.window.caps,
                guard_joint_posture_metric=self.guard_joint_posture_metric)
            self.serial = serial
        return self.cache


def native_task_reward(env, name, *, guard_joint_posture_metric="rms"):
    metric = validate_guard_joint_posture_metric(guard_joint_posture_metric)
    from human_punching.wbc.telemetry.cycle_runtime import get_cycle_runtime
    from human_punching.wbc.telemetry.runtime_telemetry import get_observer
    bridge = get_cycle_runtime(env)
    cache = getattr(bridge, "_task_reward_cache", None)
    if cache is None:
        cache = RewardCache(get_observer(env), bridge.reward_context, guard_joint_posture_metric=metric)
        bridge._task_reward_cache = cache
    elif cache.guard_joint_posture_metric != metric:
        raise ValueError("Conflicting guard joint posture metric for reward cache")
    return cache.values()[name]


def configure_task_rewards(cfg, weights=None, *, guard_joint_posture_metric="rms"):
    """Replace stale shaping while retaining WBC and the historical RMS default.

    Nondefault metric selection is carried in every term's params so the
    measurement contract identifies it and all terms share one cache mode.
    Historical RMS params remain unchanged for recipe compatibility.
    """
    metric = validate_guard_joint_posture_metric(guard_joint_posture_metric)
    from mjlab.managers.reward_manager import RewardTermCfg
    if (cfg.decimation != 4 or abs(cfg.sim.mujoco.timestep-.005) > 1e-12
            or not cfg.scale_rewards_by_dt):
        raise ValueError("Task rewards require 5/20 ms clocks and reward rate integration")
    if "human_native_substep_valid" not in cfg.metrics:
        raise ValueError("Enable the qualified native observer before task rewards")
    selected = DEFAULT_WEIGHTS if weights is None else weights
    if set(selected) != set(DEFAULT_WEIGHTS):
        raise ValueError("Explicit complete task objective weights required")
    import math
    if any(not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w)
           for w in selected.values()):
        raise ValueError("Finite task reward weights required")
    if any("human_"+name in cfg.rewards for name in selected):
        raise ValueError("Task objectives are already configured")
    cfg.rewards.pop("timed_target", None)
    # Replace generic sole slip with measured load-bearing material-patch slip.
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = 0.
    for name, weight in selected.items():
        params = {"name": name}
        if metric != "rms":
            params["guard_joint_posture_metric"] = metric
        cfg.rewards["human_"+name] = RewardTermCfg(func=native_task_reward,
                                                weight=weight, params=params)
    return cfg
