"""Global PPO accounting and bounded stage validation; no custom learner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainingScale:
    global_environments: int = 8192
    world_size: int = 2
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Training dimensions must be positive integers")
        if self.global_environments % self.world_size:
            raise ValueError("Global environments must divide evenly across ranks")
        if self.transitions_per_rank % self.minibatches:
            raise ValueError("Each rank's rollout must divide into equal minibatches")

    @property
    def environments_per_rank(self) -> int:
        return self.global_environments // self.world_size

    @property
    def transitions_per_rank(self) -> int:
        return self.environments_per_rank * self.rollout_steps

    @property
    def fresh_transitions(self) -> int:
        return self.global_environments * self.rollout_steps

    @property
    def optimizer_steps(self) -> int:
        return self.epochs * self.minibatches

    def report(self, iterations: int = 10_000) -> dict:
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Iterations must be a positive integer")
        return {**asdict(self), "environments_per_rank": self.environments_per_rank,
                "fresh_transitions_per_iteration": self.fresh_transitions,
                "global_minibatch_size": self.fresh_transitions // self.minibatches,
                "configured_optimizer_steps_per_iteration": self.optimizer_steps,
                "iterations": iterations, "total_transitions": iterations * self.fresh_transitions,
                "allocation_authorized": False}


STAGE_CEILINGS = {"canonical": 2000, "domain": 6000, "guard_start": 2000}


def projected_hours(transitions: int, transitions_per_second: float) -> float:
    if transitions <= 0 or not math.isfinite(transitions_per_second) or transitions_per_second <= 0:
        raise ValueError("Projection requires positive transitions and measured throughput")
    return transitions / transitions_per_second / 3600


def plateau(panels: list[dict[str, float]], *, success_gain: float = .01,
            relative_metric_gain: float = .01) -> bool:
    """Four panels with neither worst-family success nor error improvement.

    Inputs are validation-only aggregates. Lower targeting/timing/recovery errors
    are better; missing/nonfinite panels cannot trigger a plateau decision.
    """
    if len(panels) < 4:
        return False
    recent = panels[-4:]
    keys = ("worst_family_success", "targeting_error", "timing_error", "recovery_time")
    if any(k not in p or not math.isfinite(p[k]) for p in recent for k in keys):
        raise ValueError("Plateau panels must contain finite validation metrics")
    if max(p["worst_family_success"] for p in recent[1:]) - recent[0]["worst_family_success"] >= success_gain:
        return False
    for key in keys[1:]:
        first = recent[0][key]
        if first < 0 or any(p[key] < 0 for p in recent):
            raise ValueError("Error metrics cannot be negative")
        if first > 0 and (first - min(p[key] for p in recent[1:])) / first >= relative_metric_gain:
            return False
    return True
