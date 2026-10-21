"""Versioned engineering learning recipes, separate from strict evaluation.

These pure definitions do not change any configuration merely by importing
this module. The historical recipe remains the default; the revised recipe
must be selected explicitly and measured under its own runtime contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

SCHEMA = "human_wbc_engineering_recipe_v1"
LEGACY_RECIPE_ID = "legacy_guard_start_v1"
REVISED_RECIPE_ID = "reference_start_guard_max_v1"


@dataclass(frozen=True)
class RecipeSettings:
    """Concrete controls for one training environment or strict panel."""

    reference_start_probability: float
    initial_support_grace_ticks: int
    guard_joint_posture_metric: str
    guard_joint_posture_weight: float

    def __post_init__(self) -> None:
        probability = self.reference_start_probability
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Reference-start probability must be finite in [0,1]")
        grace = self.initial_support_grace_ticks
        if type(grace) is not int or not 0 <= grace <= 4 or (grace and probability == 0):
            raise ValueError("Support grace requires reference starts and an integer0..4 ticks")
        if self.guard_joint_posture_metric not in ("rms", "maximum"):
            raise ValueError("Guard posture metric must be rms or maximum")
        weight = self.guard_joint_posture_weight
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError("Guard posture reward weight must be finite and nonnegative")

    def initialization_parameters(self) -> dict:
        return {"reference_start_probability": self.reference_start_probability,
                "initial_support_grace_ticks": self.initial_support_grace_ticks}

    def task_reward_weights(self, base: Mapping[str, float]) -> dict[str, float]:
        """Change only this named weight, never the supplied historical table."""
        if "guard_joint_posture" not in base:
            raise ValueError("Complete task weight table with guard_joint_posture required")
        return {**base, "guard_joint_posture": self.guard_joint_posture_weight}


@dataclass(frozen=True)
class EngineeringRecipe:
    recipe_id: str
    training: RecipeSettings

    def strict_panel(self) -> RecipeSettings:
        """Preserve the entire historical panel, including its diagnostic rewards."""
        return RecipeSettings(0.0, 0, "rms", 0.25)

    def manifest(self) -> dict:
        return {"schema": SCHEMA, "recipe_id": self.recipe_id,
                "training": asdict(self.training), "strict_panel": asdict(self.strict_panel()),
                "reference_start_scope": "frame_zero_at_ordinary_episode_reset_only",
                "training_reference_start_is_strict_ready_launch": False,
                "interior_reference_state_initialization": False,
                "support_grace_scope": "selected_training_start_no_support_flag_only_until_first_support",
                "physics_timestep_s": 0.005, "policy_update_period_s": 0.020,
                "strict_guard_and_complete_cycle_thresholds_changed": False,
                "external_support": False, "mid_cycle_state_replacement": False}

    @property
    def sha256(self) -> str:
        value = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode()
        return hashlib.sha256(value).hexdigest()


LEGACY_RECIPE = EngineeringRecipe(LEGACY_RECIPE_ID, RecipeSettings(0.0, 0, "rms", 0.25))
REVISED_RECIPE = EngineeringRecipe(REVISED_RECIPE_ID, RecipeSettings(0.75, 4, "maximum", 1.0))
RECIPE_IDS = (LEGACY_RECIPE_ID, REVISED_RECIPE_ID)
_REGISTRY = {recipe.recipe_id: recipe for recipe in (LEGACY_RECIPE, REVISED_RECIPE)}


def resolve_engineering_recipe(value: str | EngineeringRecipe = LEGACY_RECIPE_ID) -> EngineeringRecipe:
    """Reject an unknown ID or changed parameters hidden under a known version."""
    key = value.recipe_id if isinstance(value, EngineeringRecipe) else value
    if not isinstance(key, str) or key not in _REGISTRY:
        raise ValueError(f"Select an explicit engineering recipe from {RECIPE_IDS}")
    recipe = _REGISTRY[key]
    if isinstance(value, EngineeringRecipe) and value != recipe:
        raise ValueError("Changed recipe parameters require a new registered recipe version")
    return recipe


def verify_recipe_manifest(manifest: dict, expected_sha256: str) -> EngineeringRecipe:
    if not isinstance(manifest, dict):
        raise ValueError("Engineering recipe manifest must be an object")
    recipe = resolve_engineering_recipe(manifest.get("recipe_id"))
    try:
        actual_sha256 = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError("Engineering recipe manifest is not finite canonical JSON") from error
    if actual_sha256 != recipe.sha256 or expected_sha256 != recipe.sha256:
        raise ValueError("Engineering recipe manifest/hash differs from its registered version")
    return recipe
