"""CPU-only versioning and strict-panel contracts; no runtime construction."""
from dataclasses import FrozenInstanceError, replace
import math
import pytest
from human_punching.wbc.runtime.engineering_recipe import (
    LEGACY_RECIPE, LEGACY_RECIPE_ID, REVISED_RECIPE, REVISED_RECIPE_ID,
    RecipeSettings, resolve_engineering_recipe, verify_recipe_manifest,
)


def test_default_preserves_ct01_parameters():
    recipe = resolve_engineering_recipe()
    assert recipe is LEGACY_RECIPE
    assert recipe.recipe_id == LEGACY_RECIPE_ID
    assert recipe.training == RecipeSettings(0.0, 0, "rms", .25)
    assert recipe.strict_panel() == recipe.training


def test_revised_training_is_explicit_and_panels_remain_strict():
    recipe = resolve_engineering_recipe(REVISED_RECIPE_ID)
    assert recipe is REVISED_RECIPE
    assert recipe.training.initialization_parameters() == {
        "reference_start_probability": .75, "initial_support_grace_ticks": 4}
    panel = recipe.strict_panel()
    assert panel.initialization_parameters() == {
        "reference_start_probability": 0.0, "initial_support_grace_ticks": 0}
    assert panel.guard_joint_posture_metric == "rms"
    assert panel.guard_joint_posture_weight == .25
    assert panel == LEGACY_RECIPE.training
    assert recipe.manifest()["strict_guard_and_complete_cycle_thresholds_changed"] is False


def test_changed_weight_does_not_mutate_or_change_other_weights():
    baseline = {"guard_joint_posture": .25, "guard_footprint": 1., "anything_else": -2.}
    selected = REVISED_RECIPE.training.task_reward_weights(baseline)
    assert baseline["guard_joint_posture"] == .25
    assert selected == {**baseline, "guard_joint_posture": 1.}
    with pytest.raises(ValueError):
        REVISED_RECIPE.training.task_reward_weights({"guard_footprint": 1.})


def test_recipe_identity_is_frozen_and_changes_require_a_version():
    assert LEGACY_RECIPE.sha256 != REVISED_RECIPE.sha256
    assert verify_recipe_manifest(REVISED_RECIPE.manifest(), REVISED_RECIPE.sha256) is REVISED_RECIPE
    with pytest.raises(FrozenInstanceError):
        REVISED_RECIPE.training.reference_start_probability = .5
    modified = replace(REVISED_RECIPE, training=replace(REVISED_RECIPE.training, reference_start_probability=.5))
    with pytest.raises(ValueError, match="new registered"):
        resolve_engineering_recipe(modified)
    manifest = REVISED_RECIPE.manifest()
    manifest["strict_panel"]["initial_support_grace_ticks"] = 4
    with pytest.raises(ValueError, match="manifest/hash"):
        verify_recipe_manifest(manifest, REVISED_RECIPE.sha256)
    with pytest.raises(ValueError):
        resolve_engineering_recipe("unregistered_recipe")


@pytest.mark.parametrize("changes", [
    {"reference_start_probability": True}, {"reference_start_probability": math.nan},
    {"reference_start_probability": -1.}, {"reference_start_probability": 1.1},
    {"initial_support_grace_ticks": True}, {"initial_support_grace_ticks": 4.0},
    {"initial_support_grace_ticks": 5}, {"reference_start_probability": 0., "initial_support_grace_ticks": 1},
    {"guard_joint_posture_metric": "mean"}, {"guard_joint_posture_weight": math.inf},
    {"guard_joint_posture_weight": True}, {"guard_joint_posture_weight": -1.},
])
def test_malformed_or_unbounded_recipe_settings_rejected(changes):
    with pytest.raises(ValueError):
        replace(REVISED_RECIPE.training, **changes)


def test_manifest_numeric_boolean_alias_cannot_forge_canonical_hash():
    manifest = REVISED_RECIPE.manifest()
    manifest["strict_panel"]["reference_start_probability"] = False
    with pytest.raises(ValueError, match="manifest/hash"):
        verify_recipe_manifest(manifest, REVISED_RECIPE.sha256)
