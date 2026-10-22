"""Evaluation integrity checks independent of GPU policy quality."""
from copy import deepcopy
import math

import numpy as np
import pytest

from human_punching.task10c.evaluate import (
    create_recipes, select_representatives, summarize_attempts, validate_recipes,
    validate_checkpoint_identity,
)
from human_punching.task10c.targets import PROJECT_ROOT, TargetPool

POOL = PROJECT_ROOT / "outputs/task10c/setup/target_pool.npz"


def test_recipes_preserve_original_splits_heads_and_are_reproducible():
    dev = create_recipes(POOL, "validation", 64, 17, (.55, .68), .25)
    test = create_recipes(POOL, "test", 256, 19, (.5, .9), 1.)
    assert dev == create_recipes(POOL, "validation", 64, 17, (.55, .68), .25)
    validate_recipes(dev)
    validate_recipes(test)
    dev_ids = {row["snapshot_id"] for row in dev["rows"]}
    test_ids = {row["snapshot_id"] for row in test["rows"]}
    train_ids = set(TargetPool.load(POOL).snapshot_ids("train"))
    assert not dev_ids & test_ids and not train_ids & (dev_ids | test_ids)
    assert len(dev["rows"]) == 64 and len(test["rows"]) == 256
    assert dev["candidate_target_count"] == 32
    assert len(test_ids) == 128
    with pytest.raises(ValueError, match="validation or test"):
        create_recipes(POOL, "train", 64, 17, (.55, .68))


def test_recipe_mutation_is_detected_before_execution():
    recipes = create_recipes(POOL, "validation", 64, 17, (.55, .68), .25)
    mutated = deepcopy(recipes)
    mutated["rows"][0]["target_world"][2] += .2
    with pytest.raises(ValueError, match="digest"):
        validate_recipes(mutated)


def test_evaluation_checkpoint_rejects_setup_and_identity_mismatch():
    infos = {"phase": "task10c_phase2_jab", "family": "jab", "lineage_id": "fresh-jab",
             "reference_sha256": "reference", "target_pool_sha256": "pool"}
    validate_checkpoint_identity(infos, "reference", "pool")
    for key, value in (("phase", "phase1_setup"), ("family", "cross"),
                       ("reference_sha256", "other"), ("target_pool_sha256", "other")):
        with pytest.raises(ValueError):
            validate_checkpoint_identity({**infos, key: value}, "reference", "pool")


def _row(recipe_id, hit=False, actual=None, fall=False, recovered=False):
    return {"recipe_id": recipe_id, "attempted": True, "disposition": "completed",
        "hit": hit, "actual_ttc_s": actual, "requested_ttc_s": .6,
        "signed_timing_error_s": None if actual is None else actual - .6,
        "fall": fall, "settled_terminal_guard": recovered,
        "complete_horizon": not fall, "inactive_encounter": False, "launch_overlap": False,
        "settled_guard_elapsed_s": 1.1 if recovered else None,
        "guard_glove_error_m": .05 if recovered else .25,
        "guard_glove_speed_m_s": .1 if recovered else 1.,
        "raw_component_totals": {"encounter": 10. if hit else 0., "guard": .4}}


def test_summary_uses_hits_for_timing_and_all_attempts_for_outcomes():
    recipes = create_recipes(POOL, "test", 4, 19, (.55, .65))
    rows = [_row(0, hit=True, actual=.4, recovered=True),
            _row(1, hit=True, actual=.6, fall=True), _row(2),
            {"recipe_id": 3, "attempted": True, "disposition": "failed_before_terminal_record"}]
    result = summarize_attempts(rows, recipes)
    assert result["requested_attempts"] == result["attempted"] == 4
    assert result["completed"] == result["hit_denominator"] == 3
    assert result["hit_count"] == result["timing_error_denominator"] == 2
    assert result["miss_count"] == result["incomplete_or_failed"] == 1
    assert result["early_encounters_below_0_5_s"] == 1
    assert result["timing_within_0_10_s_count"] == 1
    assert result["recovery"]["hits"]["denominator"] == 2
    assert result["recovery"]["misses"]["denominator"] == 1
    assert result["recovery"]["hits"]["settled_at_terminal"] == 1
    assert np.isclose(result["signed_timing_error_s"]["mean"], -.1)
    assert math.isfinite(result["selection_score"])


def test_representatives_include_failures_and_keep_first_recipe():
    rows = [_row(0, hit=True, actual=.6, recovered=True),
            _row(1, hit=True, actual=.4), _row(2, fall=True), _row(3)]
    chosen = select_representatives(rows, 4)
    assert chosen[0][0]["recipe_id"] == 0
    assert any(row["fall"] for row, _ in chosen)
    assert any(not row["hit"] for row, _ in chosen)
    assert len({row["recipe_id"] for row, _ in chosen}) == len(chosen)


def test_representatives_include_surviving_recovery_despite_worse_timing():
    rows = [_row(0, hit=True, actual=.6, fall=True),
            _row(1, hit=True, actual=.601, fall=True),
            _row(2, hit=True, actual=.65, recovered=True),
            _row(3, hit=True, actual=.4), _row(4)]
    chosen = select_representatives(rows, 6)
    recovery_row, reason = chosen[1]
    assert recovery_row["recipe_id"] == 2
    assert recovery_row["complete_horizon"] and not recovery_row["fall"]
    assert recovery_row["settled_terminal_guard"]
    assert reason == "lowest_guard_error_settled_hit"
    # If no attempt satisfies the existing settled diagnostic, retain a real
    # surviving hit and label it without implying that recovery succeeded.
    rows[2]["settled_terminal_guard"] = False
    chosen = select_representatives(rows, 6)
    assert chosen[1][0]["recipe_id"] == 2
    assert chosen[1][1] == "lowest_guard_error_hit_with_full_horizon_without_fall"
