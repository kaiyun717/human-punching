from types import SimpleNamespace

import pytest

from human_punching.wbc.training_session import completed_progress, observation_schema


def test_completed_counts_advance_instead_of_repeating_rsl_last_index():
    first = completed_progress(None, "canonical", 250)
    assert first.completed_iterations == first.stage_completed_iterations == 250
    from dataclasses import asdict
    second = completed_progress(asdict(first), "canonical", 250)
    assert second.completed_iterations == second.stage_completed_iterations == 500
    with pytest.raises(ValueError, match="cumulative|Cumulative"):
        completed_progress(asdict(second), "canonical", 1501)
    with pytest.raises(ValueError, match="Stage transition"):
        completed_progress(asdict(second), "domain", 1)


def test_stage_counts_preserve_prior_validation_history():
    progress = completed_progress({"completed_iterations": 2000, "stage": "domain",
        "stage_completed_iterations": 0, "completed_validation_panels": 8,
        "validation": {"selected_iteration": 1750}, "curriculum": {"level": 1}}, "domain", 250)
    assert progress.completed_iterations == 2250 and progress.stage_completed_iterations == 250
    assert progress.completed_validation_panels == 8
    assert progress.validation == {"selected_iteration": 1750}


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_iteration_counts(value):
    with pytest.raises(ValueError):
        completed_progress(None, "canonical", value)


def test_named_observation_schema_binds_order_dimensions_and_functions():
    from mjlab.managers.observation_manager import ObservationTermCfg
    def observation(env):
        return env
    manager = SimpleNamespace(
        active_terms={"actor": ["joints", "commands"], "critic": ["joints", "state"]},
        group_obs_term_dim={"actor": [(48,), (87,)], "critic": [(48,), (100,)]},
        group_obs_dim={"actor": (135,), "critic": (148,)},
        group_obs_concatenate={"actor": True, "critic": True},
        get_term_cfg=lambda group, name: ObservationTermCfg(func=observation, scale=2.))
    env = SimpleNamespace(observation_manager=manager,
                          cfg=SimpleNamespace(observations={"actor": {"history": 1},
                                                           "critic": {"history": 1}}))
    schema = observation_schema(env)
    assert [v["name"] for v in schema["groups"]["actor"]["terms"]] == ["joints", "commands"]
    assert schema["groups"]["critic"]["shape"] == [148]
    assert schema["previous_action"] == "executed_clipped_normalized_residual"
    manager.get_term_cfg = lambda group, name: ObservationTermCfg(func=observation, scale=3.)
    assert observation_schema(env) != schema
