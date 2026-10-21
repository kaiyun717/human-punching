"""Saved-array qualification rejects false readiness, homing and grace claims."""
import copy
import numpy as np
import pytest

from human_punching.wbc.telemetry.training_initialization_qualification import (
    array_identity, audit_initialization,
)


def recorded_fixture():
    count, worlds = 8, 2
    selected = np.tile([True, False], (count, 1))
    tick = np.broadcast_to(np.arange(count)[:, None], (count, worlds)).copy()
    zeros = np.zeros((count, worlds), int)
    condition = np.zeros((count, worlds, 91), np.float32)
    condition[:, 0, -3] = 1.; condition[:, 1, -4] = 1.
    condition[:, 0, 8] = np.arange(count)*.005
    pre = dict(training_selected=selected, episode_tick=tick,
        reference_tick=np.where(selected, tick, 0), launch_tick=np.where(selected, 0, -1),
        launched=selected, phase=np.where(selected, 1, 0), quiet_intervals=zeros.copy(),
        ready_endpoint=np.zeros_like(selected), event_previous_guard=np.zeros_like(selected),
        initialization_prepared=np.ones_like(selected), event_armed=selected,
        request_id=np.tile([10, 11], (count, 1)), target=np.zeros((count, worlds, 3)),
        actor_context=np.zeros((count, worlds, 77)), plan_handle=zeros.copy(),
        plan_digest=np.ones((count, worlds, 32), np.uint8), g1_launch_state=np.zeros((count, worlds, 7)),
        trajectory_id=np.tile([0, 1], (count, 1)), event_active=selected,
        support_seen=np.zeros_like(selected), grace_total_ticks=zeros.copy(),
        support_contact=np.ones((count, worlds, 2), bool), condition=condition,
        qref=np.zeros((count, worlds, 48)), future12=np.zeros((count, worlds, 48)),
        future24=np.zeros((count, worlds, 48)))
    pre['support_contact'][0] = False
    pre['support_seen'][2:, 0] = True
    pre['grace_total_ticks'][1:, 0] = 1
    support = np.ones((count, worlds, 2), bool); support[0, 0] = False
    endpoint = dict(support_contact=support, grace_total_ticks=zeros.copy(), grace_ticks=zeros.copy())
    endpoint['grace_total_ticks'][:, 0] = 1; endpoint['grace_ticks'][:, 0] = 1
    def row(t, pending):
        values = {name: pre[name][min(t, count-1):min(t, count-1)+1].copy()
            for name in ('training_selected', 'condition', 'future12', 'future24')}
        values.update(pending_reset=np.full((1, worlds), pending),
            effective_reference_active=np.asarray([[True, False]]),
            effective_reference_tick=np.asarray([[t, 0]]),
            current_reference=np.zeros((1, worlds, 48)), future_actor_error=np.zeros((1, worlds)))
        values['condition'][0, 0, 8] = t*.005
        return values
    initial = row(0, True)
    a, b = row(4, False), row(8, False)
    policy = {name: np.concatenate((a[name], b[name])) for name in a}
    receipt = {'acceptance_eligible': False, 'initialization': {
        'physical_reset_only': True, 'interior_rsi': False,
        'config': {'reference_start_probability': .5, 'initial_support_grace_ticks': 4},
        'start_type_codes': [1, 0], 'strict_launch_eligible': [False, True],
        'training_reference_start': [True, False], 'strict_ready_launch': [False, False],
        'initial_support_grace_ticks_limit': 4}}
    return pre, endpoint, policy, initial, [receipt]


def audit(data, **kwargs):
    pre, endpoint, policy, initial, receipts = data
    return audit_initialization(pre, endpoint, policy, initial, probability=.5,
        master_unchanged=True, receipts=receipts, **kwargs)


def test_mixed_frame_zero_exposure_does_not_claim_strict_launch():
    result = audit(recorded_fixture())
    assert result['status'] == 'passed', result
    assert result['training_frame_zero_starts'] == 1
    assert result['strict_guard_reset_starts'] == 1
    assert result['actual_suppressed_no_support_endpoints'] == 1
    assert result['training_support_records'] == [{'environment_id': 0, 'request_id': 10,
        'first_actual_support_tick': 2, 'unsupported_active_endpoints': 1,
        'suppressed_no_support_endpoints': 1, 'suppressed_tick_indices': [1]}]
    assert result['strict_launch_qualification'].startswith('not_granted')


@pytest.mark.parametrize('kind,expected', [
    ('fake_dwell', 'reset_has_no_fabricated_readiness_dwell'),
    ('phase', 'first_observation_training_phase'),
    ('first_reference', 'first_action_reference_unchanged_after_reset_commit'),
    ('target_chase', 'full_plan_identity_target_and_g1_context_frozen'),
    ('late_grace', 'grace_exactly_matches_initial_unsupported_endpoints'),
    ('too_much_grace', 'grace_has_four_tick_absolute_cap'),
    ('false_eligibility', 'all_receipts_explicitly_ineligible_and_start_typed'),
])
def test_corrupted_saved_claim_is_rejected(kind, expected):
    data = recorded_fixture(); pre, endpoint, policy, initial, receipts = data
    if kind == 'fake_dwell': pre['quiet_intervals'][0, 0] = 40
    elif kind == 'phase': initial['condition'][0, 0, -4:] = [1, 0, 0, 0]
    elif kind == 'first_reference': initial['current_reference'][0, 0, 3] = .1
    elif kind == 'target_chase': pre['target'][3:, 0, 0] = .05
    elif kind == 'late_grace': endpoint['grace_total_ticks'][5, 0] += 1
    elif kind == 'too_much_grace': endpoint['grace_ticks'][5, 0] = 5
    elif kind == 'false_eligibility': receipts[0]['initialization']['strict_launch_eligible'][0] = True
    result = audit(data)
    assert result['status'] == 'failed'
    assert result['checks'][expected] is False


def test_full_reference_hash_binds_names_shape_dtype_and_values():
    original = {'qpos': np.zeros((2, 55), np.float32)}
    digest = array_identity(original)
    assert digest == array_identity(copy.deepcopy(original))
    assert digest != array_identity({'qpos': original['qpos'].astype(np.float64)})
    assert digest != array_identity({'qpos': original['qpos'].reshape(1, 110)})
    changed = copy.deepcopy(original); changed['qpos'][0, 7] = .01
    assert digest != array_identity(changed)


def test_untyped_terminal_receipt_never_passes():
    data = recorded_fixture(); data[-1][0].pop('initialization')
    assert not audit(data)['checks']['all_receipts_explicitly_ineligible_and_start_typed']
