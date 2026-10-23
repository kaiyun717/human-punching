"""CPU regressions for the approved per-run nominal jab reference selection."""
from pathlib import Path
from types import MethodType, SimpleNamespace
import hashlib

import numpy as np
import pytest
import torch

from human_punching.task10c.commands import PunchCommand, TorchReference
from human_punching.task10c.env_cfg import make_env_cfg, selected_reference_path
from human_punching.task10c.events import PunchState
from human_punching.task10c.robot import JOINT_NAMES, ROOT


ORIGINAL = ROOT / 'outputs/task10c/references/jab.npz'
CANDIDATE = ROOT / 'outputs/task10c/jab/nominal_reference_adaptation_20261022_01/final/candidate/jab.npz'
APPROVED_NPZ_SHA256 = '50887274783da7364d8b7ce39ee7f956c583174b0339ca602d65bfd71ff600e8'
APPROVED_JSON_SHA256 = '96bf91e8a60f66847adfaa9f56a0319410424a522db5c53fba64b3e5ec36a74d'
EASY_TTC = (.555, .6783333333333333)
TIMING = dict(clock_mode='smooth_entry', ttc_anchor_time=.510, post_entry_duration=.910)


@pytest.fixture(scope='module')
def candidate():
    return TorchReference(CANDIDATE, 'cpu', JOINT_NAMES, family='jab', **TIMING)


def test_explicit_selection_keeps_approved_payload_metadata_and_historical_default(candidate):
    """Catch an accidental fallback to the old jab or an altered sidecar."""
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() == APPROVED_NPZ_SHA256
    assert hashlib.sha256(CANDIDATE.with_suffix('.json').read_bytes()).hexdigest() == APPROVED_JSON_SHA256
    cfg = make_env_cfg(reference_path=CANDIDATE, num_envs=1, easy_ttc_range=EASY_TTC, **TIMING)
    command = cfg.commands['punch']
    assert Path(command.reference_path).resolve() == CANDIDATE.resolve()
    assert selected_reference_path('jab') == ORIGINAL  # Per-run selection, no promotion.
    assert command.clock_mode == candidate.clock_mode == 'smooth_entry'
    assert command.ttc_anchor_time == candidate.ttc_anchor_time == pytest.approx(.510)
    assert command.post_entry_duration == candidate.recovery == pytest.approx(.910)
    assert tuple(command.easy_ttc_range) == pytest.approx(EASY_TTC)
    assert candidate.strike_time == pytest.approx(.680)  # Still the center annotation.
    assert candidate.metadata['nominal_first_entry_s'] == pytest.approx(.510)


def test_experiment_contract_roundtrip_preserves_collection_and_evaluation_selection():
    from human_punching.task10c.evaluate import (
        reference_env_kwargs, resolve_reference_contract, verify_reference_contract,
    )

    experiment = ROOT / 'configs/task10c/jab_contact_reference_20261022.json'
    contract = resolve_reference_contract(experiment_config=experiment)
    checkpoint_infos = {'reference_contract': contract,
                        'reference_sha256': APPROVED_NPZ_SHA256,
                        'reference_metadata_sha256': APPROVED_JSON_SHA256}
    restored = resolve_reference_contract(checkpoint_infos=checkpoint_infos)
    assert restored == contract
    for selected in (contract, restored):
        cfg = make_env_cfg(num_envs=1, **reference_env_kwargs(selected))
        command = cfg.commands['punch']
        assert Path(command.reference_path).resolve() == CANDIDATE.resolve()
        assert command.clock_mode == 'smooth_entry'
        assert command.ttc_anchor_time == pytest.approx(.510)
        assert command.post_entry_duration == pytest.approx(.910)
        assert command.ttc_range == (.5, .9)
        assert command.easy_ttc_range == pytest.approx(EASY_TTC)
    for change in ({'clock_mode': 'legacy'}, {'ttc_anchor_time': .680},
                   {'easy_ttc_range': [.612, .748]}, {'target_height_curriculum': True},
                   {'reference_metadata_sha256': '0' * 64},
                   {'reference_path': str(ORIGINAL.relative_to(ROOT))}):
        with pytest.raises(ValueError):
            verify_reference_contract({**contract, **change})
    # A historical checkpoint cannot silently become a corrected-reference run.
    with pytest.raises(ValueError, match='fresh lineage'):
        resolve_reference_contract(experiment_config=experiment,
                                   checkpoint_infos={'reference_sha256': 'legacy'})
    historical = resolve_reference_contract()
    assert historical['clock_mode'] == 'legacy'
    assert ROOT / historical['reference_path'] == ORIGINAL


def test_torch_reference_loads_candidate_full_body_arrays_instead_of_original(candidate):
    with np.load(CANDIDATE, allow_pickle=False) as saved, np.load(ORIGINAL, allow_pickle=False) as old:
        for key in ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w',
                    'glove_pos_w', 'glove_quat_w', 'joint_pos', 'joint_vel'):
            torch.testing.assert_close(candidate.arrays[key], torch.tensor(saved[key], dtype=torch.float32))
        frame = int(np.argmin(abs(saved['time_s'] - .680)))
        wrist = tuple(saved['body_names']).index('left_wrist_yaw_link')
        assert np.linalg.norm(saved['body_pos_w'][frame, wrist] - old['body_pos_w'][frame, wrist]) > .20
        assert not np.allclose(saved['body_quat_w'][frame, wrist], old['body_quat_w'][frame, wrist])
        # Source lower-body/head data stays available in the chosen full-body file.
        for name in ('pelvis', 'torso_link', 'head_pitch_link', 'head_yaw_link',
                     'left_knee_link', 'right_ankle_pitch_link'):
            index = tuple(saved['body_names']).index(name)
            np.testing.assert_array_equal(saved['body_pos_w'][:, index], old['body_pos_w'][:, index])


def test_selected_reference_scales_every_velocity_array_and_zeroes_hold(candidate):
    query = torch.tensor([.510, .680, .940, 1.420])
    rates = torch.tensor([.75, 1.25, 1.6, 0.])
    sample = candidate.sample(query, rates)
    with np.load(CANDIDATE, allow_pickle=False) as saved:
        velocity_keys = {key for key in saved.files if 'vel' in key}
        assert velocity_keys == {'qvel', 'joint_vel', 'body_lin_vel_w', 'body_ang_vel_w',
                                 'glove_lin_vel_w', 'glove_ang_vel_w'}
        for key in velocity_keys:
            flat = saved[key].reshape(len(saved['time_s']), -1)
            interpolated = np.stack([np.interp(query.numpy(), saved['time_s'], flat[:, j])
                                     for j in range(flat.shape[1])], axis=-1)
            expected = interpolated.reshape(sample[key].shape) * rates.numpy().reshape(
                (-1,) + (1,) * (sample[key].ndim - 1))
            np.testing.assert_allclose(sample[key].numpy(), expected, atol=2e-5, rtol=2e-5)
            assert bool((sample[key][-1] == 0).all())


def test_easy_timing_is_explicit_and_not_rederived_from_changed_center_annotation(candidate):
    points = np.array([[0., 0., 1.22], [.1, 0., 1.23], [-.1, 0., 1.21], [0., .1, 1.20]])
    command = SimpleNamespace(reference=candidate, device='cpu',
        fixed_target_index=None,
        cfg=SimpleNamespace(target_split='train', family='jab', easy_ttc_range=EASY_TTC),
        pool=SimpleNamespace(points=lambda *_: points.copy()))
    PunchCommand.set_curriculum(command, {'stage': 0, 'target_fraction': 1.,
                                         'ttc_range': [.5, .9], 'easy_fraction': .5})
    assert command.initial_ttc_range == pytest.approx(EASY_TTC)
    assert command.initial_ttc_range != pytest.approx((.9 * candidate.strike_time, 1.1 * candidate.strike_time))
    assert command.cfg.ttc_range == (.5, .9)


def _reset_harness(reference, recovery_fraction):
    """Real reset/clock/reference logic, replacing only simulation writes."""
    n = 8
    zeros = lambda *shape: torch.zeros(*shape)
    captured = {}

    def write_root(value, ids):
        captured['root'] = value.clone()

    def write_joints(position, velocity, *, env_ids):
        captured['joint_pos'], captured['joint_vel'] = position.clone(), velocity.clone()

    action = SimpleNamespace(target_ids=list(range(31)), offset=zeros(n, 31),
                             scale=torch.ones(n, 31), _raw_actions=zeros(n, 31))
    env = SimpleNamespace(step_dt=.02, scene=SimpleNamespace(env_origins=zeros(n, 3)),
        action_manager=SimpleNamespace(get_term=lambda _: action, action=zeros(n, 31),
                                       prev_action=zeros(n, 31), prev_prev_action=zeros(n, 31)))
    robot = SimpleNamespace(write_root_state_to_sim=write_root, write_joint_state_to_sim=write_joints,
                            set_joint_position_target=lambda *_, **__: None)
    command = SimpleNamespace(device='cpu', _env=env, robot=robot,
        reference_origin_w=zeros(n, 3),
        root_index=reference.body_names.index('pelvis'), reference=reference,
        cfg=SimpleNamespace(target_split='train', family='jab', recovery_fraction=recovery_fraction,
                            easy_ttc_range=EASY_TTC, **TIMING),
        targets=torch.tensor([[1.061, .260, 1.228]]), target_candidates=torch.tensor([0]),
        easy_target_candidates=torch.tensor([0]), target_index=torch.zeros(n, dtype=torch.long),
        state=PunchState(n, 'cpu'), horizon=torch.ones(n), interaction_z=torch.ones(n), guard_z=torch.ones(n),
        pending_launch=torch.zeros(n, dtype=torch.bool), ready=torch.zeros(n, dtype=torch.bool),
        steps_seen=torch.zeros(n, dtype=torch.long), settled_duration=zeros(n),
        first_settled_elapsed=torch.full((n,), float('nan')), component_totals={},
        reset_counts={'full': 0, 'snippet': 0}, evaluation_recipes=None,
        initial_ttc_range=EASY_TTC)
    command.cfg.ttc_range, command.cfg.easy_fraction = (.6, .6), 0.
    command.refresh_reference = MethodType(PunchCommand.refresh_reference, command)
    return command, captured


@pytest.mark.parametrize('recovery_fraction', [0., 1.])
def test_guard_and_snippet_resets_write_same_selected_reference(candidate, recovery_fraction):
    command, captured = _reset_harness(candidate, recovery_fraction)
    with torch.random.fork_rng():
        torch.manual_seed(102224)
        PunchCommand._resample_command(command, torch.arange(8))
    expected = candidate.sample(command.source_time, command.source_rate)
    torch.testing.assert_close(captured['joint_pos'], expected['joint_pos'])
    torch.testing.assert_close(captured['root'][:, :3], expected['root_pos_w'])
    torch.testing.assert_close(captured['root'][:, 3:7], expected['root_quat_w'])
    torch.testing.assert_close(command.horizon, torch.full((8,), 1.92))
    if recovery_fraction == 0:
        assert command.reset_counts == {'full': 8, 'snippet': 0}
        assert bool((command.source_time == 0).all())
        assert bool((captured['joint_vel'] == 0).all())
        assert bool((captured['root'][:, 7:] == 0).all())
    else:
        assert command.reset_counts == {'full': 0, 'snippet': 8}
        assert not bool(command.state.history_known.any())
        assert bool((command.source_time > candidate.ttc_anchor_time).all())
        torch.testing.assert_close(captured['joint_vel'], expected['joint_vel'])
        torch.testing.assert_close(captured['root'][:, 7:10], expected['body_lin_vel_w'][:, command.root_index])
        torch.testing.assert_close(captured['root'][:, 10:13], expected['body_ang_vel_w'][:, command.root_index])
        old = TorchReference(ORIGINAL, 'cpu', JOINT_NAMES, family='jab')
        old_poses = old.sample(command.source_time, command.source_rate)['joint_pos']
        assert float((captured['joint_pos'] - old_poses).abs().max()) > .1
    # The same refresh path drives rewards after reset and zeros every hold velocity.
    command.state.start_elapsed[:] = .6 + candidate.recovery + .05
    command.refresh_reference()
    torch.testing.assert_close(command.source_time, torch.full((8,), candidate.end_time))
    assert bool((command.source_rate == 0).all())
    for key, value in command.ref.items():
        if 'vel' in key:
            assert bool((value == 0).all())
