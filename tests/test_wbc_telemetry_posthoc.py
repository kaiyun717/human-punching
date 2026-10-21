"""Independent recorded-data audit tests; no native integration or GPU work."""
from types import SimpleNamespace
import numpy as np
import pytest
from human_punching.wbc.telemetry import posthoc_qualification as p


def test_rotation_error_identifies_double_cover_and_real_rotation():
    a = np.array([[1., 0, 0, 0], [.5, .5, .5, .5]])
    assert np.max(p.rotation_error_rad(a, -a)) == 0
    assert np.max(p.rotation_error_rad(a, a*1.000001)) < 1e-15
    angle = .01
    b = np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.])
    assert p.rotation_error_rad(a[0], b) == pytest.approx(angle)
    with pytest.raises(ValueError, match='Zero quaternion'):
        p.rotation_error_rad(a[0], np.zeros(4))


def test_boundary_is_measured_per_contact_and_does_not_change_radius():
    interior = p.patch_boundary_record([[.064, 0, 0]], [0, 0, 0], [0, 0, 0])
    exterior = p.patch_boundary_record([[.066, 0, 0]], [0, 0, 0], [0, 0, 0])
    assert interior['classification'] == 'interior'
    assert exterior['classification'] == 'exterior'
    # Only a tiny measured anchor discrepancy straddles the declared boundary.
    boundary = p.patch_boundary_record([[.065, 0, 0]], [-1e-8, 0, 0], [1e-8, 0, 0])
    assert boundary['boundary_uncertain']
    assert boundary['cpu_distance_minus_radius_m'] > 0
    assert boundary['gpu_anchor_distance_minus_radius_m'] < 0
    assert boundary['boundary_uncertainty_bound_m'] < 1e-7
    assert boundary['native_contact_position_difference_m'] == 0


def synthetic_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'audit_probe_arrays', lambda *a, **kw: {'status': 'failed',
        'checks': {'zero_learned_residual': False, 'unchanged_native_check': True}})
    samples, worlds = 9, 2
    q = np.zeros((samples, 55), np.float32); q[:, 7:] = np.arange(samples)[:, None]*.001
    v = np.zeros((samples, 54), np.float32); v[:, 6:] = .01
    ff = np.full((samples, 48), .2, np.float32)
    master = tmp_path/'master.npz'; np.savez(master, qpos=q, qvel=v, joint_feedforward=ff)
    body = {'kp': [10.]*48, 'kd': [1.]*48, 'effort': [10.]*48, 'joint_limits_rad': [[-2., 2.]]*48}
    bundle = SimpleNamespace(body=body, clips=[{'master_path': 'master.npz'}], path=lambda _: master)
    tick = np.maximum(np.arange(samples)-1, 0)
    phase = np.arange(48, dtype=np.float32)[None]*np.float32(.17)+(tick//4)[:, None].astype(np.float32)*np.float32(.11)
    action = np.broadcast_to((np.float32(.02)*np.sin(phase))[:, None], (samples, worlds, 48)).copy(); action[0] = 0
    target = q[tick, None, 7:]+.25*action
    velocity = np.broadcast_to(v[tick, None, 6:], target.shape).copy()
    feedforward = np.broadcast_to(ff[tick, None], target.shape).copy()
    drive = 10.*target+velocity+feedforward
    native = dict(raw_action=action, trajectory_id=np.zeros((samples, worlds), int),
        joint_position_target=target, joint_velocity_target=velocity, joint_effort_target=feedforward,
        ctrl=drive, actuator_force=drive.copy(), qpos=np.zeros((samples, worlds, 55)),
        qvel=np.zeros((samples, worlds, 54)))
    return native, bundle


def test_nonzero_residual_controller_checks_reference_clock_and_force(tmp_path, monkeypatch):
    native, bundle = synthetic_controller(tmp_path, monkeypatch)
    result = p.command_drive_audit(native, bundle, 'punch')
    assert result['status'] == 'passed'
    assert result['original_zero_action_premise'] is False
    assert result['checks']['unchanged_native_check']
    native['actuator_force'][5, 0, 0] += .1
    assert not p.command_drive_audit(native, bundle, 'punch')['checks']['preceding_interval_capped_force']
    native['actuator_force'][5, 0, 0] -= .1
    native['joint_position_target'][2] += .001
    assert not p.command_drive_audit(native, bundle, 'punch')['checks']['reference_and_residual_target']


def test_nonzero_residual_audit_rejects_uncommanded_and_unheld_action(tmp_path, monkeypatch):
    native, bundle = synthetic_controller(tmp_path, monkeypatch)
    native['raw_action'][2, 0, 0] += .001
    checks = p.command_drive_audit(native, bundle, 'punch')['checks']
    assert not checks['recorded_nonzero_command']
    assert not checks['residual_held_for_four_integrations']
