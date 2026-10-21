"""Pure CPU tensor reducers: no simulator imports, stepping, GPU, or learning."""
import math

import pytest
import torch

from human_punching.wbc.telemetry.event_state import FAILURE_CATEGORIES, FLAGS, PunchEventState, Thresholds, angle


from task10_telemetry_helpers import state


def make(n=2, dtype=torch.float64, **kw):
    s = state(n, dtype=dtype)
    reducer = PunchEventState(n, torch.tensor([[-1., 1.]]*3, dtype=dtype), torch.full((3,), 10., dtype=dtype), **kw)
    command = {"target_w": torch.tensor([[0., 0., 1.]]*n, dtype=dtype),
        "active_hand": torch.zeros(n, dtype=torch.int64), "ttc_s": torch.full((n,), .005, dtype=dtype),
        "outward_end_s": torch.full((n,), .010, dtype=dtype)}
    reducer.launch(list(range(n)), s, **command)
    return reducer, s, command


def step(r, s, **changes):
    changed = {k: v.clone() for k, v in s.items()}
    for key, value in changes.items():
        changed[key] = value
    r.step(r.tick+1, changed)


def advance(r, s, ticks):
    for _ in range(ticks):
        if r.window_count in (None, 4):
            r.begin_window()
        step(r, s)


def hit(s, hand=0, distance=.14):
    s = {k: v.clone() for k, v in s.items()}
    s["glove_position_w"][:, hand, 0] = distance
    return s


def test_contact_inclusive_first_sample_and_latched_bonus_only_once():
    r, s, _ = make(); r.begin_window()
    step(r, hit(s)); step(r, hit(s, distance=.13)); step(r, s); step(r, s)
    report = r.window_summary()
    assert r.first_contact.tolist() == [1, 1]
    assert r.withdrawal.tolist() == [3, 3]
    assert report["contact_bonus_rate"].tolist() == [50., 50.]
    assert torch.equal(report["contact_bonus_rate"]*.020, report["new_contact"].double())
    advance(r, s, 4)
    assert not r.window_summary()["new_contact"].any()
    assert r.terminal_summary()["ttc_error_s"].tolist() == [0., 0.]


def test_contact_after_outward_end_does_not_count():
    r, s, _ = make(); r.begin_window(); step(r, s); step(r, s); step(r, hit(s)); step(r, hit(s))
    assert r.first_contact.tolist() == [-1, -1]
    assert not r.terminal_summary()["has_contact"].any()


def test_hands_are_per_environment_and_target_is_copied():
    r, s, commands = make()
    commands["target_w"].fill_(100.)
    assert torch.equal(r.target, torch.tensor([[0., 0., 1.]]*2, dtype=r.dtype))
    r.hand[1] = 1  # a second predeclared family, independent batch indexing
    h = hit(s); h["glove_position_w"][1, 0, 0] = .3; h["glove_position_w"][1, 1, 0] = .14
    r.begin_window(); step(r, h)
    assert r.first_contact.tolist() == [1, 1]
    assert not r.flags["inactive_contact"].any()


def test_clearance_contact_and_renewed_contact_are_distinct():
    r, s, _ = make(); r.begin_window(); step(r, hit(s))
    clear = hit(s, distance=.16); step(r, clear)
    assert r.withdrawal.tolist() == [2, 2]
    h = hit(s, hand=1, distance=.15); step(r, h)
    assert r.flags["inactive_clearance"].all() and not r.flags["inactive_contact"].any()
    step(r, hit(s, hand=1, distance=.14))
    assert r.flags["inactive_contact"].all() and r.flags["renewed_contact"].all()


def test_four_tick_extrema_survive_a_safe_final_pose():
    r, s, _ = make(); r.begin_window()
    bad = {k: v.clone() for k, v in s.items()}; bad["joint_position"][:, 0] = 1.03; bad["effort"][:, 1] = 10.5
    step(r, s); step(r, bad); step(r, s); step(r, s)
    report = r.window_summary()
    assert torch.allclose(report["joint_peak"], torch.full((2,), .03, dtype=r.dtype))
    assert report["effort_peak"].tolist() == [.5, .5]
    assert report["effort_abs_peak_nm"][:, 1].tolist() == [10.5, 10.5]
    assert r.flags["joint_limit"].all() and r.flags["effort_limit"].all()
    advance(r, s, 4)
    assert not r.window_summary()["joint_peak"].any()
    assert torch.allclose(r.terminal_summary()["joint_peak"], torch.full((2,), .03, dtype=r.dtype))
    assert r.terminal_summary()["first_flag_tick"]["joint_limit"].tolist() == [2, 2]
    report = r.terminal_summary()
    assert report["first_failure_tick"].tolist() == [2, 2]
    assert report["first_failure_category"].tolist() == [FAILURE_CATEGORIES.index("joint_limit")]*2
    assert report["first_failure_mask"].tolist() == [
        (1 << FAILURE_CATEGORIES.index("joint_limit")) | (1 << FAILURE_CATEGORIES.index("effort_limit"))]*2


def test_saturation_seconds_use_actual_capped_force_and_all_ticks():
    r, s, _ = make(); s["effort"][:, 0] = 10.; advance(r, s, 8)
    assert torch.allclose(r.saturation_seconds[:, 0], torch.full((2,), .04, dtype=r.dtype))
    assert not r.flags["effort_limit"].any()


def test_slip_uses_preceding_interval_velocity_and_patch_load():
    r, s, _ = make(); r.begin_window()
    first = {k: v.clone() for k, v in s.items()}
    first["support_anchor_velocity_w"][:, 0, 0] = 2.
    step(r, first)  # previous velocity was zero
    second = {k: v.clone() for k, v in first.items()}; second["support_anchor_position_w"][:, 0, 0] += .01
    step(r, second)  # loaded first foot: 2 m/s at interval start
    third = {k: v.clone() for k, v in second.items()}; third["support_patch_contact"][:, 0] = False
    third["support_anchor_position_w"][:, 0, 0] += .01
    step(r, third); step(r, third)
    assert torch.allclose(r.slip_distance[:, 0], torch.full((2,), .01, dtype=r.dtype))
    assert r.window_summary()["loaded_slip_speed_rate"].tolist() == [.5, .5]
    assert r.episode_loaded_drift_peak.tolist() == pytest.approx([.01, .01])
    assert r.episode_footprint_peak.tolist() == pytest.approx([.02, .02])


def test_unloaded_anchor_motion_is_not_slip_but_steps_are_retained():
    r, s, _ = make(); s["support_patch_contact"][:] = False
    s["support_anchor_position_w"][:, 0, 0] += .04
    advance(r, s, 4)
    assert not r.slip_distance.any() and not r.episode_loaded_drift_peak.any()
    assert r.flags["step_or_drift"].all()


def test_unloaded_heel_height_does_not_imply_planar_anchor_slip():
    r, s, _ = make(); s["support_anchor_position_w"][:, 0, 2] += .1
    s["support_patch_contact"][:, 0] = False
    advance(r, s, 4)
    assert not r.flags["step_or_drift"].any() and not r.slip_distance.any()


def test_strict_recovery_starts_after_contact_withdrawal_and_outward_end():
    r, s, _ = make(); r.begin_window(); step(r, hit(s)); step(r, s)
    assert r.recovery_start.tolist() == [2, 2]
    advance(r, s, 199)
    assert r.tick.tolist() == [201, 201] and r.recovery_tick.tolist() == [-1, -1]
    step(r, s)
    assert r.recovery_tick.tolist() == [202, 202]
    advance(r, s, 2)
    assert r.window_summary()["new_recovery"].all()
    advance(r, s, 4)
    assert not r.window_summary()["new_recovery"].any()


def test_quiet_guard_without_punch_is_not_recovery():
    r, s, _ = make(); advance(r, s, 204)
    assert r.guard_intervals.tolist() == [204, 204]
    assert r.recovery_tick.tolist() == [-1, -1]


def test_loss_of_support_breaks_dwell_without_resetting_physics_clock():
    r, s, _ = make(); r.begin_window(); step(r, hit(s)); advance(r, s, 100)
    unsupported = {k: v.clone() for k, v in s.items()}; unsupported["support_contact"][:, 0] = False
    step(r, unsupported)
    assert not r.guard_intervals.any()
    advance(r, s, 199)
    assert r.tick.tolist() == [301, 301] and r.recovery_tick.tolist() == [-1, -1]
    step(r, s)
    assert r.recovery_tick.tolist() == [-1, -1]
    step(r, s)
    assert r.recovery_tick.tolist() == [303, 303]


def test_single_support_is_allowed_but_complete_support_loss_is_failure():
    r, s, _ = make(); r.begin_window()
    one = {k: v.clone() for k, v in s.items()}; one["support_contact"][:, 0] = False
    step(r, one)
    assert not r.flags["fall"].any()
    none = {k: v.clone() for k, v in one.items()}; none["support_contact"][:, 1] = False
    step(r, none); step(r, s); step(r, s)
    assert r.flags["fall"].all()
    assert r.terminal_summary()["first_flag_tick"]["fall"].tolist() == [2, 2]


@pytest.mark.parametrize("field,value", [("joint_velocity", .51), ("glove_velocity_guard_frame", .16),
    ("root_linear_velocity_w", .051), ("root_angular_velocity_w", .121)])
def test_strict_quiet_velocity_breaks_dwell(field, value):
    r, s, _ = make(); advance(r, s, 4)
    noisy = {k: v.clone() for k, v in s.items()}; noisy[field].reshape(2, -1)[:, 0] = value
    r.begin_window(); step(r, noisy)
    assert not r.guard_intervals.any()


def test_recovery_onset_deadline_cannot_be_repaired_by_waiting():
    r, s, _ = make(); r.begin_window(); step(r, hit(s))
    restless = {k: v.clone() for k, v in s.items()}; restless["joint_velocity"][:] = 1.
    advance(r, restless, 604); advance(r, s, 204)
    assert r.recovery_start.tolist() == [606, 606]
    assert r.recovery_tick.tolist() == [-1, -1]


@pytest.mark.parametrize("fault", ["contact_overflow", "numerical_failure", "nan", "nonunit", "clock"])
def test_measurement_faults_latch_and_cannot_be_hidden_by_later_safe_tick(fault):
    r, s, _ = make(); r.begin_window(); bad = hit(s)
    tick = r.tick+1
    if fault in ("contact_overflow", "numerical_failure"): bad[fault][:] = True
    elif fault == "nan": bad["root_position_w"][:, 0] = torch.nan
    elif fault == "nonunit": bad["root_quat_wxyz"][:, 0] = 2.
    elif fault == "clock": tick += 1
    r.step(tick, bad); step(r, hit(s)); step(r, s); step(r, s)
    report = r.window_summary()
    assert not report["measurement_valid"].any()
    assert not report["contact_bonus_rate"].any()
    assert r.first_contact.tolist() == [-1, -1]
    advance(r, s, 4)
    assert not r.window_summary()["measurement_valid"].any()


@pytest.mark.parametrize("flag", ["external_support", "forbidden_collision", "human_g1_collision"])
def test_physical_flags_preserved_after_the_contact_interval(flag):
    r, s, _ = make(); r.begin_window(); bad = {k: v.clone() for k, v in s.items()}; bad[flag][:] = True
    step(r, bad); advance(r, s, 3)
    assert r.terminal_summary()["flags"][flag].all()


def test_partial_reset_requires_logging_and_preserves_other_episodes():
    r, s, commands = make(); r.begin_window(); step(r, hit(s)); advance(r, s, 3)
    with pytest.raises(ValueError, match="Preserve terminal"):
        r.reset_after_log([0])
    saved = r.terminal_summary([0]); r.reset_after_log([0]); r.launch([0], s, **commands)
    assert saved["first_contact_tick"].tolist() == [1]
    assert r.first_contact.tolist() == [-1, 1] and r.tick.tolist() == [0, 4]
    advance(r, s, 4)
    assert r.tick.tolist() == [4, 8] and saved["first_contact_tick"].tolist() == [1]


def test_window_and_terminal_reads_do_not_consume_or_alias():
    r, s, _ = make(); r.begin_window(); step(r, hit(s)); advance(r, s, 3)
    a = r.window_summary(); b = r.window_summary(); a["new_contact"][:] = False
    assert b["new_contact"].all() and r.window_summary()["new_contact"].all()
    t = r.terminal_summary(); t["first_contact_tick"][:] = 99
    assert r.first_contact.tolist() == [1, 1]


def test_clock_window_shape_and_launch_guards_fail_closed():
    r, s, _ = make()
    with pytest.raises(ValueError, match="begin_window"): step(r, s)
    r.begin_window(); step(r, s)
    with pytest.raises(ValueError, match="exactly four"): r.begin_window()
    with pytest.raises(ValueError, match="Incomplete"): r.window_summary()
    malformed = {**s, "joint_position": s["joint_position"].float()}
    with pytest.raises(ValueError, match="Malformed joint_position"): step(r, malformed)
    with pytest.raises(ValueError, match="Unique"): r.reset_after_log([0, 0])


@pytest.mark.parametrize("condition", ["joint", "effort", "speed", "tilt", "clearance", "support"])
def test_launch_rejects_nonadmissible_initial_state(condition):
    s = state(); r = PunchEventState(2, torch.tensor([[-1., 1.]]*3, dtype=torch.float64), torch.full((3,), 10., dtype=torch.float64))
    if condition == "joint": s["joint_position"][:, 0] = 1.001
    elif condition == "effort": s["effort"][:, 0] = 11.
    elif condition == "speed": s["joint_velocity"][:, 0] = .6
    elif condition == "tilt": s["root_quat_wxyz"][:, 0] = math.cos(.4); s["root_quat_wxyz"][:, 1] = math.sin(.4)
    elif condition == "clearance": s["glove_position_w"][:, 0, 0] = .159
    elif condition == "support": s["support_contact"][:, 0] = False
    _, _, command = make()
    before = {name: getattr(r, name).clone() for name in ("target", "hand", "ttc", "outward_end", "armed", "tick")}
    goals = {name: value.clone() for name, value in r.goal.items()}
    with pytest.raises(ValueError, match="Launch requires"): r.launch([0, 1], s, **command)
    assert not r.armed.any()
    assert all(torch.equal(value, getattr(r, name)) for name, value in before.items())
    assert all(torch.equal(value, r.goal[name]) for name, value in goals.items())
    assert r._armed_count == 0


def test_identifiers_cannot_silently_truncate_floats():
    r, _, _ = make()
    with pytest.raises(ValueError, match="Integer environment IDs"):
        r.terminal_summary([.5])


def test_hot_step_performs_no_host_tensor_reads(monkeypatch):
    r, s, _ = make()
    def forbidden(*a, **k): raise AssertionError("host tensor read")
    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__"):
            m.setattr(torch.Tensor, name, forbidden)
        r.begin_window()
        step(r, s)
    assert r.tick.tolist() == [1, 1]


def test_float32_and_float64_agree_on_tick_events():
    summaries = []
    for dtype in (torch.float32, torch.float64):
        r, s, _ = make(dtype=dtype); r.begin_window(); step(r, hit(s, distance=.139)); advance(r, s, 203)
        summaries.append(r.terminal_summary())
    for key in ("first_contact_tick", "withdrawal_tick", "recovery_tick"):
        assert torch.equal(summaries[0][key], summaries[1][key])


def test_float32_quaternion_self_and_sign_alias_angle_are_exact_zero():
    q = torch.tensor([[.312341, .199, -.435, .765]], dtype=torch.float32)
    q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    assert angle(q, q).item() == 0.
    assert angle(q, -q).item() == 0.
    identity = torch.tensor([[1., 0., 0., 0.]], dtype=q.dtype)
    quarter_turn = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]], dtype=q.dtype)
    assert angle(identity, quarter_turn).item() == pytest.approx(math.pi/2, abs=1e-6)


def unarmed():
    _, s, command = make()
    r = PunchEventState(2, torch.tensor([[-1., 1.]]*3, dtype=s["joint_position"].dtype),
                        torch.full((3,), 10., dtype=s["joint_position"].dtype))
    return r, s, command


def test_asynchronous_launch_excludes_settling_rows_from_every_cycle_event():
    r, s, command = unarmed(); r.launch([0], s, **command)
    r.begin_window()
    for k in range(1, 5):
        observation = hit(s)
        observation["effort"][1] = 12.
        observation["joint_position"][1, 0] = 1.05
        observation["external_support"][1] = True
        r.step(torch.tensor([k, 0]), observation)
    first = r.window_summary()
    assert r.first_contact.tolist() == [1, -1] and r.tick.tolist() == [4, 0]
    assert first["cycle_active"].tolist() == [True, False]
    assert first["measurement_valid"].tolist() == [True, False]
    assert first["diagnostic_measurement_valid"].tolist() == [True, True]
    assert not r.episode_joint_peak[1] and not r.episode_effort_abs_peak[1].any()
    assert not any(flag[1].item() for flag in r.flags.values())
    r.launch([1], s, **command)
    r.begin_window()
    for k in range(1, 5):
        observation = {key: value.clone() for key, value in s.items()}
        observation["glove_position_w"][1, 0, 0] = .14 if k == 1 else .3
        r.step(torch.tensor([4+k, k]), observation)
    assert r.first_contact.tolist() == [1, 1]
    assert r.window_summary()["new_contact"].tolist() == [False, True]
    assert r.tick.tolist() == [8, 4]
    assert r.terminal_summary()["ttc_error_s"].tolist() == [0., 0.]


def test_unarmed_diagnostics_do_not_invent_cycle_validity_or_a_failure_tick():
    r, s, _ = unarmed(); r.begin_window()
    bad = hit(s); bad["root_position_w"][0, 0] = torch.nan; bad["contact_overflow"][1] = True
    for _ in range(4): r.step(torch.zeros(2, dtype=torch.int64), bad)
    report = r.terminal_receipt()
    assert report["cycle_armed"].tolist() == [False, False]
    assert not report["measurement_valid"].any() and not report["diagnostic_measurement_valid"].any()
    assert report["first_contact_tick"].tolist() == [-1, -1]
    assert report["first_failure_tick"].tolist() == [-1, -1]
    assert not report["has_contact"].any()
    r.reset_after_log([0, 1])
    assert r._armed_count == 0 and not r.armed.any()
    assert r.terminal_summary()["diagnostic_measurement_valid"].all()


def test_finished_rows_freeze_events_extrema_and_clock_while_other_rows_continue():
    r, s, _ = make(); r.begin_window(); step(r, hit(s)); advance(r, s, 3)
    before = r.terminal_receipt([0])
    r.begin_window()
    for _ in range(4):
        bad = hit(s, hand=1); bad["effort"][0] = 12.; bad["root_position_w"][0, 0] = torch.nan
        r.step(r.tick+1, bad)
    after = r.terminal_summary([0])
    for name in ("first_contact_tick", "withdrawal_tick", "recovery_tick", "joint_peak", "effort_peak", "saturation_s", "first_failure_tick"):
        assert torch.equal(before[name], after[name]), name
    assert r.tick.tolist() == [4, 8]
    assert r.window_summary()["cycle_active"].tolist() == [False, True]
    assert not r.window_summary()["new_contact"][0]
    assert after["measurement_valid"].item() and not after["diagnostic_measurement_valid"].item()
    assert r.flags["inactive_contact"].tolist() == [False, True]


def test_reset_never_launched_receipt_does_not_erase_another_active_cycle():
    r, s, command = unarmed(); r.launch([0], s, **command); advance(r, s, 4)
    r.terminal_receipt([1]); r.reset_after_log([1])
    assert r.active.tolist() == [True, False] and r.tick.tolist() == [4, 0]
    assert r._armed_count == 1
    r.launch([1], s, **command)
    assert r._armed_count == 2
    advance(r, s, 4)
    assert r.tick.tolist() == [8, 4]


def test_nominal_posture_goal_and_actual_ready_footprint_have_separate_identity():
    r, nominal, command = unarmed()
    actual = {name: value.clone() for name, value in nominal.items()}
    actual["root_position_w"][:, 0] = .01
    actual["joint_position"][:, 0] = .01
    actual["glove_position_guard_frame"][:, :, 0] += .01
    actual["support_anchor_position_w"][:, :, 0] += .02
    r.launch([0, 1], actual, nominal_guard=nominal, **command)
    assert torch.equal(r.goal["root_position_w"], nominal["root_position_w"])
    assert torch.equal(r.goal["joint_position"], nominal["joint_position"])
    assert torch.equal(r.goal["support_anchor_position_w"], actual["support_anchor_position_w"])
    assert torch.equal(r.initial_guard["root_position_w"], actual["root_position_w"])
    returned = {name: value.clone() for name, value in nominal.items()}
    returned["support_anchor_position_w"] = actual["support_anchor_position_w"].clone()
    r.begin_window(); step(r, hit(actual)); advance(r, returned, 203)
    assert r.recovery_tick.tolist() == [202, 202]
    actual["root_position_w"][:] = 100.; nominal["joint_position"][:] = 100.
    assert r.initial_guard["root_position_w"][:, 0].tolist() == [.01, .01]
    assert not r.goal["joint_position"].any()


def test_nominal_guard_launch_rejection_is_atomic_and_cannot_bless_current_pose():
    r, nominal, command = unarmed(); actual = {name: value.clone() for name, value in nominal.items()}
    actual["root_position_w"][:, 0] = .04
    with pytest.raises(ValueError, match="Launch requires"):
        r.launch([0, 1], actual, nominal_guard=nominal, **command)
    assert not r.armed.any() and not r.goal["root_position_w"].any()
    assert not r.initial_guard["root_position_w"].any()


def test_finish_cycles_is_boundary_only_and_cannot_reactivate_a_terminal_row():
    r, s, _ = make(); r.begin_window(); step(r, s)
    with pytest.raises(ValueError, match="four-tick boundary"):
        r.finish_cycles(torch.tensor([True, False]))
    advance(r, s, 3); r.finish_cycles(torch.tensor([True, False]))
    r.finish_cycles(torch.zeros(2, dtype=torch.bool))
    assert r.active.tolist() == [False, True]
    advance(r, s, 4)
    assert r.tick.tolist() == [4, 8]


def test_async_masking_callbacks_keep_no_host_read_contract(monkeypatch):
    r, s, command = unarmed(); r.launch([0], s, **command)
    def forbidden(*args, **kwargs): raise AssertionError("host read in masked event callbacks")
    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__"):
            m.setattr(torch.Tensor, name, forbidden)
        r.begin_window()
        for k in range(1, 5): r.step(torch.tensor([k, 0]), hit(s))
        r.finish_cycles(torch.tensor([True, False]))
    assert r.first_contact.tolist() == [1, -1] and not r.active.any()
