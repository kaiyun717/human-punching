"""Distinct Phase3C event semantics and truthful delayed-frame attribution."""
import numpy as np

from human_punching.task10d.standing_replay import classify_trace, summarize_delivered_vision


def trace(frozen, moving, mechanical=None):
    count = len(frozen)
    return dict(time_s=np.arange(count)*.005, source_s=np.minimum(np.arange(count)*.005, .015),
        frozen_distances=np.asarray(frozen), moving_distances=np.asarray(moving),
        mechanically_safe=np.ones(count, dtype=bool) if mechanical is None else np.asarray(mechanical))


def test_outward_active_entry_remains_distinct_from_recovery_and_inactive_hits():
    arrays = trace([[.3, .3], [.3, .14], [.3, .3], [.14, .3], [.14, .3]],
                   [[.16, .3], [.16, .3], [.159, .3], [.141, .3], [.14, .3]])
    result = classify_trace(arrays, active_index=0, outward_end_s=.01, reference_end_s=.015)
    assert result['frozen_head_active_outward_entry_s'] is None
    assert result['frozen_head_active_recovery_entry_s'] == .015
    assert result['frozen_head_inactive_entry_s'] == .005
    assert result['first_glove_clearance_violation_s'] == .01
    assert result['moving_head_first_encounter_s'] == .02
    assert not result['sampled_active_outward_aimed']
    assert not result['unaided_analytic_miss']
    assert result['recovery_complete']


def test_unaided_misses_and_mechanical_failure_are_not_filtered_or_conflated():
    arrays = trace([[.3, .3], [.14, .3], [.3, .3]], [[.3, .3]]*3)
    result = classify_trace(arrays, active_index=0, outward_end_s=.005, reference_end_s=.015)
    assert result['sampled_active_outward_aimed']
    assert result['unaided_analytic_miss']
    assert result['unaided_clearance_avoided']
    assert not result['recovery_complete']
    assert not result['safe_arrival_evaluated']
    arrays['mechanically_safe'][-1] = False
    result = classify_trace(arrays, active_index=0, outward_end_s=.005, reference_end_s=.015)
    assert result['mechanical_failure']
    assert not result['unaided_analytic_miss']
    assert result['first_safety_failure_components'] == ['mechanical']


def test_delivered_source_counts_do_not_use_current_or_fabricated_old_frames():
    rows = []
    for frame in range(5):
        rows.append(dict(time_s=frame*.02, capture_frame_id=frame,
            delivered_frame_id=max(0, frame-2), delivered_capture_timestamp_s=max(0,frame-2)*.02,
            padding=frame<2, delivered_valid_union_cells=5,
            glove_rgb_valid_source_pixels=[2, 2] if frame==0 else [0, 0],
            glove_acquired_min_cells=[1, 1] if frame==0 else [0, 0]))
    result = summarize_delivered_vision(rows, first_hazard_s=.07, end_s=.08)
    assert result['acquisition_count'] == 4
    assert result['padded_delivery_count'] == 2
    assert result['nonpadding_delivery_count'] == 2
    assert result['both_represented_source_nonempty_delivery_count'] == 1
    assert result['first_both_source_nonempty_delivery_s'] == .04
    assert result['last_pre_hazard_delivery']['time_s'] == .06
    assert result['last_pre_hazard_delivery']['source_glove_rgb_min_cells'] == [0, 0]
