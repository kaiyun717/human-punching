"""Fresh96-only diagnostics from the single Phase 3C standing fixture.

The source clock is exactly one. Placement is committed before the first
integration and both gloves are observed through the complete reference and
terminal hold. Continuing after a hazard is diagnostic, never safe arrival.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .g1 import PROJECT
from .phase1 import cpu, dump, synchronized_geometry
from .phase3_render import assert_unchanged, initial_panel, state_copy
from .replay import CommittedReplay, Placement


def _first(time_s: np.ndarray, mask: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    return None if not len(indices) else float(time_s[indices[0]])


def _finite_min(values: np.ndarray) -> float | None:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return None if not len(finite) else float(finite.min())


def classify_trace(arrays: dict[str, np.ndarray], *, active_index: int,
                   outward_end_s: float, reference_end_s: float) -> dict[str, Any]:
    """Keep aimed coverage, moving encounters and mechanical failures separate."""
    t, source = np.asarray(arrays["time_s"]), np.asarray(arrays["source_s"])
    frozen, moving = arrays["frozen_distances"], arrays["moving_distances"]
    if active_index not in (0, 1) or len(t) == 0 or frozen.shape != (len(t), 2) or moving.shape != frozen.shape:
        raise ValueError("Expected two-hand distances and a nonempty physics trace")
    outward = source <= outward_end_s + 1e-12
    recovery = (source > outward_end_s + 1e-12) & (t <= reference_end_s + 1e-12)
    hold = t > reference_end_s + 1e-12
    frozen_hit, moving_hit = frozen <= .14, moving <= .14
    clearance = moving < .16
    mechanical = ~np.asarray(arrays["mechanically_safe"], dtype=bool)
    first_clearance = _first(t, clearance.any(-1))
    first_mechanical = _first(t, mechanical)
    failures = [(value, reason) for value, reason in
                ((first_clearance, "glove_clearance"), (first_mechanical, "mechanical"))
                if value is not None]
    first_failure = min((v for v, _ in failures), default=None)
    result = dict(
        frozen_head_active_outward_entry_s=_first(t, outward & frozen_hit[:, active_index]),
        frozen_head_active_recovery_entry_s=_first(t, recovery & frozen_hit[:, active_index]),
        frozen_head_active_hold_entry_s=_first(t, hold & frozen_hit[:, active_index]),
        frozen_head_inactive_entry_s=_first(t, frozen_hit[:, 1-active_index]),
        frozen_head_first_entry_by_hand_s=[_first(t, frozen_hit[:, hand]) for hand in (0, 1)],
        moving_head_first_encounter_s=_first(t, moving_hit.any(-1)),
        moving_head_first_encounter_by_hand_s=[_first(t, moving_hit[:, hand]) for hand in (0, 1)],
        moving_head_active_outward_encounter_s=_first(t, outward & moving_hit[:, active_index]),
        moving_head_active_recovery_encounter_s=_first(t, recovery & moving_hit[:, active_index]),
        moving_head_inactive_encounter_s=_first(t, moving_hit[:, 1-active_index]),
        first_glove_clearance_violation_s=first_clearance,
        first_clearance_violation_by_hand_s=[_first(t, clearance[:, hand]) for hand in (0, 1)],
        first_mechanical_safety_violation_s=first_mechanical,
        first_safety_failure_s=first_failure,
        first_safety_failure_components=[reason for value, reason in failures if value == first_failure],
        minimum_frozen_center_distances_m=[_finite_min(frozen[:, hand]) for hand in (0, 1)],
        minimum_moving_center_distances_m=[_finite_min(moving[:, hand]) for hand in (0, 1)],
        minimum_frozen_active_outward_distance_m=_finite_min(frozen[outward, active_index]),
        sampled_active_outward_aimed=bool((outward & frozen_hit[:, active_index]).any()),
        moving_head_clearance_violated=bool(clearance.any()),
        moving_head_analytic_encounter=bool(moving_hit.any()),
        mechanical_failure=bool(mechanical.any()),
        recovery_complete=bool(t[-1] + 1e-12 >= reference_end_s),
        recovery_complete_source_s=float(reference_end_s),
        first_sampled_recovery_complete_s=_first(t, t + 1e-12 >= reference_end_s),
        reference_end_s=float(reference_end_s), logged_end_s=float(t[-1]),
        completed_terminal_hold_s=max(0., float(t[-1])-reference_end_s),
        diagnostic_continuation=True, safe_arrival_evaluated=False,
        arrival_gate="No arrival is evaluated; standing at launch never ends a trial.",
    )
    # A mechanical failure is an independent explanation, not an unaided miss.
    result["unaided_analytic_miss"] = bool(result["sampled_active_outward_aimed"]
        and not result["moving_head_analytic_encounter"] and not result["mechanical_failure"])
    result["unaided_clearance_avoided"] = bool(result["sampled_active_outward_aimed"]
        and not result["moving_head_clearance_violated"] and not result["mechanical_failure"])
    return result


def summarize_delivered_vision(rows: list[dict], first_hazard_s: float | None,
                               end_s: float) -> dict[str, Any]:
    """Join delayed packets to their genuine source acquisition, retaining padding.

    Source-hand attribution is not attribution of corrupted delivered cells.
    Only the actual delivered union's nonempty-cell count is claimed downstream.
    """
    before = [row for row in rows if row["time_s"] <
              (first_hazard_s if first_hazard_s is not None else end_s + 1e-12)]
    source_lookup = {row["capture_frame_id"]: row for row in rows}
    delivered = []
    for row in before:
        source = source_lookup[row["delivered_frame_id"]]
        delivered.append(dict(time_s=row["time_s"],
            delivered_capture_timestamp_s=row["delivered_capture_timestamp_s"],
            padding=row["padding"], valid_union_cells=row["delivered_valid_union_cells"],
            source_glove_rgb_valid_pixels=source["glove_rgb_valid_source_pixels"],
            source_glove_rgb_min_cells=source["glove_acquired_min_cells"],
            source_both_gloves_represented=all(x > 0 for x in source["glove_rgb_valid_source_pixels"])
                and all(x > 0 for x in source["glove_acquired_min_cells"])))
    nonpadding = [row for row in delivered if not row["padding"]]
    informative = [row for row in nonpadding if row["valid_union_cells"] > 0]
    both = [row for row in informative if row["source_both_gloves_represented"]]
    return dict(pre_hazard_end_s=first_hazard_s, acquisition_count=len(before),
        padded_delivery_count=len(delivered)-len(nonpadding),
        nonpadding_delivery_count=len(nonpadding),
        nonempty_nonpadding_delivery_count=len(informative),
        both_represented_source_nonempty_delivery_count=len(both),
        first_nonempty_nonpadding_delivery_s=None if not informative else informative[0]["time_s"],
        first_both_source_nonempty_delivery_s=None if not both else both[0]["time_s"],
        last_pre_hazard_delivery=None if not delivered else delivered[-1],
        interpretation="Actual corrupted/delayed union cells; per-hand counts refer to its clean RGB-HSV source acquisition, not surviving corrupted cells.")


def _camera_row(exposure: dict, elapsed: float) -> dict:
    scalar = lambda key: np.asarray(exposure[key])[0].item()
    row = dict(time_s=float(elapsed), capture_frame_id=int(scalar("capture_frame_ids")),
        delivered_frame_id=int(scalar("delivered_frame_ids")),
        delivered_capture_timestamp_s=float(scalar("delivered_capture_timestamps")),
        delivery_timestamp_s=float(scalar("delivery_timestamps")),
        padding=bool(scalar("padding")), delay_ticks=int(scalar("delay_ticks")),
        delivered_valid_union_cells=int(exposure["delivered_masks"][0].sum()))
    row.update({key: exposure[key][0].tolist() for key in exposure if key.startswith("glove_")})
    return row


def run_dynamic(g1, binding, camera, primitive, fixture,
                placement: Placement | Callable[[Mapping[str, Any]], Placement],
                output: Path, label: str, *, terminal_hold_s: float = .4,
                save_media: bool = True, selection: dict | None = None) -> dict[str, Any]:
    """Restore, commit a complete guard, and execute one unit-clock diagnostic.

    ``placement`` may be a tested whole-motion transform or a selector invoked
    after restored native head/pelvis/camera geometry is read. The latter permits
    exact bank-index lookup in the actual launch frame before committing H2.
    Selection never uses a subsequent dynamic outcome.
    The caller supplies a count-one accepted combined-scene runtime and camera.
    """
    import torch
    from .media import DiagnosticMedia
    from .standing_fixture import launch_geometry

    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT.resolve()) or Path(label).name != label or label in ("", ".", ".."):
        raise ValueError("Replay outputs must be new paths inside human-punching")
    if g1.num_envs != 1 or camera.count != 1:
        raise ValueError("Representative dynamic replay uses one world")
    if not np.isfinite(terminal_hold_s) or terminal_hold_s < 0:
        raise ValueError("Terminal hold must be finite and nonnegative")
    destination = (output / label).resolve()
    if not destination.is_relative_to(output) or destination.exists():
        raise ValueError("Replay destination must be a new direct child")
    output.mkdir(parents=True, exist_ok=True)
    reset = fixture.restore(g1, camera_id=camera.ids.camera_id)
    before = state_copy(g1)
    with g1.ordered_execution(), torch.no_grad():
        # Restoration already refreshes FK. No physics or action occurs here.
        actual_launch = launch_geometry(g1, camera.ids.camera_id)
        frozen_head = actual_launch["head_world"]
        launch_head_velocity = actual_launch["head_velocity_world"]
        np.testing.assert_allclose(frozen_head, fixture.geometry["head_world"], atol=5e-6, rtol=0)
        assert float(cpu(g1.data.time)[0]) == 0.
    selector_after_restore = callable(placement)
    if selector_after_restore:
        placement = placement(actual_launch)
    if not isinstance(placement, Placement):
        raise TypeError("Placement selector must return a whole-motion Placement")
    assert_unchanged(g1, before)
    replay = CommittedReplay(primitive.reference, tuple(frozen_head), placement)
    assert replay.rate == 1. and replay.requested_ttc is None
    steps = 4 * int(np.ceil((replay.duration_s + terminal_hold_s)/.02))
    active = ("left", "right").index(primitive.active_hand)
    outward_end = float(primitive.record["annotations"]["withdrawal_start_s"])
    media = DiagnosticMedia(destination, camera.renderer.host_model, label) if save_media else None
    if media is None:
        destination.mkdir(exist_ok=False)
    trace = {key: [] for key in ("time_s", "source_s", "qpos", "qvel", "head", "head_velocity",
        "gloves", "frozen_distances", "moving_distances", "mechanically_safe", "finite", "stance_energy")}
    targets, camera_rows, visual_arrays = [], [], []
    first_hazard, first_mechanical, max_glove_error, contract = None, None, 0., None
    camera_episode_before = dict(faults=cpu(camera.faults.episode_counts).tolist(),
                                 delay=cpu(camera.delay.episode_counts).tolist())
    numerical_stop = False
    try:
        for step in range(steps+1):
            elapsed = .005 * step
            sample = replay.sample(elapsed)
            with g1.ordered_execution(), torch.no_grad():
                geometry, gloves = synchronized_geometry(g1, binding, sample)
                head = cpu(geometry["head_position"])[0]
                frozen = np.linalg.norm(gloves-frozen_head, axis=-1)
                moving = np.linalg.norm(gloves-head, axis=-1)
                mechanical = bool(cpu(geometry["mechanically_safe"])[0])
                finite = bool(cpu(geometry["finite"])[0])
                max_glove_error = max(max_glove_error, float(np.abs(gloves-sample.glove_pos).max()))
                if first_hazard is None and moving.min() < .16:
                    first_hazard = elapsed
                if first_mechanical is None and not mechanical:
                    first_mechanical = elapsed
                values = dict(time_s=elapsed, source_s=sample.source_time_s,
                    qpos=cpu(g1.data.qpos)[0], qvel=cpu(g1.data.qvel)[0], head=head,
                    head_velocity=cpu(geometry["head_velocity"])[0], gloves=gloves,
                    frozen_distances=frozen, moving_distances=moving,
                    mechanically_safe=mechanical, finite=finite,
                    stance_energy=float(cpu(geometry["stance_energy"])[0]))
                for key, value in values.items():
                    trace[key].append(value)
                if not finite:
                    numerical_stop = True
                    break
                if step % 4 == 0:
                    exposure = camera.capture(elapsed, reset=step == 0, include_source=step == 0)
                    if step == 0:
                        contract = assert_unchanged(g1, before)
                        np.testing.assert_allclose(exposure["camera_position"][0],
                            fixture.geometry["camera_position_world"], atol=5e-6, rtol=0)
                        for key in ("capture_timestamps", "delivered_capture_timestamps", "delivery_timestamps",
                                    "capture_frame_ids", "delivered_frame_ids"):
                            np.testing.assert_array_equal(exposure[key], 0)
                        np.testing.assert_array_equal(exposure["padding"], True)
                        if media is not None:
                            initial_panel(destination/"initial_scene.png", label,
                                media.renderer.render(g1.data, 0.), exposure)
                    camera_rows.append(_camera_row(exposure, elapsed))
                    visual_arrays.append({key: exposure[key][0].copy() for key in
                        ("acquired_masks", "acquired_normalized", "corrupted_masks", "corrupted_normalized",
                         "delivered_masks", "delivered_normalized")})
                    if media is not None:
                        media.append(g1.data, exposure, time_s=elapsed, source_s=sample.source_time_s,
                            hazard=f"glove={first_hazard}, mechanical={first_mechanical}",
                            keyframe=step in (0, steps) or abs(elapsed-outward_end) < .010001)
                    if step < steps:
                        target = g1.control_action()
                        targets.append(cpu(target)[0])
                        g1.apply_physical_target(target)
                if step < steps:
                    binding.write(replay.sample((step+1)*.005), g1.data)
                    g1.step_substep()
    finally:
        if media is not None:
            media.close()
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays["physical_targets"] = np.asarray(targets)
    np.savez_compressed(destination/"trace.npz", **arrays)
    if visual_arrays:
        np.savez_compressed(destination/"delivered_visuals.npz",
            **{key: np.stack([row[key] for row in visual_arrays]) for key in visual_arrays[0]},
            acquisition_time_s=np.array([row["time_s"] for row in camera_rows]))
    dump(destination/"camera.json", camera_rows)
    outcome = classify_trace(arrays, active_index=active, outward_end_s=outward_end,
                             reference_end_s=replay.duration_s)
    first_safety = outcome["first_safety_failure_s"]
    fixture_id = getattr(fixture, "metadata", {}).get("fixture_id", "standing_fixture")
    displacements = np.linalg.norm(arrays["qpos"][:,:3]-arrays["qpos"][0,:3],axis=1)
    finite_displacements = displacements[np.isfinite(displacements)]
    summary = dict(label=label, fixture_id=fixture_id, primitive_id=primitive.primitive_id,
        reference_npz=str(primitive.reference.path), reference_sha256=primitive.reference.sha256,
        active_hand=primitive.active_hand, family=primitive.family, selection=selection,
        reset=reset, atomic_preserved=contract, launch_head_world=frozen_head,
        launch_head_velocity_world=launch_head_velocity,
        actual_launch_geometry=actual_launch,
        placement_selected_after_actual_launch_geometry=selector_after_restore,
        actual_initial_h2_pelvis_world=primitive.reference.sample(0, placement).body_pos[0],
        placement=asdict(placement), clock="unit source playback", clock_rate=1.,
        fixed_calibration_landmark_used_for_clock=False, requested_ttc_s=None,
        physics_dt_s=.005, control_dt_s=.02, outward_end_source_s=outward_end,
        nominal_observation_noise=bool(g1.observation_noise_enabled),
        nominal_observation_noise_seed=g1.noise_seed,
        nominal_noise_context="restored exact captured fixture measurement and generator state",
        visual_fault_seed=camera.seed, delay_seed=camera.seed+1,
        visual_episode_counters_before_reset=camera_episode_before,
        controller="frozen Fresh96FlatPolicy/v1_20261007", avoidance_residual=0.,
        numerical_stop=numerical_stop, maximum_glove_fk_error_m=max_glove_error,
        g1_max_root_displacement_m=None if not len(finite_displacements) else float(finite_displacements.max()),
        g1_min_root_height_m=_finite_min(arrays["qpos"][:,2]),
        pre_hazard_visual_information=summarize_delivered_vision(camera_rows, first_safety, arrays["time_s"][-1]),
        video_frames=0 if media is None else media.count,
        media_complete=bool(media is not None and media.count == steps//4+1),
        **outcome)
    dump(destination/"summary.json", summary)
    if max_glove_error > 2e-5:
        raise AssertionError(f"Prescribed glove/FK mismatch {max_glove_error}")
    print(f"DYNAMIC {label}: outward={outcome['frozen_head_active_outward_entry_s']} "
          f"moving={outcome['moving_head_first_encounter_s']} clearance={first_hazard} "
          f"mechanical={first_mechanical} complete={outcome['recovery_complete']}", flush=True)
    return summary
