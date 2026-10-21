#!/usr/bin/env python3
"""Independent saved-trace audit of Phase 1; no integration or policy updates.

Reconstructs all operational measurements, interval dwell, phase/contact ledgers
and diagnostic quantities without importing the functional evaluator. The
accepted Phase 0 independent auditor supplies controller/history reconstruction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase0_independent", ROOT / "scripts/task10b_phase0_audit.py")
phase0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase0)
digest, read_json, read_npz = phase0.digest, phase0.read_json, phase0.read_npz
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
FIXED = dict(glove_position_m=.05, root_linear_speed_m_s=.05, root_angular_speed_rad_s=.12,
             glove_relative_speed_m_s=.15, dwell_s=.20, minimum_root_height_fraction=.75,
             maximum_root_tilt_rad=.60, support_normal_force_n=1.e-6,
             diagnostic_glove_orientation_rad=float(np.deg2rad(20)), diagnostic_torso_orientation_rad=.15,
             diagnostic_root_height_fraction=.90, diagnostic_com_margin_m=.02,
             diagnostic_joint_speed_rad_s=.50, diagnostic_clearance_m=.16, diagnostic_stability_s=.50)


def reference_schedule(master, clip):
    """Reference-only motion onset, fixed before native Phase 1 evaluation."""
    changed = (np.any(np.abs(master["qpos"]-master["qpos"][0]) > 1.e-7, axis=1)
               | np.any(np.abs(master["qvel"]) > 1.e-7, axis=1))
    moving = np.flatnonzero(changed)
    if not len(moving):
        raise ValueError("Reference contains no outward execution")
    onset = int(moving[0])
    return dict(first_reference_motion_tick=onset, launch_tick=onset//4*4,
                outward_tick=int(np.ceil(float(clip["outward_end_s"])/.005-1.e-10)))


def continuous_interval_dwell(times, pose, support, *, start_index=0):
    """Vectorized independent duration since last invalid complete interval.

    Contact forces at row k belong to (k-1,k); pose is required at both ends.
    The initial/reset row contributes no duration. A support interruption cannot
    borrow an earlier valid interval, and outward completion starts a new clock.
    """
    t, pose, support = np.asarray(times, float), np.asarray(pose, bool), np.asarray(support, bool)
    if t.ndim != 1 or pose.shape != t.shape or support.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("Strictly increasing matched time/pose/support arrays required")
    valid = np.r_[False, pose[:-1] & pose[1:] & support[1:]]
    valid[:start_index+1] = False
    previous_bad = np.maximum.accumulate(np.where(valid, 0, np.arange(len(t))))
    return np.where(valid, t-t[previous_bad], 0.)


def independently_classify(times, measured, launch_tick, outward_tick, active_hand):
    """Recompute events as masks and contiguous interval runs, without a state machine."""
    t = np.asarray(times, float)
    ready = np.asarray(measured["readiness_admission"], bool)
    admitted = bool(ready[launch_tick])
    quiet = continuous_interval_dwell(t, measured["operational_pose"], measured["operational_support"])
    recovery = (continuous_interval_dwell(t, measured["operational_pose"], measured["operational_support"],
                                         start_index=outward_tick) if admitted else np.zeros(len(t)))
    found = np.flatnonzero(recovery >= .20-1.e-10)
    complete = np.zeros(len(t), bool)
    if len(found):
        complete[found[0]:] = True
    outward = (np.arange(len(t)) >= launch_tick) & (np.arange(len(t)) <= outward_tick)
    contact = measured["frozen_target_glove_distance_m"][:, active_hand] <= .14
    hits = np.flatnonzero(contact & outward)
    any_contact = np.flatnonzero(np.any(measured["frozen_target_glove_distance_m"] <= .14, axis=1))
    inactive_contact = np.flatnonzero(measured["frozen_target_glove_distance_m"][:, 1-active_hand] <= .14)
    inactive_outward = np.flatnonzero((measured["frozen_target_glove_distance_m"][:, 1-active_hand] <= .14) & outward)
    phases = np.full(len(t), "holding", dtype="U8")
    if admitted:
        phases[launch_tick:outward_tick] = "outward"
        phases[outward_tick:] = "recovery"
    phases[complete] = "complete"
    first_ready = np.flatnonzero(ready)
    physical = bool(np.all(measured["operational_finite"]) and np.all(measured["operational_upright"]))
    recovery_time = float(t[found[0]]) if len(found) else None
    stability = np.zeros(len(t), bool) if recovery_time is None else (t >= recovery_time) & (t <= recovery_time+.5+1.e-10)
    observed = 0. if recovery_time is None else min(float(t[-1]-recovery_time), .5)
    full = recovery_time is not None and t[-1]+1.e-10 >= recovery_time+.5
    events = dict(phase=phases, ready=ready, complete=complete, ready_dwell_s=quiet,
                  recovery_dwell_s=recovery, outward_window=outward, outward_contact=outward & contact,
                  stability_observation=stability)
    summary = dict(initial_completion_false=not bool(complete[0]),
        readiness_time_s=float(t[first_ready[0]]) if len(first_ready) else None,
        launch_time_s=float(t[launch_tick]) if admitted else None, launch_rejected=not admitted,
        outward_completion_time_s=float(t[outward_tick]) if admitted else None,
        recovery_time_s=recovery_time, recovered_after_outward=bool(len(found)),
        physical_finite_upright=physical, functional_complete=bool(len(found) and physical and admitted),
        contact_outcome="hit" if len(hits) else "miss", active_hand_outward_contact=bool(len(hits)),
        first_active_hand_outward_contact_s=float(t[hits[0]]) if len(hits) else None,
        first_any_hand_contact_s=float(t[any_contact[0]]) if len(any_contact) else None,
        first_inactive_hand_contact_s=float(t[inactive_contact[0]]) if len(inactive_contact) else None,
        first_inactive_hand_outward_contact_s=float(t[inactive_outward[0]]) if len(inactive_outward) else None,
        inactive_hand_first_outward_contact=bool(len(inactive_outward) and (not len(hits) or inactive_outward[0] < hits[0])),
        final_recovery_dwell_s=float(recovery[-1]), diagnostic_stability_observed_s=observed,
        diagnostic_stability_complete=bool(full),
        diagnostic_stability_pass=bool(np.all(measured["operational_guard"][stability])) if full else None)
    return summary, events


def independent_support(model, raw):
    floor = model.geom("terrain").id
    feet = [model.geom("robot/physical/"+side+"Foot").id for side in ("Left", "Right")]
    frames, worlds = raw["time"].shape
    result = np.zeros((frames, worlds, 2), bool)
    offsets = raw["contact_offsets"]
    if len(offsets) != frames+1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("Malformed interval contact offsets")
    for tick in range(frames):
        first, last = offsets[tick:tick+2]
        pair, force, owner = (raw["contact_"+key][first:last] for key in ("geom", "force", "worldid"))
        loaded = (force[:, 0] > 1.e-6) & np.any(pair == floor, axis=1)
        for side, foot in enumerate(feet):
            result[tick, owner[loaded & np.any(pair == foot, axis=1)], side] = True
    return result


def independent_measurements(model, q, v, guard, target, support):
    """Fresh FK, point Jacobians and fixed nominal-foot-face geometry for every row."""
    import mujoco
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    q, v, guard = np.asarray(q, float), np.asarray(v, float), np.asarray(guard, float)
    chest, head, root = [model.body("robot/"+name).id for name in ("Chest", "Head", "Hips")]
    gloves = [model.site("robot/"+side+"Hand_center").id for side in ("Left", "Right")]
    feet = [model.geom("robot/physical/"+side+"Foot").id for side in ("Left", "Right")]
    data = mujoco.MjData(model)
    data.qpos[:] = guard
    mujoco.mj_kinematics(model, data)
    sole = []
    for geom in feet:
        rotation = data.geom_xmat[geom].reshape(3, 3)
        normal = int(np.argmax(abs(rotation[2])))
        other = [i for i in range(3) if i != normal]
        corners = np.zeros((4, 3))
        corners[:, normal] = (-1 if rotation[2, normal] >= 0 else 1)*model.geom_size[geom, normal]
        corners[:, other] = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*model.geom_size[geom, other]
        sole.append(corners)
    rows = []
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    # Nominal row is evaluated by the same independent FK call, without using saved nominal measurements.
    for pose, velocity in zip(np.vstack((guard, q)), np.vstack((np.zeros(model.nv), v)), strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        chest_rot = data.xmat[chest].reshape(3, 3)
        mujoco.mj_jacBody(model, data, jp, jr, chest)
        chest_vel, chest_omega = jp @ velocity, jr @ velocity
        point, rotation, speed = [], [], []
        for site in gloves:
            offset = data.site_xpos[site]-data.xpos[chest]
            point.append(chest_rot.T @ offset)
            rotation.append(chest_rot.T @ data.site_xmat[site].reshape(3, 3))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            speed.append(chest_rot.T @ (jp @ velocity-chest_vel-np.cross(chest_omega, offset)))
        corners = np.array([sole[i] @ data.geom_xmat[g].reshape(3, 3).T+data.geom_xpos[g]
                            for i, g in enumerate(feet)])
        hull = ConvexHull(corners[..., :2].reshape(-1, 2))
        margin = -np.max((hull.equations[:, :2] @ data.subtree_com[root, :2]+hull.equations[:, 2])
                         / np.linalg.norm(hull.equations[:, :2], axis=1))
        rows.append(dict(glove_position_w=data.site_xpos[gloves].copy(),
            glove_position_chest=np.array(point), glove_rotation_chest=np.array(rotation),
            glove_velocity_chest=np.array(speed), chest_rotation_w=data.xmat[chest].reshape(3,3).copy(),
            head_rotation_w=data.xmat[head].reshape(3,3).copy(), root_rotation_w=data.xmat[root].reshape(3,3).copy(),
            root_position_w=data.xpos[root].copy(), sole_corners_w=corners,
            com_w=data.subtree_com[root].copy(), com_support_margin_m=float(margin)))
    nominal, values = rows[0], {key: np.asarray([row[key] for row in rows[1:]]) for key in rows[0]}
    def angle(base, rotation):
        relative = np.einsum("...ji,...jk->...ik", base, rotation)
        return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
    rotations = Rotation.from_quat(q[:,3:7], scalar_first=True)
    up = rotations.apply(np.broadcast_to(rotations[0].inv().apply([0,0,1]), (len(q),3)))
    values.update(hand_position_error_m=np.linalg.norm(values["glove_position_chest"]-nominal["glove_position_chest"], axis=-1),
        hand_relative_speed_m_s=np.linalg.norm(values["glove_velocity_chest"], axis=-1),
        root_linear_speed_m_s=np.linalg.norm(v[:,:3], axis=-1), root_angular_speed_rad_s=np.linalg.norm(v[:,3:6], axis=-1),
        root_height_m=values["root_position_w"][:,2], root_tilt_rad=np.arccos(np.clip(up[:,2], -1,1)),
        root_height_fraction=values["root_position_w"][:,2]/nominal["root_position_w"][2],
        root_position_error_m=np.linalg.norm(q[:,:3]-guard[:3], axis=-1),
        root_orientation_error_rad=angle(nominal["root_rotation_w"], values["root_rotation_w"]),
        joint_return_error_rad=abs(q[:,7:]-guard[7:]), joint_speed_rad_s=abs(v[:,6:]),
        hand_orientation_error_rad=angle(nominal["glove_rotation_chest"], values["glove_rotation_chest"]),
        torso_orientation_error_rad=angle(nominal["chest_rotation_w"], values["chest_rotation_w"]),
        head_orientation_error_rad=angle(nominal["head_rotation_w"], values["head_rotation_w"]),
        frozen_target_glove_distance_m=np.linalg.norm(values["glove_position_w"]-target, axis=-1))
    components = dict(finite=np.isfinite(q).all(-1)&np.isfinite(v).all(-1),
        upright=(values["root_height_fraction"]>=.75)&(values["root_tilt_rad"]<=.60),
        hand_position=np.all(values["hand_position_error_m"]<=.05,axis=1),
        root_linear_speed=values["root_linear_speed_m_s"]<=.05,
        root_angular_speed=values["root_angular_speed_rad_s"]<=.12,
        hand_relative_speed=np.all(values["hand_relative_speed_m_s"]<=.15,axis=1))
    values.update({"operational_"+key: value for key,value in components.items()})
    values["support_contact"] = support
    values["operational_support"] = support.all(-1)
    values["operational_pose"] = np.logical_and.reduce(list(components.values()))
    values["operational_guard"] = values["operational_pose"]&values["operational_support"]
    values["readiness_admission"] = components["finite"]&components["upright"]&components["hand_position"]&values["operational_support"]
    return values, nominal


def same(actual, expected, atol=1.e-8):
    if actual is None or expected is None:
        return actual is None and expected is None
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "biuUSO" or b.dtype.kind in "biuUSO":
        return bool(np.array_equal(a,b))
    return bool(np.allclose(a,b,atol=atol,rtol=0,equal_nan=True))


def audit_functional(folder, model, raw, clip, legacy, origins):
    panel = read_json(folder/"functional_evaluation/summary.json")
    steps, worlds = int(np.ceil(float(clip["legacy_episode_duration_s"])/.005)), raw["time"].shape[1]
    n = steps+1
    t = np.arange(n)*.005
    support = independent_support(model, raw)
    launch = int(round(panel["launch_time_s"]/.005))
    outward = int(round(panel["outward_complete_time_s"]/.005))
    active = 0 if clip["family"] in ("jab","left_hook") else 1
    checks = dict(trial_ids=panel["trial_ids"]==list(range(worlds)),
        all_initial_thresholds_unchanged=all(panel["predicate"].get(k)==v for k,v in FIXED.items()),
        original_horizon=panel["historical_physics_steps"]==steps,
        explicit_actor_padding=panel["excluded_actor_padding_steps"]==len(raw["time"])-n,
        weights_frozen=panel["weights_changed"] is False and panel["training"] is False)
    rows=[]
    for trial in range(worlds):
        path=folder/"functional_evaluation"/f"functional_{trial:02d}.npz"
        saved, report = read_npz(path), read_json(path.with_name(f"trial_{trial:02d}.json"))
        guard = np.asarray(legacy["qpos"][0],float).copy()
        # Match the recorded physical world origin itself; no inference from float32 subtraction.
        guard[:3] += origins[trial]
        measured, nominal = independent_measurements(model, raw["qpos"][:n,trial],raw["qvel"][:n,trial],
            guard,np.asarray(clip["target_w"])+origins[trial],support[:n,trial])
        expected, events = independently_classify(t, measured, launch, outward, active)
        flags = {"measurement_"+key:same(saved.get(key),value,atol=5.e-8) for key,value in measured.items()}
        flags.update({"event_"+key:same(saved.get(key),value) for key,value in events.items()})
        flags.update({"ledger_"+key:same(report.get(key),value) for key,value in expected.items()})
        flags["source_trace_hash"] = report["measurement_sha256"]==digest(path)
        flags["panel_row_matches_trial"] = report==panel["trials"][trial]
        flags["time_grid"] = same(saved.get("time_s"),t,atol=1.e-12)
        nominal_saved=read_json(path.with_name(f"nominal_{trial:02d}.json"))
        flags["nominal_centers_from_qualified_reference"] = same(nominal_saved["glove_centers_chest_m"],nominal["glove_position_chest"])
        historical=read_json(folder/"historical_evaluation"/f"trial_{trial:02d}.json")
        flags["historical_result_separate"] = report["historical_exact_pose"]["strict_historical_release_pass"]==historical["strict_historical_release_pass"]
        commanded=bool(expected["active_hand_outward_contact"] and historical["acceptance_checks"]["active_hand_attack_contact"]
            and all(historical["historical_release_precision"].values()))
        flags["commanded_hit_does_not_follow_from_recovery"] = report["commanded_hit_success"]==commanded
        flags["nominal_complete_cycle_requires_hit_and_recovery"] = report["nominal_complete_cycle"]==(commanded and expected["functional_complete"])
        expected.update(commanded_hit_success=commanded,nominal_complete_cycle=bool(commanded and expected["functional_complete"]))
        diagnostics=dict(hand_orientation_20deg=np.all(measured["hand_orientation_error_rad"]<=np.deg2rad(20),axis=-1),
            torso_orientation_0p15rad=measured["torso_orientation_error_rad"]<=.15,
            root_height_90pct=measured["root_height_fraction"]>=.90,com_margin_2cm=measured["com_support_margin_m"]>=.02,
            joint_speed_0p50rad_s=measured["joint_speed_rad_s"].max(-1)<=.50,
            frozen_target_clearance_0p16m=np.all(measured["frozen_target_glove_distance_m"]>=.16,axis=-1))
        flags.update({"diagnostic_"+k:same(saved.get("diagnostic_"+k),v) for k,v in diagnostics.items()})
        flags["diagnostic_final_checks"] = report["diagnostic_final_checks"]=={k:bool(v[-1]) for k,v in diagnostics.items()}
        flags["diagnostic_final_values"] = all(same(v,measured[k][-1],atol=5.e-8) for k,v in report["diagnostic_final_values"].items())
        flags["operational_failure_counts"] = report["operational_failure_sample_counts"]=={
            k:int(np.count_nonzero(~v)) for k,v in measured.items() if k.startswith("operational_")}
        rows.append(dict(trial_id=trial,passed=all(flags.values()),failed_checks=[k for k,v in flags.items() if not v],
                         checks=flags,independent_result=expected))
    checks["all_trial_measurements_events_and_ledgers"] = all(r["passed"] for r in rows)
    checks["functional_count"] = panel["functional_complete_count"]==sum(r["independent_result"]["functional_complete"] for r in rows)
    checks["contact_count"] = panel["active_hand_outward_hit_count"]==sum(r["independent_result"]["active_hand_outward_contact"] for r in rows)
    checks["commanded_hit_count"] = panel["commanded_hit_success_count"]==sum(r["independent_result"]["commanded_hit_success"] for r in rows)
    checks["nominal_cycle_count"] = panel["nominal_complete_cycle_count"]==sum(r["independent_result"]["nominal_complete_cycle"] for r in rows)
    checks["recovered_miss_count"] = panel["recovered_miss_count"]==sum(r["independent_result"]["functional_complete"] and r["independent_result"]["contact_outcome"]=="miss" for r in rows)
    return dict(checks=checks,trials=rows,trial_count=worlds,functional_complete_count=panel["functional_complete_count"])


def audit_packet(packet_path, samples):
    import mujoco
    import torch
    torch.set_num_threads(1)
    packet_path=Path(packet_path).resolve(strict=True)
    packet=read_json(packet_path)
    archive=packet_path.parent/"source_snapshot"
    checks={"phase1_only":packet["program_phase"]==1 and packet["training"] is False and packet["learning_updates"]==0}
    bad,changed=[],[]
    for relative,expected in packet["files"].items():
        path=(archive/relative).resolve(strict=True)
        if not path.is_relative_to(archive.resolve()) or digest(path)!=expected:bad.append(relative)
        live=ROOT/relative
        if not live.is_file() or digest(live)!=expected:changed.append(relative)
    checks["all_archived_input_hashes"]=not bad
    if bad:raise ValueError("Executed archived inputs changed: "+str(bad))
    execution=read_json(packet_path.parent/"execution.json")
    native_path=ROOT/packet["output"]/"audit.json"
    native=read_json(native_path)
    checks.update(execution_binds_packet=execution["packet_sha256"]==digest(packet_path),
        execution_binds_native=execution.get("audit_sha256")==digest(native_path),
        execution_complete=execution["status"]=="passed" and native["status"]=="passed",
        four_accepted_frozen_conditions=[(r["family"],r["solver_profile"],r["residual_mode"],r["feedforward_enabled"]) for r in native["conditions"]]
            ==[(f,"accepted","learned",True) for f in FAMILIES],
        truthful_unattempted=native["unattempted_conditions"]==4-len(native["conditions"]))
    clips={c["family"]:c for c in read_json(archive/packet["bundle"]/"manifest.json")["clips"]}
    results=[]
    for condition in native["conditions"]:
        family=condition["family"]
        folder=native_path.parent/condition["name"]
        raw=read_npz(folder/"raw_trace.npz")
        model=mujoco.MjModel.from_binary_path(str(folder/"compiled_model.mjb"))
        source=archive/packet["release"]/family
        bank=read_npz(source/"evaluation/evaluation_bank.npz")
        worlds=packet["environments_per_condition"]
        schema=read_json(source/"controller_schema.json")
        schema["_compiled_joint_limits"]=model.jnt_range[1:].copy()
        clip=clips[family]
        master=read_npz(archive/packet["bundle"]/clip["master_path"])
        legacy=read_npz(archive/packet["bundle"]/clip["legacy_runtime_path"])
        control,origins=phase0.audit_arrays(raw,schema,master,bank["qpos"][:worlds],bank["qvel"][:worlds],feedforward=True,residual="learned")
        c=control["checks"]
        c["raw_evidence_hash"]=digest(folder/"raw_trace.npz")==condition["raw_trace_sha256"]
        c["compiled_model_hash"]=digest(folder/"compiled_model.mjb")==condition["compiled_model"]["sha256"]
        c["standalone_condition_receipt"]=condition==read_json(folder/"audit.json")
        c["frozen_actor_and_one_initialization"]=(condition["actor_unchanged"] is True and condition["physical_write_counts"]==[1]*worlds
            and condition["physical_reset_counts"]==[1]*worlds)
        c["actual_compiled_solver"]=(model.opt.timestep,int(model.opt.integrator),int(model.opt.solver),int(model.opt.cone),
            int(model.opt.iterations),int(model.opt.ls_iterations),model.opt.tolerance)==(.005,3,2,0,10,20,1e-8)
        c["actual_compiled_controller"] = bool((model.nq,model.nv,model.nu)==(55,54,48)
            and np.all(model.actuator_gainprm[:,0]==1)
            and np.array_equal(model.actuator_biasprm[:,1],-np.asarray(schema["kp"]))
            and np.array_equal(model.actuator_biasprm[:,2],-np.asarray(schema["kd"]))
            and np.array_equal(model.actuator_forcerange,schema["torque_limits"])
            and np.all(model.actuator_forcelimited) and not np.any(model.actuator_ctrllimited)
            and np.array_equal(model.actuator_trnid[:,0],np.arange(1,49)))
        c["frozen_release_inputs"] = all(digest(source/relative)==native["source_inputs"][family][key]
                                         for key,relative in phase0.SOURCE_KEYS.items())
        c["executed_counters"]=(condition["actual_physics_steps"]==control["steps"] and condition["actual_actor_updates"]==control["policies"]
            and condition["actual_physics_steps_per_world"]==[control["steps"]]*worlds)
        original=int(np.ceil(float(clip["legacy_episode_duration_s"])/.005))
        c["original_horizon_and_actor_padding"]=control["steps"]==((original+3)//4)*4
        actor=phase0.sampled_actor(model,raw,origins,legacy,clip["legacy_episode_duration_s"],source/"checkpoint.pt","learned",samples)
        c.update(actor["checks"])
        historical=phase0.parity_evidence(folder,raw,condition,origins,archive,packet["files"],model,bank["qpos"][:worlds],original)
        c.update(historical["checks"])
        # Float32 subtraction is not a faithful way to reconstruct origins. Each
        # archived rollout retains the actual origin, already forward-audited above.
        recorded_origins=np.array([read_npz(folder/"historical_evaluation"/f"rollout_{i:02d}.npz")["world_origin"] for i in range(worlds)])
        functional=audit_functional(folder,model,raw,clip,legacy,recorded_origins)
        c.update(functional["checks"])
        schedule=reference_schedule(master,clip)
        panel=condition["functional_evaluation"]
        c["reference_only_launch_schedule"]=(panel["launch_time_s"]==schedule["launch_tick"]*.005
            and panel["outward_complete_time_s"]==schedule["outward_tick"]*.005)
        c["native_binds_functional_summary"]=panel==read_json(folder/"functional_evaluation/summary.json")
        media=None
        if packet["mode"]=="pilot":
            import imageio.v2 as imageio
            media=condition["media"]
            bound=all(digest(folder/name)==item["sha256"] and (folder/name).stat().st_size==item["bytes"] for name,item in media["artifacts"].items())
            with imageio.get_reader(media["video"]) as reader:decoded=sum(1 for _ in reader)
            expected_frames=len(np.arange(0,original*.005+1e-9,1/50))
            c["nominal_media_full_decode_and_hashes"]=bool(bound and decoded==media["video_frames"]==media["decoded_frames"]==expected_frames)
            media=dict(video=media["video"],decoded_frames=decoded,sha256=digest(Path(media["video"])))
        results.append(dict(family=family,passed=all(c.values()),checks=c,control=control,
            actor=actor,historical=historical,functional=functional,reference_schedule=schedule,media=media))
        print(json.dumps(dict(family=family,audit_pass=results[-1]["passed"],functional=panel["functional_complete_count"],worlds=worlds)),flush=True)
    checks["all_conditions_independently_pass"]=all(r["passed"] for r in results) and len(results)==4
    checks["world_transition_count"]=native["actual_world_physics_transitions"]==sum(r["control"]["steps"]*r["control"]["worlds"] for r in results)
    return dict(schema="task10b_phase1_independent_audit_v1",status="passed" if all(checks.values()) else "failed_or_incomplete",
        packet=str(packet_path),packet_sha256=digest(packet_path),auditor_sha256=digest(__file__),checks=checks,
        archived_input_files=len(packet["files"]),later_live_source_changes=changed,conditions=results,
        no_physics_integration=True,no_learning=True,
        scope="Every operational/diagnostic FK and ledger row, all controller/history intervals, sampled fresh CPU actor observations/inference, all historical trial links and nominal/last historical re-evaluation; nominal media fully decoded.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sample-policy-boundaries",type=int,default=9)
    args=parser.parse_args(argv)
    destination=args.output.resolve()
    if not destination.is_relative_to((ROOT/"outputs/task10b").resolve()) or destination.exists() or args.sample_policy_boundaries<1:
        raise ValueError("New human-owned Task10B audit output and positive sampling required")
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"auditor_source.py").write_bytes(Path(__file__).read_bytes())
    result=audit_packet(args.packet,args.sample_policy_boundaries)
    with (destination/"audit.json").open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],output=str(destination/"audit.json"),checks=result["checks"])))
    return 0 if result["status"]=="passed" else 2


if __name__=="__main__":
    raise SystemExit(main())
