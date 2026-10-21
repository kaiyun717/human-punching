#!/usr/bin/env python3
"""Independent Phase2 saved-trace audit; no integration or policy execution.

Reconstructs contact, guard recovery, reference sampling, physical forces and raw
action history without importing the Phase2 native evaluator. A correct negative
measurement remains negative; this audit cannot confer visual/domain acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


# BEGIN INDEPENDENT CLOCK V2 ANCESTRY
AUDIT_CLOCK_VERSION = "task10b_phase2_independent_clock_v2"
AUDIT_CLOCK_PARENT_SHA256 = "86de2f9e1260c98bbc5379c6156b4840b284b79342981dde3ba06c384aa824dd"


def clock_version_receipt():
    parent = Path(__file__).with_name("task10b_phase2_audit.py")
    if digest(parent) != AUDIT_CLOCK_PARENT_SHA256:
        raise ValueError("Preserved independent v1 auditor bytes changed")
    return dict(schema="task10b_phase2_independent_clock_ancestry_v1",
        version=AUDIT_CLOCK_VERSION,source_sha256=digest(__file__),
        parent_source=dict(path=str(parent.resolve()),sha256=AUDIT_CLOCK_PARENT_SHA256),
        corrected_original_own_step_expression="ceil(episode_duration_s / .005 - 1e-10)",
        integer_lattice_roundoff_tolerance_ticks=1e-10,
        nonoriginal_actor_padding_unchanged=True,event_clocks_unchanged=True,
        old_failed_attempts_reclassified=False,learning_helper_imported=False,
        new_physics_steps=0)
# END INDEPENDENT CLOCK V2 ANCESTRY


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def sphere_crossing(times, centers, target, radius=.14):
    """First entry, including a crossing between two outside endpoint samples."""
    t, p, head = np.asarray(times, float), np.asarray(centers, float), np.asarray(target, float)
    if (t.ndim != 1 or len(t) == 0 or p.shape != (len(t), 3) or head.shape != (3,)
            or np.any(np.diff(t) <= 0) or not all(np.isfinite(a).all() for a in (t, p, head))):
        raise ValueError("Finite strictly increasing matched center trace required")
    relative = p-head
    if relative[0]@relative[0] <= radius**2:
        return float(t[0])
    for i in range(len(t)-1):
        delta = relative[i+1]-relative[i]
        a, b, c = float(delta@delta), float(2*relative[i]@delta), float(relative[i]@relative[i]-radius**2)
        discriminant = b*b-4*a*c
        candidates = []
        if a > 0 and discriminant >= 0:
            entry = (-b-np.sqrt(discriminant))/(2*a)
            if 0 <= entry <= 1:
                candidates.append(entry)
        if relative[i+1]@relative[i+1] <= radius**2:
            candidates.append(1.)
        if candidates:
            return float(t[i]+min(candidates)*(t[i+1]-t[i]))
    return None


def interval_dwell(times, pose, support, start_tick):
    """Support at k belongs to interval (k-1,k); both endpoint poses must pass."""
    t, pose, support = np.asarray(times), np.asarray(pose, bool), np.asarray(support, bool)
    if t.ndim != 1 or pose.shape != t.shape or support.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("Matched strictly increasing interval trace required")
    good = np.r_[False, pose[:-1]&pose[1:]&support[1:]]
    good[:start_tick+1] = False
    last_bad = np.maximum.accumulate(np.where(good, 0, np.arange(len(t))))
    return np.where(good, t-t[last_bad], 0.)


def decode_normal_forces(raw, *, pyramidal=True):
    """Decode each contact's normal from its owned recorded constraint rows."""
    offsets = raw["contact_offsets"]
    owners, dimensions, addresses = (raw[k] for k in
        ("contact_worldid", "contact_dimension", "contact_efc_address"))
    constraint = raw["efc_force"]
    result = np.zeros(len(owners))
    for tick in range(len(offsets)-1):
        for i in range(int(offsets[tick]), int(offsets[tick+1])):
            owner, dimension, address = int(owners[i]), int(dimensions[i]), int(addresses[i,0])
            if owner < 0 or owner >= constraint.shape[1] or dimension not in (1,3,4,6) or address < -1:
                raise ValueError("Invalid contact owner, dimension or constraint address")
            if address < 0:
                continue
            width = 1 if not pyramidal or dimension == 1 else 2*(dimension-1)
            if address+width > constraint.shape[2]:
                raise ValueError("Contact constraint span exceeds recorded allocation")
            result[i] = float(np.sum(constraint[tick,owner,address:address+width]))
    return result


def independent_support(raw, floor_id, foot_ids, *, pyramidal=True):
    frames, worlds = raw["time"].shape
    offsets, geom, force, owner = (raw[name] for name in
        ("contact_offsets", "contact_geom", "contact_force", "contact_worldid"))
    if (offsets.shape != (frames+1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or geom.shape != (int(offsets[-1]), 2) or force.shape != (len(geom), 6)
            or owner.shape != (len(geom),) or np.any(owner < 0) or np.any(owner >= worlds)
            or not np.isfinite(force).all()):
        raise ValueError("Malformed native contact force ledger")
    result = np.zeros((frames, worlds, 2), bool)
    normal = decode_normal_forces(raw,pyramidal=pyramidal) if "efc_force" in raw else force[:,0]
    for tick in range(frames):
        section = slice(offsets[tick], offsets[tick+1])
        pair, load, world = geom[section], normal[section], owner[section]
        ground = (load > 1e-6)&np.any(pair == floor_id, axis=1)
        for hand, foot in enumerate(foot_ids):
            result[tick, world[ground&np.any(pair == foot, axis=1)], hand] = True
    return result


def interpolate(times, values, query, *, quaternion=False):
    """Independent implementation of source float32 linear interpolation."""
    t, x, u = (np.asarray(value, np.float32) for value in (times, values, query))
    u = np.clip(u, t[0], t[-1])
    hi = np.clip(np.searchsorted(t, u, side="right"), 1, len(t)-1)
    lo = hi-1
    alpha = (u-t[lo])/(t[hi]-t[lo])
    result = x[lo]+alpha.reshape(alpha.shape+(1,)*(x.ndim-1))*(x[hi]-x[lo])
    if quaternion:
        result[..., 3:7] /= np.maximum(np.linalg.norm(result[..., 3:7], axis=-1, keepdims=True), 1e-8)
    return result


def integer_sample(values,ticks,*,quaternion=False):
    """Independent exact stored-knot sampling for expanded 5ms recipes."""
    ticks=np.asarray(ticks)
    if ticks.dtype.kind not in 'iu' or np.any(ticks<0):
        raise ValueError('Nonnegative integer reference ticks required')
    result=np.asarray(values,np.float32)[np.minimum(ticks,len(values)-1)].copy()
    if quaternion:result[...,3:7]/=np.maximum(np.linalg.norm(result[...,3:7],axis=-1,keepdims=True),np.float32(1e-8))
    return result


def reference_arrays(path,model):
    """Rebuild runtime-only glove FK after explicit native float32 conversion.

    Packed references preserve original offline evidence separately. Native
    playback borrows q/v/torque as float32 and derives missing glove knots from
    exactly those rounded poses; no hidden original full-precision FK is used.
    """
    reference=read_npz(path)
    if 'glove_pos' not in reference:
        data=mujoco.MjData(model);hands=[model.site(side+'Hand_center').id for side in ('Left','Right')]
        points=[]
        for pose in np.asarray(reference['qpos'],np.float32):
            data.qpos[:]=pose;mujoco.mj_kinematics(model,data);points.append(data.site_xpos[hands].copy())
        reference['glove_pos']=np.asarray(points,np.float32)
    return reference


def observation(q, v, gloves, previous, clock, reference, duration, rejected,*,reference_ticks=None):
    """Independent original369 vector including full phase denominator/history."""
    def sample(key, offset=0.):
        if reference_ticks is not None:
            extra={0.:0,.1:20,.25:50}[offset]
            return integer_sample(reference[key],np.asarray(reference_ticks)+np.where(rejected,0,extra),quaternion=key=='qpos')
        query=clock+np.where(rejected,np.float32(0.),np.float32(offset))
        return interpolate(reference["t"],reference[key],query,quaternion=key=="qpos")
    desired,velocity=sample("qpos"),sample("qvel")
    phase=clock/np.float32(duration)
    parts=[q[:,2:7],v[:,:6]*np.float32(.2),q[:,7:],v[:,6:]*np.float32(.1),
        (desired[:,:3]-q[:,:3])*5,desired[:,3:7],desired[:,7:]-q[:,7:],
        (velocity[:,6:]-v[:,6:])*np.float32(.1),
        np.stack((phase,np.sin(phase*np.float32(2*np.pi)),np.cos(phase*np.float32(2*np.pi))),axis=-1),previous]
    for offset in (.1,.25):
        future=sample("qpos",offset)
        parts.extend(((future[:,:3]-q[:,:3])*5,future[:,7:]-q[:,7:]))
    parts.append((sample("glove_pos")-gloves).reshape(len(q),-1)*5)
    return np.clip(np.concatenate(parts,axis=-1),-20,20)


def matching_checkpoint(paths, required_digest):
    import torch
    for path in paths:
        saved=torch.load(path,map_location="cpu",weights_only=True)
        h=hashlib.sha256()
        for key,value in sorted(saved["policy"].items()):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if h.hexdigest()==required_digest:
            return path,saved["policy"]
    raise ValueError("No preserved checkpoint matches the panel's complete actor/critic state digest")


def sampled_actor_audit(model,raw,row,world,reference,checkpoint,state,*,integer=False):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    policies=len(raw["policy_action"])
    selected={0,policies-1,min(policies-1,row["launch_tick"]//4),min(policies-1,row["outward_end_tick"]//4)}
    for index in np.linspace(0,policies-1,min(9,policies)).astype(int):
        if len(selected)>=min(9,policies):
            break
        selected.add(int(index))
    selected=np.asarray(sorted(selected),int)
    ticks=raw["policy_episode_tick"][selected].astype(int)
    q,v=raw["qpos"][ticks,world].copy(),raw["qvel"][ticks,world].copy()
    hands=[model.site(side+"Hand_center").id for side in ("Left","Right")]
    data=mujoco.MjData(model)
    gloves=np.empty((len(q),2,3),np.float32)
    for index,pose in enumerate(q):
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        gloves[index]=data.site_xpos[hands]
    clock=raw["policy_reference_tick"][selected,world].astype(np.float32)*np.float32(.005)
    rebuilt=observation(q,v,gloves,raw["policy_previous_raw_action"][selected,world],clock,
        reference,row["phase_denominator_s"],raw["policy_rejected"][selected,world],
        reference_ticks=raw['policy_reference_tick'][selected,world] if integer else None)
    recorded=raw["policy_observation"][selected,world]
    obs_error=float(np.max(np.abs(rebuilt-recorded),initial=0))
    actor=nn.Sequential(nn.Linear(369,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,48))
    actor.load_state_dict({key.removeprefix("actor."):value for key,value in state.items() if key.startswith("actor.")})
    with torch.no_grad():
        action=actor(torch.from_numpy(recorded)).numpy()
    action_error=float(np.max(np.abs(action-raw["policy_action"][selected,world]),initial=0))
    return {"passed":obs_error<=5e-5 and action_error<=5e-5,"policy_indices":selected.tolist(),
        "actor_episode_ticks":ticks.tolist(),"sampled_observations":len(selected),
        "maximum_observation_error":obs_error,"maximum_cpu_action_error":action_error,
        "tolerance":5e-5,"checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint),
        "sampling":"exact_integer_5ms_knots" if integer else "original_float32_interpolation",
        "scope":"Fresh CPU FK and original369 reconstruction; CPU actor inference on recorded observations separates inference arithmetic from FK error"}


def commitment_audit(raw,commitments,plan,cells,accepted_by_world=None):
    by_world={}
    checks={"no_duplicate_world_commitment":True,"all_reached_boundaries_recorded":True}
    for item in commitments:
        world=int(item["world"])
        if world in by_world:
            checks["no_duplicate_world_commitment"]=False
        by_world[world]=item
        row=cells[plan["candidates"][world]]
        tick=int(item["requested_episode_tick"])
        actor=tick//4
        prefix=f"world_{world}_"
        checks[prefix+"actor_boundary"]=tick%4==0 and tick==row["launch_tick"] and actor<len(raw["policy_action"])
        if not checks[prefix+"actor_boundary"]:
            continue
        admitted=(bool(raw["policy_readiness"][actor,world]) if accepted_by_world is None
            else accepted_by_world.get(world,False))
        state=item["measured_human_state"]
        checks[prefix+"current_state"]=all(np.array_equal(state[key],raw[key][tick,world]) for key in ("qpos","qvel","ctrl"))
        checks[prefix+"controller_history"]=np.array_equal(item["previous_raw_action"],raw["policy_previous_raw_action"][actor,world]) and np.array_equal(item["outgoing_raw_action"],raw["policy_action"][actor,world])
        checks[prefix+"warmstart"]=np.array_equal(state["qacc_warmstart"],raw["policy_qacc_warmstart"][actor,world])
        checks[prefix+"unchanged_request"]=item["requested_family"]==row["family"] and item["requested_ttc_s"]==row["requested_ttc_s"] and item["requested_style"]==row["candidate_id"]
        checks[prefix+"reference_target_identity"]=item["reference_sha256"]==row["reference_sha256"] and item["legacy_runtime_sha256"]==row["legacy_runtime_sha256"] and np.array_equal(item["target_w"],row["target_w"])
        checks[prefix+"current_readiness_launch"]=item["actual_launch_time_s"]==(tick*.005 if admitted else None) and (item["disposition"]=="committed")==admitted and not item["state_and_history_replaced"]
    for world,candidate in enumerate(plan["candidates"]):
        reached=cells[candidate]["launch_tick"]//4<len(raw["policy_action"])
        checks["all_reached_boundaries_recorded"] &= (world in by_world)==reached
    return {"passed":all(checks.values()),"checks":checks,"commitments":len(commitments)}


def json_identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def launch_snapshot_digest(state):
    """Independent typed digest; imports no launch/evaluator implementation."""
    result=hashlib.sha256()
    for key,value in sorted(state.items()):
        result.update(key.encode())
        if isinstance(value,np.ndarray):
            result.update(value.dtype.str.encode());result.update(str(value.shape).encode())
            result.update(np.ascontiguousarray(value).tobytes())
        else:result.update(json.dumps(value,sort_keys=True,allow_nan=False).encode())
    return result.hexdigest()


def closed_launch_audit(raw,receipt,plan,cells,launch,declaration,domain,targets,boundary_arrays,*,shared_initialization=None):
    """Audit the isolated exact-prefix finalizer, including preserved rejections.

    This checks the supplied native/controller inventory against trace fields.
    It does not claim that inventory is a full mjSTATE_INTEGRATION restore state,
    qualify live G1 interaction, or convert a rejected request into a cycle.
    """
    checks={};accepted_by_world={};worlds=len(plan["candidates"])
    try:
        shared=declaration.get('start_mode')=='shared_jab_saved_state'
        expected_initialization=('task10b_shared_jab_complete_native_initialization_v1' if shared
            else 'task10b_independent_family_guard_zero_history_v1')
        checks["schema"]=(receipt["schema"]=="task10b_phase2_closed_protocol_native_receipt_v1"
            and receipt["initialization_version"]==expected_initialization
            and receipt["physical_state_or_history_replaced"] is False
            and receipt["arbitrary_live_state_admission"] is False
            and receipt["real_time_operation_qualified"] is False)
        bindings={b["trial_id"]:b for b in declaration["bindings"]}
        groups={g["snapshot"]["command_id"]:g for g in domain["selected_command_groups"]}
        candidates={t["candidate_identity"]:t for t in targets["geometry_candidates"]}
        checks["unique_canonical_targets"]=len(candidates)==len(targets["geometry_candidates"])
        checks["requested_trial_order"]=len(receipt["requests"])==worlds and len(plan["trials"])==worlds
        records=receipt["boundary_snapshots"]["snapshots"]
        checks["snapshot_schema"]=receipt["boundary_snapshots"]["schema"]=="task10b_phase2_lossless_launch_boundaries_v1"
        decoded=[];used_arrays=[]
        for index,record in enumerate(records):
            pair={}
            for side in ("before","after"):
                fields=record[side];state=dict(fields["scalar_fields"])
                if set(state)&set(fields["array_fields"]):raise ValueError("Overlapping boundary state fields")
                for key,name in fields["array_fields"].items():
                    value=boundary_arrays[name]
                    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                        raise ValueError("Nonfinite or object boundary array")
                    state[key]=value
                    used_arrays.append(name)
                checks[f"snapshot_{index}_{side}_digest"]=launch_snapshot_digest(state)==fields["snapshot_sha256"]
                pair[side]=state
            checks[f"snapshot_{index}_inventory"]=set(pair["before"])==set(pair["after"])
            decoded.append(pair)
        checks["lossless_archive_inventory"]=len(used_arrays)==len(set(used_arrays)) and set(used_arrays)==set(boundary_arrays)
        decisions={}
        checks["unique_world_decisions"]=True
        for item in receipt["decisions"]:
            world=item["world"]
            if type(world) is not int or not 0<=world<worlds:raise ValueError("Invalid decision world")
            if world in decisions:checks["unique_world_decisions"]=False
            decisions[world]=item
        fields=("trial_id","request_identity","family","requested_ttc_s","style","candidate_id",
            "reference_sha256","checkpoint_sha256","target_candidate_identity","snapshot_payload_sha256",
            "snapshot_slot_id","placement","head_point_velocity_w","target_w")
        target_fields=("trial_id","request_identity","target_candidate_identity","snapshot_payload_sha256",
            "snapshot_slot_id","placement","head_point_velocity_w","target_w")
        for world,candidate in enumerate(plan["candidates"]):
            key=f"world_{world}_";row=cells[candidate];trial_id=plan["trials"][world];binding=bindings[trial_id]
            target=candidates[row["target_candidate_identity"]];group=groups[binding["command_id"]]
            snapshot=group["snapshot"]
            expected=dict(trial_id=trial_id,request_identity=json_identity(binding),family=row["family"],
                requested_ttc_s=row["requested_ttc_s"],style=candidate,candidate_id=candidate,
                reference_sha256=row["reference_sha256"],checkpoint_sha256=launch['evaluated_checkpoint']['sha256'],
                target_candidate_identity=target["candidate_identity"],snapshot_payload_sha256=target["payload_sha256"],
                snapshot_slot_id=target["snapshot_slot_id"],placement=target["placement"],
                head_point_velocity_w=target["head_point_velocity_w"],target_w=target["head_target_w"])
            checks[key+"canonical_target_binding"]=(json_identity({k:v for k,v in target.items() if k!="candidate_identity"})==target["candidate_identity"]
                and target["source_domain_sha256"]==launch["domain_sha256"]
                and target["command_id"]==binding["command_id"]==row["command_id"]
                and target["split"]==group["split"]==declaration["split"]==row["split"]
                and target["original_replica_id"]==binding["original_replica_id"]==snapshot["replica_id"]
                and target["snapshot_slot_id"]==snapshot["slot_id"]
                and target["payload_sha256"]==snapshot["payload_sha256"]
                and target["head_target_w"]==binding["target_w"]==row["target_w"]
                and row["reference_sha256"]==binding["reference_sha256"]
                and row["requested_ttc_s"]==binding["ttc_s"]
                and (launch.get('checkpoint_template_is_unbound') is True
                    or launch["evaluated_checkpoint"]["sha256"]==declaration["checkpoint_sha256"]))
            checks[key+"request_exact"]=receipt["requests"][world]==expected
            q=raw["qpos"][0,world].astype(np.float32);v=raw["qvel"][0,world].astype(np.float32)
            initial=dict(version="task10b_independent_family_guard_zero_history_v1",qpos=q.tolist(),qvel=v.tolist(),
                previous_raw_action=[0.]*48,held_action_initialization="original independent sim.reset, before first actor call",
                controller_initialization="fresh native Simulation and original sim.reset; no saved-state restoration")
            checks[key+"declared_initial_state"]=(shared_initialization is not None
                and shared_initialization.get('passed') is True
                and shared_initialization.get('trial_state_bindings',{}).get(trial_id)==binding['start_state_sha256']
                if shared else json_identity(initial)==binding["start_state_sha256"])
            checks[key+"single_initialization"]=bool(np.all(raw["reset_count"][:,world]==1)
                and np.all(raw["policy_previous_raw_action"][0,world]==0))
            launch_tick=row["launch_tick"];actor=launch_tick//4
            reached=actor<len(raw["policy_action"])
            checks[key+"reached_boundary_recorded"]=(world in decisions)==reached
            if not reached:continue
            item=decisions[world];index=item["boundary_snapshot_index"]
            before,after=(decoded[index][side] for side in ("before","after"))
            checks[key+"boundary_events"]=(type(launch_tick) is int and launch_tick%4==0
                and records[index]["episode_tick"]==launch_tick and item["reference_tick"]==launch_tick
                and item["trial_id"]==trial_id and item["request"]==expected)
            checks[key+"boundary_digest"]=(launch_snapshot_digest(before)==item["measured_boundary_state_sha256"]
                and launch_snapshot_digest(after)==item["after_planning_state_sha256"]
                and sorted(before)==item["native_fields_captured"])
            sources={"qpos":raw["policy_qpos"][actor],"qvel":raw["policy_qvel"][actor],
                "ctrl":raw["policy_ctrl"][actor],"qacc_warmstart":raw["policy_qacc_warmstart"][actor],
                "qfrc_applied":raw["qfrc_applied"][launch_tick],"xfrc_applied":raw["xfrc_applied"][launch_tick],
                "previous_raw_action":raw["policy_previous_raw_action"][actor],
                "ticks":raw["policy_reference_tick"][actor],"reference_ids":raw["policy_reference_id"][actor],
                "reset_counts":raw["reset_count"][launch_tick],"support":raw["support_contact"][launch_tick],
                "time":raw["time"][launch_tick],"phase_denominators_s":raw["policy_phase_denominator_s"][actor]}
            checks[key+"measured_state_history"]=all(np.array_equal(before[k],value) for k,value in sources.items())
            checks[key+"counter_state"]=(before["physics_steps"]==launch_tick and before["actor_updates"]==actor
                and np.array_equal(before["reference_ids"],plan["reference_ids"])
                and np.array_equal(before["phase_denominators_s"],np.asarray(plan["phase_denominators_s"],np.float32)))
            checks[key+"reference_and_phase_immutable"]=bool(
                np.all(raw["policy_reference_id"][:,world]==plan["reference_ids"][world])
                and np.all(raw["policy_phase_denominator_s"][:,world]==np.float32(plan["phase_denominators_s"][world])))
            checks[key+"actuator_state_inventory"]=before["act"].shape==(worlds,0) and after["act"].shape==(worlds,0)
            clocks=[item[k] for k in ("planning_wall_s","target_callback_wall_s","cache_lookup_wall_s","simulation_pause_wall_s")]
            checks[key+"wall_clocks"]=(all(type(c) in (int,float) and np.isfinite(c) and c>=0 for c in clocks)
                and clocks[1]+clocks[2]<=clocks[0]+1.e-6 and clocks[3]==clocks[0]
                and item["simulated_time_advanced_s"]==0 and np.array_equal(before["time"],after["time"]))
            fresh=item["target_input_receipt"]
            target_ok=(all(fresh.get(k)==expected[k] for k in target_fields)
                and type(fresh.get("capture_episode_tick")) is int and fresh["capture_episode_tick"]==launch_tick
                and not isinstance(fresh.get("input_episode_time_s"),bool) and fresh.get("input_episode_time_s")==launch_tick*.005
                and fresh.get("source_kind")=="fixed_canonical_snapshot" and fresh.get("no_prediction") is True
                and fresh.get("head_geometry")=="head_collision" and fresh.get("head_velocity_method")=="jacobian_qvel")
            current=item["observed_current_request"]
            request_ok=isinstance(current,dict) and all(current.get(k)==expected[k] for k in fields)
            ready=bool(raw["policy_readiness"][actor,world])
            checks[key+"current_readiness"]=item["current_positional_support_ready"] is ready
            changed=launch_snapshot_digest(before)!=launch_snapshot_digest(after)
            reason=("physical_or_controller_state_changed_during_planning" if changed else
                "command_or_reference_identity_changed" if not request_ok else
                "launch_target_input_mismatch_or_stale" if not target_ok else
                "current_positional_support_readiness_failed" if not ready or not before["support"][world].all() else
                "final_planning_late" if clocks[0]>.020 else None)
            accepted=reason is None;accepted_by_world[world]=bool(item["accepted"])
            checks[key+"disposition"]=(item["accepted"] is accepted and item["reason"]==reason
                and item["reasons"]==([] if accepted else [reason])
                and item["actual_launch_tick"]==(launch_tick if accepted else None))
            checks[key+"recorded_launch_flags"]=(bool(raw["policy_launched"][actor,world])==accepted
                and bool(raw["policy_rejected"][actor,world])==(not accepted)
                and not bool(before["launched"][world]) and not bool(before["rejected"][world]))
            interval_after=np.arange(len(raw["reference_tick"]))>launch_tick
            checks[key+"immutable_disposition_through_recovery"]=(
                np.array_equal(raw["launched"][:,world],interval_after&accepted)
                and np.array_equal(raw["rejected"][:,world],interval_after&(not accepted)))
            expected_ticks=np.arange(len(raw["reference_tick"]),dtype=raw["reference_tick"].dtype)
            if not accepted:expected_ticks=np.minimum(expected_ticks,launch_tick)
            checks[key+"uninterrupted_reference_clock"]=np.array_equal(raw["reference_tick"][:,world],expected_ticks)
            checks[key+"scope"]=(item["same_continuous_cached_reference"] is True
                and item["exact_joint_return_required"] is False and item["quiet_hook_gate_added"] is False
                and item["live_g1_state_recaptured"] is False and item["full_saved_state_restore"] is False
                and item["hardware_real_time_qualified"] is False)
        return dict(passed=all(bool(value) for value in checks.values()),checks={k:bool(v) for k,v in checks.items()},
            accepted_by_world=accepted_by_world,decisions=len(decisions),snapshots=len(records),
            complete_saved_state_restore_audited=bool(shared and shared_initialization and shared_initialization.get('passed')),
            live_target_execution_audited=False)
    except (KeyError,IndexError,TypeError,ValueError,OverflowError) as error:
        return dict(passed=False,checks={k:bool(v) for k,v in checks.items()},accepted_by_world=accepted_by_world,
            reason="Malformed or inconsistent closed-protocol evidence",exception=dict(type=type(error).__name__,message=str(error)))


def controller_audit(raw, model, schema):
    n = len(raw["pre_qpos"])
    qadr, vadr = np.asarray(schema["qpos_addresses"]), np.asarray(schema["dof_addresses"])
    kp, kd = np.asarray(schema["kp"]), np.asarray(schema["kd"])
    caps = np.asarray(schema["torque_limits"])
    target = np.clip(raw["pre_qref"][..., qadr]+.25*np.clip(raw["pre_raw_action"], -1, 1),
        model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    drive = kp*target+kd*raw["pre_vref"][..., vadr]+raw["pre_feedforward"]
    force = np.clip(raw["pre_drive"]-kp*raw["pre_qpos"][..., qadr]-kd*raw["pre_qvel"][..., vadr], caps[:, 0], caps[:, 1])
    force_error = float(np.max(np.abs(force-raw["actuator_force"][1:]), initial=0))
    held = np.repeat(raw["policy_action"], 4, axis=0)[:n]
    checks = {
        "state_interval_pairing": np.array_equal(raw["pre_qpos"], raw["qpos"][:-1]) and np.array_equal(raw["pre_qvel"], raw["qvel"][:-1]),
        "physics_clock_5ms": np.allclose(np.diff(raw["time"], axis=0), .005, atol=2e-6, rtol=0),
        "reset_count_constant": np.all(raw["reset_count"] == raw["reset_count"][0]),
        "actor_every_four_steps": np.array_equal(raw["policy_episode_tick"], np.arange(len(raw["policy_action"]))*4),
        "raw_previous_action": np.all(raw["policy_previous_raw_action"][0] == 0) and np.array_equal(raw["policy_previous_raw_action"][1:], raw["policy_action"][:-1]),
        "residual_held_four_steps": np.array_equal(held, raw["pre_raw_action"]),
        "joint_target_law": np.allclose(target, raw["pre_target"], atol=2e-7, rtol=0),
        "preintegration_drive_law": np.allclose(drive, raw["pre_drive"], atol=2e-4, rtol=0),
        "ctrl_equals_drive": np.array_equal(raw["ctrl"][1:], raw["pre_drive"]),
        "force_law_interval_pairing": force_error <= 1e-3,
        "effort_caps": np.all(raw["actuator_force"] >= caps[:, 0]-1e-4) and np.all(raw["actuator_force"] <= caps[:, 1]+1e-4),
        "no_root_actuation": np.max(np.abs(raw["qfrc_actuator"][..., :6]), initial=0) <= 1e-9,
        "no_external_force": np.max(np.abs(raw["qfrc_applied"]), initial=0) <= 1e-9 and np.max(np.abs(raw["xfrc_applied"]), initial=0) <= 1e-9,
        "finite_physical_state": np.isfinite(raw["qpos"]).all() and np.isfinite(raw["qvel"]).all(),
        "no_contact_overflow": not np.any(raw["overflow"]),
    }
    return {"passed": all(bool(v) for v in checks.values()), "checks": {k: bool(v) for k,v in checks.items()},
            "maximum_force_error_nm": force_error}


def bound_json(record,*,inside=None):
    path=Path(record['path']).resolve(strict=True)
    if not path.is_relative_to(ROOT) or (inside is not None and not path.is_relative_to(inside)) or digest(path)!=record['sha256']:
        raise ValueError('Changed or unowned audit evidence')
    return path,json.loads(path.read_text())


def shared_initialization_audit(raw,receipt,plan,declaration,metadata,model):
    """Read-only full native bank/initial trace comparison, never restoration."""
    checks={};bindings={b['trial_id']:b for b in declaration['bindings']};trial_bindings={}
    try:
        from human_punching.phase2_native_state import load_snapshot,NativeComparisonSnapshot
        descriptor=metadata['native_start_state_bank'];path,_=bound_json(descriptor)
        bank=load_snapshot(path)
        checks['schema']=receipt['schema']=='task10b_shared_native_initialization_audit_v1'
        checks['bank_identity']=bank.identity==descriptor['identity'] and receipt['bank']==descriptor and len(bank.states)==4
        _,capture=bound_json(metadata['native_start_state_capture'])
        checks['counted_capture']=(capture.get('schema')=='task10b_phase2_native_shared_bank_capture_v1'
            and capture.get('complete_capture') is True and bool(capture.get('slurm_job_id')) and capture.get('exception') is None
            and capture.get('captured_native_states')==4 and capture.get('physics_steps_executed')==0
            and capture.get('actor_updates_executed')==0 and capture.get('no_actor_execution') is True
            and capture.get('split')==declaration['split'] and capture.get('native_start_state_bank')==descriptor)
        _,parity=bound_json(metadata['native_state_parity']);next_states=parity.get('receipt',{}).get('next_four_integration_states_equal')
        checks['counted_native_parity']=(parity.get('schema')=='task10b_phase2_native_state_parity_v1'
            and parity.get('passed') is True and bool(parity.get('slurm_job_id')) and parity.get('family')=='jab'
            and parity.get('exception') is None and parity.get('receipt',{}).get('observations_equal') is True
            and parity.get('receipt',{}).get('actor_outputs_equal') is True and next_states==[True]*4)
        archive=Path(receipt['retained_native_state']['path']).resolve(strict=True)
        checks['retained_state_owned']=archive.is_relative_to(ROOT) and digest(archive)==receipt['retained_native_state']['sha256']
        arrays=read_npz(archive);native=arrays['native_integration_state']
        checks['retained_state_inventory']=set(arrays)=={'native_integration_state'} and native.dtype==np.dtype('float64') and native.shape==(len(plan['trials']),410)
        checks['ordered_trials']=receipt['trial_ids']==plan['trials'] and len(set(plan['trials']))==len(plan['trials'])
        checks['native_schema']=int(mujoco.mjtState.mjSTATE_INTEGRATION)==16383 and mujoco.mj_stateSize(model,16383)==410
        restore=receipt['restore_receipt']
        checks['restore_scope']=(restore['complete_native_integration_equal'] is True
            and restore['physical_state_history_preserved'] is True and restore['selected_reference_is_explicit'] is True
            and restore['allowed_reference_metadata_changes']==['reference_id','ids_ref','phase_denominator_s']
            and restore['selected_reference_ids']==plan['reference_ids']
            and restore['selected_reference_sha256']==[bindings[t]['reference_sha256'] for t in plan['trials']]
            and restore['source_snapshot_identity']==NativeComparisonSnapshot(tuple(bank.states[bindings[t]['start_index']] for t in plan['trials']),bank.inventory_json).identity)
        data=mujoco.MjData(model)
        for world,trial_id in enumerate(plan['trials']):
            prefix=f'world_{world}_';binding=bindings[trial_id];index=binding['start_index'];state=bank.states[index]
            state.validate_model_size(model);trial_bindings[trial_id]=state.digest()
            checks[prefix+'bank_binding']=(receipt['start_indices'][world]==index
                and receipt['start_state_sha256'][world]==binding['start_state_sha256']==state.digest())
            checks[prefix+'complete410_equal']=np.array_equal(native[world],state.native_integration_state)
            mujoco.mj_setState(model,data,native[world],16383)
            for field in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied'):
                checks[prefix+'raw_initial_'+field]=np.array_equal(raw[field][0,world],getattr(data,field))
            checks[prefix+'time']=float(raw['time'][0,world])==data.time==state.support_interval_end_s==0.
            checks[prefix+'support']=np.array_equal(raw['support_contact'][0,world],state.support_contact)
            checks[prefix+'history']=(np.array_equal(raw['policy_previous_raw_action'][0,world],state.previous_raw_action)
                and not np.any(state.previous_raw_action) and not np.any(state.held_raw_action))
            checks[prefix+'initial_controller']=(state.reference_clock_s==0 and state.reference_phase=='preparation'
                and state.actor_update_index==state.decimation_phase==0 and state.adapter_inventory_complete is True
                and int(state.adapter_state['ticks'])==0 and int(state.adapter_state['reset_counts'])==1
                and not bool(state.adapter_state['launched']) and not bool(state.adapter_state['rejected'])
                and all(int(state.adapter_state[name])==0 for name in ('completed','fallen','physics_steps_executed','actor_updates_executed'))
                and json.loads(state.adapter_state['hard_faults_json'].tobytes())==[])
            checks[prefix+'trace_controller']=(np.all(raw['reset_count'][:,world]==1)
                and int(raw['reference_tick'][0,world])==0 and int(raw['policy_reference_tick'][0,world])==0
                and int(raw['policy_reference_id'][0,world])==plan['reference_ids'][world]
                and float(raw['policy_phase_denominator_s'][0,world])==np.float32(plan['phase_denominators_s'][world]))
            # The initial native vector is captured before the inherited
            # actor-boundary sim.forward. Its post-forward warmstart is a
            # separately recorded controller input, never claimed identical.
            checks[prefix+'first_actor_current_state']=(np.array_equal(raw['policy_qpos'][0,world],data.qpos)
                and np.array_equal(raw['policy_qvel'][0,world],data.qvel) and np.array_equal(raw['policy_ctrl'][0,world],data.ctrl)
                and np.isfinite(raw['policy_qacc_warmstart'][0,world]).all())
        return dict(passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},
            trial_state_bindings=trial_bindings,bank=descriptor,retained_native_state=receipt['retained_native_state'],
            full410_initial_state_audited=True,first_actor_warmstart_scope='Recorded after explicit actor-boundary forward; original captured warmstart remains bound in the complete initial410vector',
            no_integration_or_restore_performed=True)
    except (OSError,KeyError,TypeError,ValueError,IndexError) as error:
        return dict(passed=False,checks={k:bool(v) for k,v in checks.items()},trial_state_bindings=trial_bindings,
            exception=dict(type=type(error).__name__,message=str(error)))


def measure(model, q, v, nominal_qpos, support):
    """Fresh FK and point Jacobians; operational predicate constants remain fixed."""
    data = mujoco.MjData(model)
    chest, root = model.body("Chest").id, model.body("Hips").id
    hands = [model.site(side+"Hand_center").id for side in ("Left", "Right")]
    data.qpos[:] = nominal_qpos
    mujoco.mj_kinematics(model, data)
    nominal_center = (data.site_xpos[hands]-data.xpos[chest])@data.xmat[chest].reshape(3, 3)
    nominal_height = float(data.xpos[root, 2])
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    gloves, position_error, relative_speed, upright = [], [], [], []
    local_up = None
    for pose, velocity in zip(q, v, strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        R = data.xmat[chest].reshape(3, 3)
        root_R = data.xmat[root].reshape(3, 3)
        if local_up is None:
            local_up = root_R.T@np.array([0., 0., 1.])
        upright.append(bool(data.xpos[root, 2] >= .75*nominal_height and (root_R@local_up)[2] >= np.cos(.60)))
        mujoco.mj_jacBody(model, data, jp, jr, chest)
        chest_v, omega = jp@velocity, jr@velocity
        offsets = data.site_xpos[hands]-data.xpos[chest]
        position_error.append(np.linalg.norm(offsets@R-nominal_center, axis=1))
        speeds = []
        for hand, site in enumerate(hands):
            mujoco.mj_jacSite(model, data, jp, jr, site)
            speeds.append(np.linalg.norm(R.T@(jp@velocity-chest_v-np.cross(omega, offsets[hand]))))
        relative_speed.append(speeds)
        gloves.append(data.site_xpos[hands].copy())
    finite = np.isfinite(q).all(axis=1)&np.isfinite(v).all(axis=1)
    position_ok = np.all(np.asarray(position_error) <= .05, axis=1)
    upright = np.asarray(upright)
    pose_ok = (finite&upright&position_ok&(np.linalg.norm(v[:, :3], axis=1) <= .05)
        &(np.linalg.norm(v[:, 3:6], axis=1) <= .12)&np.all(np.asarray(relative_speed) <= .15, axis=1))
    return {"gloves": np.asarray(gloves), "pose_ok": pose_ok,
        "ready": finite&upright&position_ok&support.all(axis=1),
        "upright": upright, "finite": finite}


def offline_rejection_audit(binding,row,mask,mask_sha256,catalog_sha256,model):
    """Bind a zero-attempt rejection to actual immutable files and exact request.

    Screening is evidence of the stated exclusion, never native qualification.
    The caller saves the returned receipt separately and binds its bytes in the
    trial resolution. Failed generation and generated-but-rejected paths remain
    distinct kinds of evidence.
    """
    checks={};evidence={}
    try:
        checks['physical_reference_binding']=(binding['reference_sha256']==row['reference_sha256']
            and binding['ttc_s']==row['requested_ttc_s'] and binding['command_id']==row['command_id']
            and binding['original_replica_id']==row['original_replica_id']
            and binding['target_w']==row['target_w'])
        if row.get('reference_identity_kind')=='generation_failure_receipt':
            path=Path(row['generation_failure_path']).resolve(strict=True)
            checks['generation_failure_file']=(path.is_relative_to(ROOT) and digest(path)==binding['reference_sha256']==row['generation_failure_sha256']
                and row.get('kinematic_admission') is False and bool(row.get('rejection_reasons'))
                and not any(name in row for name in ('reference_path','legacy_runtime_path','legacy_runtime_sha256')))
            failure=json.loads(path.read_text())
            checks['generation_failure_content']=all(failure.get(k)==row.get(k) for k in
                ('candidate_id','family','command_id','target_w','requested_ttc_s','recipe_sha256','rejection_reasons'))
            wrapper_fields={'runtime_storage_version','runtime_failure_wrapper_version','unpacked_recipe_sha256',
                'unpacked_generation_failure_path','unpacked_generation_failure_sha256'}
            if wrapper_fields.intersection(failure) or wrapper_fields.intersection(row):
                # Packing changes the declared recipe. Its new negative-cell
                # receipt must preserve the original failure byte-for-byte in
                # a separate artifact, and copy every original field except
                # that explicitly versioned recipe identity into the wrapper.
                # Never accept a mismatched old recipe by skipping its check.
                original_path=Path(failure['unpacked_generation_failure_path']).resolve(strict=True)
                original_sha=failure['unpacked_generation_failure_sha256']
                checks['packed_failure_owned_original']=(original_path.is_relative_to(ROOT) and original_path!=path
                    and digest(original_path)==original_sha
                    and str(original_path)==str(Path(row['unpacked_generation_failure_path']).resolve(strict=True))
                    and original_sha==row['unpacked_generation_failure_sha256'])
                if not checks['packed_failure_owned_original']:raise ValueError('Packed failure original receipt changed or escaped ownership')
                original=json.loads(original_path.read_text())
                checks['packed_failure_exact_inventory']=(not wrapper_fields.intersection(original)
                    and set(failure)==set(original)|wrapper_fields
                    and not any(key in original for key in ('reference_path','legacy_runtime_path','legacy_runtime_sha256','generation_failure_path')))
                checks['packed_failure_original_fields']=(all(failure.get(key)==value for key,value in original.items() if key!='recipe_sha256')
                    and all(row.get(key)==value for key,value in failure.items()))
                checks['packed_failure_recipe_ancestry']=(failure['runtime_storage_version']==row['runtime_storage_version']=='task10b_float32_native_runtime_pool_v1'
                    and failure['runtime_failure_wrapper_version']==row['runtime_failure_wrapper_version']=='task10b_phase2_packed_generation_failure_v1'
                    and failure['unpacked_recipe_sha256']==row['unpacked_recipe_sha256']==original['recipe_sha256']
                    and failure['recipe_sha256']==row['recipe_sha256']
                    and failure['recipe_sha256']!=original['recipe_sha256'])
                evidence['unpacked_generation_failure']=dict(path=str(original_path),sha256=original_sha)
            return dict(passed=all(checks.values()),checks=checks,kind='generation_failure',
                generation_failure_path=str(path),generation_failure_sha256=digest(path),evidence=evidence)
        if mask is None:raise ValueError('Hash-bound pre-outcome mask required for screen/mask rejection')
        physical={k:v for k,v in binding.items() if k!='trial_id'}
        matches=[m for m in mask['requests'] if all(m.get(k)==v for k,v in physical.items()) and m.get('family')==row['family']]
        checks['exact_mask_request']=mask.get('schema')=='task10b_phase2_declared_evaluation_mask_v1' and len(matches)==1 and type(matches[0].get('admitted')) is bool
        if not checks['exact_mask_request']:raise ValueError('Rejected mask request differs from exact declaration')
        excluded=matches[0]['admitted'] is False
        checks['explicit_rejection']=(excluded and isinstance(matches[0].get('reason'),str) and bool(matches[0]['reason'].strip())) or (
            row.get('kinematic_admission') is False and bool(row.get('rejection_reasons')))
        for name in ('reference','legacy_runtime'):
            path=Path(row[name+'_path']).resolve(strict=True)
            checks[name+'_hash']=path.is_relative_to(ROOT) and digest(path)==row[name+'_sha256']
            evidence[name]=dict(path=str(path),sha256=digest(path))
        screen=Path(row['diagnostic_screen_path']).resolve(strict=True)
        checks['screen_owned']=screen.is_relative_to(ROOT) and screen.is_file()
        screen_data=json.loads(screen.read_text())
        checks['screen_finite_summary']=(type(screen_data.get('frames')) is int and screen_data['frames']>0
            and type(screen_data.get('completed_chunks')) is int and screen_data['completed_chunks']>0)
        evidence['diagnostic_screen']=dict(path=str(screen),sha256=digest(screen))
        # Independently reconstruct first crossings to verify every generated
        # kinematic rejection reason. A stricter immutable mask is also legal.
        if row.get('kinematic_admission') is False:
            reference=reference_arrays(row['reference_path'],model);active=row['active_hand_index']
            first=sphere_crossing(reference['t'],reference['glove_pos'][:,active],row['target_w'])
            other=sphere_crossing(reference['t'],reference['glove_pos'][:,1-active],row['target_w'])
            launch=row['launch_tick']*.005;outward=row['outward_end_tick']*.005
            timing=(first is not None and max(.5,row['requested_ttc_s']-.020)-1e-9<=first-launch<=row['requested_ttc_s']+.020+1e-9
                and launch<=first<=outward)
            wrong=other is not None and (first is None or other<=first)
            before=any(c is not None and c<launch for c in (first,other))
            known={'retargeted_first_contact_outside_requested_timing':not timing,
                'shared_first_contact_outside_requested_timing':not timing,'wrong_hand_first_or_simultaneous':wrong,
                'contact_before_launch':before,'contact_before_outward_launch_including_preparation':before}
            checks['stated_kinematic_reasons']=all(known.get(reason) is True for reason in row['rejection_reasons'])
            evidence['independent_reference_crossings']=dict(first_active_contact_s=first,first_inactive_contact_s=other)
        return dict(schema='task10b_phase2_independent_offline_rejection_v1',passed=all(bool(v) for v in checks.values()),
            checks={k:bool(v) for k,v in checks.items()},kind='screen_or_mask_rejection',trial_id=binding['trial_id'],
            request_identity=json_identity(binding),reference_sha256=binding['reference_sha256'],
            declared_mask_sha256=mask_sha256,catalog_sha256=catalog_sha256,evidence=evidence,
            no_physical_attempt=True,cycle_success_credit=0)
    except (OSError,KeyError,ValueError,TypeError,IndexError) as error:
        return dict(passed=False,checks={k:bool(v) for k,v in checks.items()},evidence=evidence,
            exception=dict(type=type(error).__name__,message=str(error)))


def native_resolution(binding,trial,rawhash,*,integrity,exact_counters,safe_boundary):
    known=bool(integrity and exact_counters and safe_boundary and trial['hard_flags_absent'])
    return dict(trial_id=trial['trial_id'],request_identity=json_identity(binding),
        outcome_resolution=('audited_complete' if trial['execution_complete'] else 'audited_failed_prefix') if known else 'unknown_or_missing',
        resolved=known,attempted=True,execution_complete=trial['execution_complete'],raw_trace_sha256=rawhash,
        prefix_integrity_pass=bool(integrity),exact_counter_pass=bool(exact_counters),safe_actor_boundary_pass=bool(safe_boundary),
        hard_faults_absent=bool(trial['hard_flags_absent']))


def audit_run(run_path, output_path):
    run=Path(run_path).resolve(strict=True);output=Path(output_path).resolve()
    if not run.is_relative_to(ROOT/'outputs') or not output.is_relative_to(ROOT/'outputs') or output.exists():
        raise ValueError('Choose an owned run and a fresh human-owned outputs directory')
    launch=json.loads((run/'launch.json').read_text())
    expanded=launch.get('schema') in ('task10b_phase2_expanded_native_launch_v1','task10b_phase2_grouped_validation_launch_v1')
    template=launch.get('checkpoint_template_is_unbound') is True
    catalog_path=Path(launch['catalog']).resolve(strict=True)
    if not catalog_path.is_relative_to(ROOT) or digest(catalog_path)!=launch['catalog_sha256']:
        raise ValueError('Committed reference catalog changed')
    catalog=json.loads(catalog_path.read_text());cells={r['candidate_id']:r for r in catalog['cells']}
    if len(cells)!=len(catalog['cells']):raise ValueError('Duplicate candidate identities')
    expanded_inputs={};bindings={};mask=None;manifest=None;ancestry=None
    if expanded:
        for name in ('declaration','domain','target_geometry'):
            descriptor=(launch.get('physical_declaration') if name=='declaration' else launch.get(name)) if template else None
            if descriptor is None:descriptor=dict(path=launch[name+'_path'],sha256=launch[name+'_sha256'])
            _,expanded_inputs[name]=bound_json(descriptor)
        declaration=expanded_inputs['declaration'];bindings={b['trial_id']:b for b in declaration['bindings']}
        expected_identity=launch['physical_request_template_identity'] if template else launch['declaration_identity']
        if json_identity(declaration)!=expected_identity or len(bindings)!=len(declaration['bindings']):
            raise ValueError('Expanded declaration identity or unique trial set differs')
        checkpoint=launch['evaluated_checkpoint'];path=Path(checkpoint['path']).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path)!=checkpoint['sha256']:raise ValueError('Evaluated checkpoint changed')
        if template:
            _,configuration=bound_json(launch['configuration']);_,manifest=bound_json(configuration['manifest'])
            mask_path=Path(configuration['mask']['path']).resolve(strict=True)
            mask_descriptor=configuration['mask']
            predicate_descriptor=configuration['predicate']
        else:
            identity=json.loads((run/'execution_identity.json').read_text())
            if json_identity(identity['frozen'])!=identity['identity'] or identity['identity']!=launch['execution_identity']:
                raise ValueError('Frozen execution identity differs')
            mask_path=Path(identity['input_paths']['mask']).resolve(strict=True)
            mask_descriptor=dict(path=str(mask_path),sha256=identity['frozen']['input_sha256']['mask'])
            predicate_descriptor=dict(path=identity['input_paths']['predicate'],sha256=identity['frozen']['input_sha256']['predicate'])
            manifest_path=Path(identity['input_paths']['manifest']).resolve(strict=True)
            _,manifest=bound_json(dict(path=str(manifest_path),sha256=identity['frozen']['input_sha256']['manifest']))
        _,mask=bound_json(mask_descriptor)
        if digest(mask_path)!=declaration['declared_mask_sha256']:raise ValueError('Declared immutable mask differs')
        predicate_path,_=bound_json(predicate_descriptor)
        if digest(predicate_path)!=declaration['predicate_sha256']:raise ValueError('Reviewed functional predicate differs')
    roots=[run]
    if expanded and not template:
        from human_punching.phase2_continuation import verified_run_ancestry
        ancestry=verified_run_ancestry(run);roots=[Path(p) for p in ancestry['run_directories']]
    intended=set()
    if expanded and not template:
        for parent in roots:
            for path in sorted(parent.glob('native_attempt_intent_*.json')):
                intent=json.loads(path.read_text());ids=intent['trial_ids']
                if (intent.get('schema')!='task10b_phase2_native_attempt_intent_v1'
                        or intent.get('execution_identity')!=launch['execution_identity']
                        or len(ids)!=len(set(ids)) or not set(ids)<=set(bindings) or intended.intersection(ids)
                        or intent.get('request_identities')!=[json_identity(bindings[t]) for t in ids]):
                    raise ValueError('Duplicate or inconsistent durable native attempt intent')
                intended.update(ids)
    schema=json.loads(Path(launch['release_inputs']['schema']['path']).read_text())
    for descriptor in launch['release_inputs'].values():
        path=Path(descriptor['path']).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path)!=descriptor['sha256']:
            raise ValueError('Accepted release input changed or escaped human-punching')
    source_model=mujoco.MjModel.from_xml_path(launch['release_inputs']['model']['path'])
    runtime_models=[parent/'runtime.xml' for parent in roots if (parent/'runtime.xml').exists()]
    if len({digest(path) for path in runtime_models})>1:raise ValueError('Continuation runtime model bytes differ')
    used_model=runtime_models[0] if runtime_models else Path(launch['release_inputs']['model']['path'])
    native_runtime_present=bool(runtime_models)
    model=mujoco.MjModel.from_xml_path(str(used_model)) if native_runtime_present else source_model
    physical_fields=('body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat','jnt_range','jnt_pos','jnt_axis',
        'geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction','dof_armature','dof_damping')
    body_unchanged=all(np.array_equal(getattr(model,k),getattr(source_model,k)) for k in physical_fields)
    runtime_checks=dict(physics_dt=bool(np.isclose(model.opt.timestep,.005,atol=1e-12)),
        accepted_solver=bool(int(model.opt.cone)==0 and int(model.opt.iterations)==10 and int(model.opt.ls_iterations)==20
            and np.isclose(model.opt.tolerance,1e-8,atol=1e-15) and model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            and model.opt.solver==mujoco.mjtSolver.mjSOL_NEWTON),
        unchanged_effort_limits=np.array_equal(model.actuator_forcerange,np.asarray(schema['torque_limits'])),
        native_affine_bias=bool(np.all(model.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE)),
        unit_hinge_actuators=bool(np.array_equal(model.actuator_trnid[:,0],np.arange(1,49)) and np.all(model.actuator_gear[:,0]==1)))
    if not native_runtime_present:
        # Wholly offline rejected panels have no exported native model. Missing
        # traces of an actual intended attempt still remain unresolved below.
        runtime_checks=dict(no_native_runtime_or_physics_qualified=True)
    nominal=read_npz(launch['release_inputs']['reference']['path'])['qpos'][0]
    output.mkdir(parents=True,exist_ok=False);panels=[];resolved={};checkpoint_cache={};seen_trials=set()
    paths=[Path(launch['release_inputs']['checkpoint']['path'])]
    for parent in roots:paths.extend(sorted((parent/'training').glob('checkpoint_*.pt')))
    trace_paths=sorted({p.resolve() for parent in roots for p in parent.glob('batch_*/raw_trace.npz')}) if expanded else sorted(run.rglob('raw_trace.npz'))
    if trace_paths and not native_runtime_present:raise ValueError('Retained native evidence has no verified runtime model')
    for trace_path in trace_paths:
        folder=trace_path.parent;raw=read_npz(trace_path)
        if not (folder/'plan.json').exists() or not (folder/'summary.json').exists():
            panels.append(dict(path=str(folder),raw_sha256=digest(trace_path),passed=False,execution_complete=False,
                planned_trials=None,trials=[],audited_completed_trials=0,recorded_status='missing_plan_or_summary',
                reason='Native attempt extent cannot be resolved without exact independent counters and world bindings'))
            continue
        plan=json.loads((folder/'plan.json').read_text())
        recorded=json.loads((folder/'summary.json').read_text());rawhash=digest(trace_path)
        if expanded and (not set(plan['trials'])<=set(bindings) or seen_trials.intersection(plan['trials'])):
            raise ValueError('An attempted request was duplicated or escaped the immutable declaration')
        if expanded and not template and not set(plan['trials'])<=intended:
            raise ValueError('Native trace lacks its durable immutable attempt intent')
        seen_trials.update(plan['trials'])
        n=len(raw.get('pre_qpos',()));actors=len(raw.get('policy_action',()))
        if not n or not actors:
            panels.append(dict(path=str(folder),raw_sha256=rawhash,passed=False,execution_complete=False,
                planned_trials=recorded['planned_trials'],trials=[],audited_completed_trials=0,recorded_status=recorded['status'],
                reason='No complete action/physics interval; extent unresolved'))
            continue
        controller=controller_audit(raw,model,schema)
        counter_checks=dict(actual_physics_steps=recorded['actual_physics_steps']==n,
            actual_policy_updates=recorded['actual_policy_updates']==actors,frames=len(raw['qpos'])==n+1,
            planned_trials=recorded['planned_trials']==len(plan['trials'])==len(plan['candidates']),
            executed_reports=recorded['executed_trials']==len(recorded['trials']),
            incomplete_reports=recorded['incomplete_trials']==recorded['planned_trials']-len(recorded['trials']))
        count_checks=all(counter_checks.values());safe_boundary=n%4==0 and actors==n//4
        closed=None;initialization=None
        if expanded:
            try:
                metadata=launch.get('validation_protocol') if template else launch.get('protocol_metadata')
                if declaration.get('start_mode')=='shared_jab_saved_state':
                    descriptor=json.loads((folder/'restored_native_initialization.json').read_text())
                    archive=Path(descriptor['retained_native_state']['path']).resolve(strict=True)
                    if not archive.is_relative_to(folder):raise ValueError('Shared initial archive escaped its batch')
                    initialization=shared_initialization_audit(raw,descriptor,plan,declaration,metadata,model)
                receipt=json.loads((folder/'closed_protocol_launch.json').read_text())
                archive=Path(receipt['boundary_snapshots']['path']).resolve(strict=True)
                if not archive.is_relative_to(folder) or digest(archive)!=receipt['boundary_snapshots']['sha256']:
                    raise ValueError('Closed boundary archive changed or escaped its batch')
                closed=closed_launch_audit(raw,receipt,plan,cells,launch,declaration,expanded_inputs['domain'],
                    expanded_inputs['target_geometry'],read_npz(archive),shared_initialization=initialization)
            except (OSError,KeyError,ValueError,TypeError) as error:
                closed=dict(passed=False,accepted_by_world={},reason='Missing or invalid closed-protocol evidence',
                    exception=dict(type=type(error).__name__,message=str(error)))
        commitments=commitment_audit(raw,json.loads((folder/'launch_commitments.json').read_text())['commitments'],plan,cells,
            accepted_by_world=closed['accepted_by_world'] if closed is not None else None)
        if plan['actor_sha256'] not in checkpoint_cache:
            checkpoint_cache[plan['actor_sha256']]=matching_checkpoint(paths,plan['actor_sha256'])
        checkpoint,policy_state=checkpoint_cache[plan['actor_sha256']]
        if closed is not None:
            closed.setdefault('checks',{})['evaluated_checkpoint_bytes']=digest(checkpoint)==launch['evaluated_checkpoint']['sha256']
            closed['passed']=bool(closed['passed'] and closed['checks']['evaluated_checkpoint_bytes'])
        normal=decode_normal_forces(raw,pyramidal=int(model.opt.cone)==0)
        force_error=float(np.max(np.abs(normal-raw['contact_force'][:,0]),initial=0));contact_equal=force_error<=1e-3
        support=independent_support(raw,model.geom('floor').id,[model.geom('physical/'+side+'Foot').id for side in ('Left','Right')],
            pyramidal=int(model.opt.cone)==0)
        support_equal=np.array_equal(support,raw['support_contact'])
        reports={r['trial_id']:r for r in recorded['trials']};trials=[]
        if len(reports)!=len(recorded['trials']) or not set(reports)<=set(plan['trials']):raise ValueError('Duplicate or undeclared native report')
        for world,trial_id in enumerate(plan['trials']):
            row=cells[plan['candidates'][world]];reported=reports.get(trial_id)
            for prefix in ('reference','legacy_runtime'):
                if digest(row[prefix+'_path'])!=row[prefix+'_sha256']:raise ValueError('Committed reference changed')
            steps=(int(np.ceil(row['episode_duration_s']/.005-1e-10)) if row['kind']=='original_control'
                else int(np.ceil(row['episode_duration_s']/.020-1e-10))*4)
            complete=n>=steps;q=raw['qpos'][:steps+1,world];v=raw['qvel'][:steps+1,world];t=np.arange(len(q))*.005
            m=measure(model,q,v,nominal,support[:len(q),world]);launch_tick=row['launch_tick'];outward_tick=row['outward_end_tick']
            reached=launch_tick<len(q) and launch_tick//4<actors
            ready=bool(m['ready'][launch_tick]) if reached else False
            accepted=bool(reached and ready and (closed is None or closed['accepted_by_world'].get(world,False)))
            launch_s=launch_tick*.005 if accepted else None
            dwell=interval_dwell(t,m['pose_ok'],support[:len(q),world].all(axis=1),outward_tick)
            recovered=np.flatnonzero(dwell>=.20-1e-10) if accepted else np.empty(0,int)
            recovery=float(t[recovered[0]]) if len(recovered) else None
            active=row['active_hand_index'];first=sphere_crossing(t,m['gloves'][:,active],row['target_w'])
            other=sphere_crossing(t,m['gloves'][:,1-active],row['target_w']);contacts=[c for c in (first,other) if c is not None]
            first_any=min(contacts) if contacts else None
            reference=reference_arrays(row['legacy_runtime_path'],model)
            integer=bool(expanded and declaration.get('start_mode')=='shared_jab_saved_state' and row['kind']!='original_control')
            actor_audit=sampled_actor_audit(model,raw,row,world,reference,checkpoint,policy_state,integer=integer)
            ticks=raw['pre_reference_tick'][:,world];queries=ticks.astype(np.float32)*np.float32(.005);reference_checks={}
            for key,saved in (('qpos','pre_qref'),('qvel','pre_vref'),('torque','pre_feedforward')):
                predicted=(integer_sample(reference[key],ticks,quaternion=key=='qpos') if integer
                    else interpolate(reference['t'],reference[key],queries,quaternion=key=='qpos'))
                reference_checks[key]=bool(np.allclose(predicted,raw[saved][:,world],atol=2e-6,rtol=0))
            desired=(integer_sample(reference['glove_pos'],np.arange(len(t))) if integer
                else interpolate(reference['t'],reference['glove_pos'],t))
            square=np.sum((m['gloves'][:,active]-desired[:,active])**2,axis=1)
            outward=(t>=launch_tick*.005)&(t<=outward_tick*.005)
            rms=float(np.sqrt(np.mean(square)));out_rms=float(np.sqrt(np.mean(square[outward]))) if outward.any() else None
            limits=np.maximum(model.jnt_range[1:,0]-q[:,7:],q[:,7:]-model.jnt_range[1:,1]);excursion=float(np.max(limits,initial=0))
            in_outward=first is not None and launch_tick*.005<=first<=outward_tick*.005
            if row['kind']=='original_control':
                expected=sphere_crossing(reference['t'],reference['glove_pos'][:,active],row['target_w'])
                timing=first is not None and expected is not None and abs(first-expected)<=.020+1e-9
            else:
                actual=None if first is None or launch_s is None else first-launch_s
                timing=actual is not None and max(.50,row['requested_ttc_s']-.020)-1e-9<=actual<=row['requested_ttc_s']+.020+1e-9
            functional=bool(accepted and recovery is not None and m['upright'].all() and m['finite'].all())
            numeric=bool(complete and functional and timing and in_outward and not(other is not None and(first is None or other<=first))
                and rms<=.02 and out_rms is not None and out_rms<=.02 and excursion<=.02 and controller['passed']
                and support_equal and contact_equal and all(reference_checks.values()))
            checks=dict(reference_immutable_samples=all(reference_checks.values()),sampled_original369_actor=actor_audit['passed'],
                recorded_launch=accepted==bool(raw['launched'][-1,world]))
            if expanded:
                checks.update(closed_protocol_launch=closed['passed'],current_positional_support_readiness=(not reached or ready==bool(raw['policy_readiness'][launch_tick//4,world])))
            def same(a,b,tolerance=2e-7):
                return a is b if a is None or b is None else bool(abs(a-b)<=tolerance)
            if reported is not None:
                checks.update(numeric_outcome=numeric==reported['numeric_complete_cycle'],first_active_contact=same(first,reported['first_active_contact_s']),
                    first_inactive_contact=same(other,reported['first_inactive_contact_s']),functional_recovery=same(recovery,reported['functional']['recovery_time_s']),
                    functional_complete=functional==reported['functional']['functional_complete'],full_rms=same(rms,reported['full_active_glove_rms_m']),
                    outward_rms=same(out_rms,reported['outward_active_glove_rms_m']),joint_excursion=same(excursion,reported['maximum_joint_limit_excursion_rad']),
                    frozen_target=not accepted or np.array_equal(reported['target_at_launch'],row['target_w']))
            elif complete:checks['full_horizon_report_present']=False
            fallen=not bool(m['upright'].all());nonfinite=not bool(m['finite'].all())
            root_assistance=not(controller['checks']['no_root_actuation'] and controller['checks']['no_external_force'])
            effort_violation=not controller['checks']['effort_caps']
            hard=dict(no_falls=not fallen,finite=not nonfinite,no_root_assistance=not root_assistance,no_effort_violation=not effort_violation,
                observed_joint_limit=excursion<=.02,observed_tracking=rms<=.02 and(out_rms is None or out_rms<=.02),
                early_contact_floor=first_any is None or(launch_s is not None and first_any-launch_s>=.50-1e-10))
            # Historical D0 original controls retain original schedule diagnostics;
            # expanded physical floor/limits are panel-wide additional gates.
            hard_absent=all(hard.values()) if expanded else not(fallen or nonfinite or root_assistance or effort_violation)
            trial=dict(trial_id=trial_id,passed=all(checks.values()),checks=checks,execution_complete=complete,
                numeric_complete_cycle=numeric,first_active_contact_s=first,first_inactive_contact_s=other,first_any_contact_s=first_any,
                recovery_time_s=recovery,launch_s=launch_s,joint_limit_excursion_rad=excursion,
                fallen=fallen,nonfinite=nonfinite,root_assistance=root_assistance,effort_cap_violation=effort_violation,
                observed_active_glove_rms_m=rms,observed_outward_active_glove_rms_m=out_rms,
                observed_physical_metrics=dict(execution_complete=complete,scope='full_cycle' if complete else 'retained_prefix',
                    active_squared_error_sum_m2=float(square.sum()),active_sample_count=len(square),
                    outward_active_squared_error_sum_m2=float(square[outward].sum()),outward_active_sample_count=int(outward.sum()),observed_end_s=float(t[-1])),
                hard_flags_absent=hard_absent,hard_flag_checks=hard,reference_checks=reference_checks,sampled_actor=actor_audit,
                readiness_pass=accepted,immutable_launch_pass=accepted and(closed is None or closed['passed']),
                continuity_history_pass=controller['passed'] and commitments['passed'] and(closed is None or closed['passed']),
                functional_recovery_pass=functional,correct_hand=first is not None and(other is None or first<other),
                request_s=0.,preparation_complete_s=row.get('preparation_complete_tick',0)*.005,
                readiness_s=launch_s,first_positional_support_readiness_s=float(t[np.flatnonzero(m['ready'])[0]]) if m['ready'].any() else None,
                final_planning_s=launch_s,outward_complete_s=outward_tick*.005 if accepted and outward_tick<len(t) else None,
                recovery_s=recovery,complete_cycle_qualified=False)
            if expanded:
                trial.update(request_identity=json_identity(bindings[trial_id]),checkpoint_sha256=launch['evaluated_checkpoint']['sha256'],
                    predicate_sha256=declaration['predicate_sha256'])
            if complete:trial.update(full_rms_m=rms,outward_rms_m=out_rms,full_active_glove_rms_m=rms,outward_active_glove_rms_m=out_rms)
            trials.append(trial)
        complete=all(r['execution_complete'] for r in trials) and n==plan['planned_physics_steps']
        integrity=bool(count_checks and safe_boundary and commitments['passed'] and controller['passed'] and support_equal and contact_equal
            and all(r['passed'] for r in trials) and(closed is None or closed['passed']))
        panel=dict(path=str(folder),raw_sha256=rawhash,controller=controller,launch_commitments=commitments,
            count_checks=count_checks,counter_checks=counter_checks,safe_actor_boundary_pass=safe_boundary,execution_complete=complete,
            native_contact_normal_decoding_matches=contact_equal,maximum_contact_normal_force_error_n=force_error,support_matches=support_equal,
            recorded_status=recorded['status'],planned_trials=recorded['planned_trials'],audited_completed_trials=sum(t['execution_complete'] for t in trials),
            trials=trials,prefix_integrity_pass=integrity,passed=bool(integrity and all(t['hard_flags_absent'] for t in trials) and(expanded or complete)))
        if expanded:panel['closed_protocol_launch']=closed
        if initialization is not None:panel['shared_saved_state_initialization']=initialization
        panels.append(panel)
        if expanded:
            for trial in trials:
                trial_id=trial['trial_id']
                resolved[trial_id]=native_resolution(bindings[trial_id],trial,rawhash,integrity=integrity,
                    exact_counters=count_checks,safe_boundary=safe_boundary)
    offline=[]
    if expanded:
        outcomes_path=run/'outcomes.json'
        if outcomes_path.exists():outcomes=json.loads(outcomes_path.read_text())
        elif (run/'numeric_ledger.json').exists():outcomes=json.loads((run/'numeric_ledger.json').read_text())
        elif template and (run/'summary.json').exists():outcomes=json.loads((run/'summary.json').read_text())['trials']
        else:outcomes=[]
        outcomes={r['trial_id']:r for r in outcomes};entries={e['trial_id']:e for e in manifest['trials']}
        for trial_id,binding in bindings.items():
            if trial_id in resolved:continue
            supplied=outcomes.get(trial_id,{})
            attempted=(supplied.get('physical_execution_attempted',supplied.get('physical_attempt')) is True
                or trial_id in seen_trials or trial_id in intended)
            resolution=dict(trial_id=trial_id,request_identity=json_identity(binding),outcome_resolution='unknown_or_missing',
                resolved=False,attempted=attempted,execution_complete=False,raw_trace_sha256=None,prefix_integrity_pass=False,
                exact_counter_pass=False,safe_actor_boundary_pass=False,hard_faults_absent=False)
            status=supplied.get('status',supplied.get('disposition'))
            if not attempted and status in ('rejected','generation_failure'):
                row=cells[entries[trial_id]['candidate_id']]
                evidence=offline_rejection_audit(binding,row,mask,digest(mask_path),launch['catalog_sha256'],model)
                path=output/('offline_rejection_'+hashlib.sha256(trial_id.encode()).hexdigest()[:20]+'.json')
                path.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n');offline.append(dict(path=str(path),sha256=digest(path),trial_id=trial_id))
                if evidence['passed']:
                    resolution.update(outcome_resolution='verified_offline_rejection',resolved=True)
                    if evidence['kind']=='generation_failure':resolution.update(generation_failure_verified=True,
                        generation_failure_sha256=evidence['generation_failure_sha256'])
                    else:resolution.update(offline_rejection_kind='screen_or_mask_rejection',rejected_request_verified=True,
                        rejection_evidence_sha256=digest(path),reference_sha256=binding['reference_sha256'],
                        declared_mask_sha256=digest(mask_path),declared_mask_verified=True,catalog_screen_verified=True)
            resolved[trial_id]=resolution
    passed=bool(body_unchanged and all(runtime_checks.values()) and(panels or offline) and all(p['passed'] for p in panels))
    result=dict(schema='task10b_phase2_independent_trace_audit_v1',run=str(run),passed=passed,body_unchanged=body_unchanged,
        runtime_checks={k:bool(v) for k,v in runtime_checks.items()},panels=panels,panels_audited=len(panels),
        trials_audited=sum(len(p['trials']) for p in panels),source_sha256=digest(__file__),new_physics_steps=0,
        policy_actions_applied_to_simulator=0,cpu_actor_predictions=sum(t['sampled_actor']['sampled_observations'] for p in panels for t in p['trials']),
        formal_acceptance=False,continuation_ancestry=ancestry,checkpoint_template_is_unbound=template,
        positive_learned_qualification_supported=bool(expanded and not template),offline_rejection_receipts=offline,
        scope='Independent retained native state/FK/force/history/events/immutable launch; visual and learned qualification remain separate')
    result['runtime_model']=dict(path=str(used_model.resolve(strict=True)),sha256=digest(used_model))
    result['immutable_runtime_model']=True
    result['native_runtime_model_present']=native_runtime_present
    result['launch_receipt']=dict(path=str(run/'launch.json'),sha256=digest(run/'launch.json'))
    result['reference_catalog']=dict(path=str(catalog_path),sha256=digest(catalog_path))
    if expanded:
        result['immutable_mask']=dict(path=str(mask_path),sha256=digest(mask_path))
        result['reviewed_predicate']=dict(path=str(predicate_path),sha256=digest(predicate_path))
    result['closed_protocol_launch_audited']=bool(expanded and panels and all(p.get('closed_protocol_launch',{}).get('passed') is True for p in panels))
    shared=expanded and declaration.get('start_mode')=='shared_jab_saved_state'
    result['shared_saved_state_initialization_audited']=bool(shared and panels and all(p.get('shared_saved_state_initialization',{}).get('passed') is True for p in panels))
    if expanded:
        result['trial_outcome_resolutions']=[resolved[b['trial_id']] for b in declaration['bindings']]
        result['outcome_resolution_passed']=all(r['resolved'] for r in result['trial_outcome_resolutions'])
        result['passed'] &= result['outcome_resolution_passed']
    result['status']='passed' if result['passed'] else 'partial' if any(not p['execution_complete'] for p in panels) else 'failed'
    result['clock_auditor_version']=clock_version_receipt()
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed=args.output.exists()
    try:result = audit_run(args.run, args.output)
    except Exception as error:
        result=dict(schema='task10b_phase2_independent_trace_audit_v1',run=str(args.run.resolve()),passed=False,
            status='unresolved_audit_failure',outcome_resolution_passed=False,closed_protocol_launch_audited=False,
            shared_saved_state_initialization_audited=False,panels_audited=0,trials_audited=0,
            source_sha256=digest(__file__),new_physics_steps=0,policy_actions_applied_to_simulator=0,
            exception=dict(type=type(error).__name__,message=str(error)),formal_acceptance=False)
        result['clock_auditor_version']=clock_version_receipt()
        output=args.output.resolve()
        if not existed and output.is_relative_to(ROOT/'outputs'):
            output.mkdir(parents=True,exist_ok=True)
            with (output/'audit.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({"passed": result["passed"], "panels": result["panels_audited"], "trials": result["trials_audited"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
