"""Saved-only independent event metadata audit, with no evaluator imports.

This supplements the separate full raw physics/actor/closed-state auditor. It
checks the new event labels; it cannot grant physical qualification itself.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

VERSION='task10b_phase2_expanded_event_clocks_v1'

def _read(path):return json.loads(Path(path).read_text())
def _identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def audit_batch(folder):
    folder=Path(folder);plan=_read(folder/'plan.json');summary=_read(folder/'summary.json')
    launch=_read(folder/'closed_protocol_launch.json');commitments=_read(folder/'launch_commitments.json')
    reports=summary['trials'];checks={
        'new_plan_version':plan.get('event_clock_version')==VERSION,
        'new_summary_version':summary.get('event_clock_version')==VERSION,
        'prep0_declared':plan.get('declared_preparation_complete_ticks')==[0]*len(plan['trials']),
        'closed_required':plan.get('closed_protocol_launch_required') is True,
        'no_static_D0_claim':all('static_D0_cached_reference_only' not in c for c in commitments['commitments']),
        'all_trials_reported':len(reports)==len(plan['trials']) and {r['trial_id'] for r in reports}==set(plan['trials'])}
    decisions={d['trial_id']:d for d in launch['decisions']}
    checks['unique_decisions']=len(decisions)==len(launch['decisions'])
    bytrial={c['trial_id']:c for c in commitments['commitments']}
    with np.load(folder/'raw_trace.npz',allow_pickle=False) as raw:
        checks['complete_raw']=len(raw['qpos'])==plan['planned_physics_steps']+1
        for world,trial in enumerate(plan['trials']):
            selected=[r for r in reports if r['trial_id']==trial]
            if len(selected)!=1:continue
            r=selected[0];e=r.get('event_clocks',{});d=decisions.get(trial);c=bytrial.get(trial)
            reached=d is not None;accepted=reached and d['accepted'] is True
            tick=None if d is None else d['reference_tick'];launch_s=tick*.005 if accepted else None
            prefix=trial+':'
            checks[prefix+'version']=r.get('event_clock_version')==e.get('version')==VERSION
            checks[prefix+'request_preparation']=r.get('request_time_s')==e.get('request_time_s')==0. and r.get('preparation_complete_time_s')==e.get('preparation_complete_time_s')==0.
            checks[prefix+'decision_commitment']=(c is None and d is None) or (c is not None and d is not None and c.get('closed_protocol_decision')==d and c.get('closed_protocol_decision_sha256')==_identity(d))
            checks[prefix+'planning']=r.get('final_planning_time_s')==e.get('final_planning_time_s')==(None if tick is None else tick*.005)
            checks[prefix+'actual_launch']=e.get('actual_launch_time_s')==r['functional'].get('launch_time_s')==launch_s and bool(raw['launched'][-1,world])==accepted
            checks[prefix+'decision_digest']=e.get('closed_decision_sha256')==(None if d is None else _identity(d))
            checks[prefix+'boundary_readiness']=e.get('launch_boundary_readiness_time_s')==(tick*.005 if reached and d['current_positional_support_ready'] else None)
            if reached:
                actor=tick//4
                checks[prefix+'actor_boundary']=tick%4==0 and actor<len(raw['policy_episode_tick']) and int(raw['policy_episode_tick'][actor])==tick
                checks[prefix+'raw_current_readiness']=bool(raw['policy_readiness'][actor,world])==d['current_positional_support_ready']
                checks[prefix+'wall_clocks']=all(e.get(k)==d[k] for k in ('planning_wall_s','cache_lookup_wall_s','target_callback_wall_s','simulation_pause_wall_s','simulated_time_advanced_s'))
            checks[prefix+'unchanged_contact_recovery']=e.get('first_active_contact_time_s')==r['first_active_contact_s'] and e.get('first_inactive_contact_time_s')==r['first_inactive_contact_s'] and e.get('functional_recovery_time_s')==(r['functional'].get('recovery_time_s') if accepted else None)
            checks[prefix+'first_readiness']=e.get('first_positional_support_readiness_time_s')==r['functional'].get('readiness_time_s')
            checks[prefix+'outward']=e.get('outward_complete_time_s')==(r['functional'].get('outward_completion_time_s') if accepted else None)
            checks[prefix+'per_trial_file']=_read(folder/f'trial_{world:04d}.json')==r
    return dict(schema='task10b_phase2_independent_expanded_clock_audit_v1',passed=all(checks.values()),checks=checks,
        trials=len(reports),qualification=False,physics_steps=0,actor_calls=0,
        limits='Event metadata consistency only. Independent FK/contact/support/controller/actor and closed-state trace audits remain required.')
