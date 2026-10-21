"""First-write event metadata for isolated, closed-protocol expanded validation.

No trajectory, simulation, admission, actor, or score changes. The synchronous
adapter delegates to the original evaluator, replacing only its inherited D0
metadata before serialization. It refuses shared/preparation-moving protocols.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION='task10b_phase2_expanded_event_clocks_v1'
SCOPE=('Isolated family-guard initialization and uninterrupted cached prefix; '
       'current positional/support readiness and fresh canonical fixed-snapshot '
       'target are checked by the closed-protocol launch adapter. No G1 actor, '
       'arbitrary-start planner, homing, or real-time qualification.')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class EventMetadata:
    def __init__(self,env,output,reference_ids,trial_ids):
        self.env=env;self.folder=Path(output).resolve()
        if self.folder.exists():raise FileExistsError('Event adapter requires fresh output')
        self.rows=[deepcopy(env.library.rows[int(index)]) for index in reference_ids]
        self.trials=list(trial_ids)
        if len(self.rows)!=env.n or len(self.trials)!=env.n or len(set(self.trials))!=env.n:
            raise ValueError('Exact world/reference/trial mapping required')
        if not hasattr(env,'closed_launch_records') or not hasattr(env,'closed_boundary_snapshots'):
            raise ValueError('Actual closed-protocol native environment required')
        for row in self.rows:
            if (row.get('start_mode')!='family_guard' or row.get('preparation_complete_tick')!=0
                    or type(row.get('launch_tick')) is not int or row['launch_tick']%4
                    or row['outward_end_tick']<=row['launch_tick']):
                raise ValueError('Version1 supports explicit prep0 isolated family-guard rows only')
        from . import phase2_learning as original
        self.ancestry={'adapter':binding(__file__),'original_evaluator':binding(original.__file__)}

    def decision(self,world,*,required=False):
        trial=self.trials[world];row=self.rows[world]
        decisions=[d for d in self.env.closed_launch_records if d.get('trial_id')==trial]
        if not decisions and not required:return None
        if len(decisions)!=1:raise ValueError('Exactly one reached closed launch decision required')
        d=decisions[0];request=d['request']
        snapshot=self.env.closed_boundary_snapshots[d['boundary_snapshot_index']]
        tick=snapshot['episode_tick']
        if (d['world']!=world or tick!=row['launch_tick'] or tick%4 or d['reference_tick']!=tick
                or request.get('trial_id')!=trial or request.get('candidate_id')!=row['candidate_id']
                or request.get('reference_sha256')!=row['reference_sha256']
                or request.get('family')!=row['family'] or request.get('requested_ttc_s')!=row['requested_ttc_s']
                or request.get('target_w')!=row['target_w']
                or type(d.get('accepted')) is not bool
                or d.get('actual_launch_tick')!=(tick if d['accepted'] else None)):
            raise ValueError('Closed decision does not bind this exact actor-boundary request')
        return d

    def clocks(self,world,report=None):
        d=self.decision(world);row=self.rows[world]
        accepted=d is not None and d['accepted']
        f={} if report is None else report['functional']
        if report is not None and f.get('launch_time_s')!=(row['launch_tick']*.005 if accepted else None):
            raise ValueError('Functional launch differs from actual closed commitment')
        return dict(version=VERSION,request_time_s=0.,preparation_complete_time_s=0.,
            first_positional_support_readiness_time_s=f.get('readiness_time_s'),
            launch_boundary_readiness_time_s=(row['launch_tick']*.005 if d is not None and d['current_positional_support_ready'] else None),
            final_planning_time_s=row['launch_tick']*.005 if d is not None else None,
            actual_launch_time_s=row['launch_tick']*.005 if accepted else None,
            outward_complete_time_s=(f.get('outward_completion_time_s') if accepted else None),
            first_active_contact_time_s=None if report is None else report['first_active_contact_s'],
            first_inactive_contact_time_s=None if report is None else report['first_inactive_contact_s'],
            functional_recovery_time_s=f.get('recovery_time_s') if accepted else None,
            preparation_semantics='No connector; declared physical initialization completes preparation at reset. Initial hold remains part of the unchanged continuous reference.',
            reference_outward_endpoint_semantics='Reviewed conservative source-playback endpoint including captured withdrawal.',
            closed_decision_sha256=None if d is None else identity(d),
            closed_protocol_accepted=bool(accepted),
            planning_wall_s=None if d is None else d['planning_wall_s'],
            cache_lookup_wall_s=None if d is None else d['cache_lookup_wall_s'],
            target_callback_wall_s=None if d is None else d['target_callback_wall_s'],
            simulation_pause_wall_s=None if d is None else d['simulation_pause_wall_s'],
            simulated_time_advanced_s=None if d is None else d['simulated_time_advanced_s'])

    def before_write(self,path,value):
        path=Path(path).resolve()
        if path.parent!=self.folder:raise ValueError('Evaluator metadata escaped its new batch directory')
        if path.exists():raise FileExistsError('Metadata adapter never rewrites saved evidence')
        if path.name=='plan.json':
            if value.get('schema')!='task10b_phase2_execution_plan_v1':raise ValueError('Unknown original plan schema')
            value.update(event_clock_version=VERSION,event_metadata_ancestry=self.ancestry,
                planning_scope=SCOPE,event_semantics='Request/preparation at reset; first positional-support readiness separately observed; final planning at reached actor boundary; actual launch only after accepted immutable commitment; reviewed source outward end; contact-independent functional recovery.',
                declared_preparation_complete_ticks=[0]*len(self.rows),closed_protocol_launch_required=True)
        elif path.name=='launch_commitments.json':
            if value.get('schema')!='task10b_phase2_launch_commitments_v1':raise ValueError('Unknown commitment schema')
            for item in value['commitments']:
                world=item['world'];d=self.decision(world,required=True)
                if item['trial_id']!=self.trials[world] or item['requested_episode_tick']!=self.rows[world]['launch_tick']:
                    raise ValueError('Original commitment world/tick mismatch')
                if item['actual_launch_time_s']!=(d['actual_launch_tick']*.005 if d['accepted'] else None):
                    raise ValueError('Original commitment differs from closed decision')
                item.pop('static_D0_cached_reference_only',None)
                item.update(event_clock_version=VERSION,request_time_s=0.,preparation_complete_time_s=0.,
                    closed_protocol_decision=deepcopy(d),closed_protocol_decision_sha256=identity(d),
                    closed_protocol_verified_at_runtime=True,isolated_expanded_fixed_snapshot=True,
                    disposition='committed' if d['accepted'] else 'closed_protocol_rejection')
            value.update(event_clock_version=VERSION,event_metadata_ancestry=self.ancestry,planning_scope=SCOPE)
        elif path.name.startswith('trial_') and path.suffix=='.json':
            world=self.trials.index(value['trial_id']);events=self.clocks(world,value)
            value.update(event_clock_version=VERSION,event_clocks=events,
                request_time_s=events['request_time_s'],preparation_complete_time_s=events['preparation_complete_time_s'],
                final_planning_time_s=events['final_planning_time_s'])
        elif path.name=='summary.json':
            if value.get('schema')!='task10b_phase2_native_panel_v1':raise ValueError('Unknown panel schema')
            value.update(event_clock_version=VERSION,event_metadata_ancestry=self.ancestry,
                planning_scope=SCOPE,closed_protocol_launch_required=True)
        else:raise ValueError('Unreviewed evaluator JSON output')
        return value


def evaluate_policy(env,policy,output,**kwargs):
    """Original evaluator once, synchronous first-write metadata interception."""
    from . import phase2_learning as original
    trial_ids=kwargs.get('trial_ids')
    if trial_ids is None:raise ValueError('Expanded explicit trial identities required')
    metadata=EventMetadata(env,output,kwargs['reference_ids'],trial_ids)
    writer=original.write_json
    def first_write(path,value):return writer(path,metadata.before_write(path,value))
    try:
        original.write_json=first_write
        return original.evaluate_policy(env,policy,output,**kwargs)
    finally:original.write_json=writer


@contextmanager
def validation_clock_scope():
    from . import phase2_validation_callback as callback
    original=callback.evaluate_policy
    try:
        callback.evaluate_policy=evaluate_policy
        yield
    finally:callback.evaluate_policy=original
