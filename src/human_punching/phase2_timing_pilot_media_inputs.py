"""Exact saved-media inputs for selected D0 timing pilots, without replay.

This separate path preserves the frozen-control closure. It admits one declared
validation panel and its matching accepted/learned checkpoint, including failed
physical trials. It does not turn rendering or checkpoint parity into success.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
import re

from . import phase2_timing_pilot as pilot

VERSION='task10b_phase2_selected_d0_pilot_media_inputs_v1'


def files(root,inputs,release,bundle,predicate,namespace_proof=None):
    """Return ROOT-relative SHA closure; no model, actor or graphics execution."""
    root=Path(root).resolve(strict=True);found={}
    if root!=pilot.ROOT.resolve() or set(inputs)!={'run','panel'}:
        raise ValueError('Exact owned selected-pilot run and panel required')
    proof=None
    if namespace_proof is not None:
        from .phase2_runtime_namespace import evidence
        proof=evidence(root,namespace_proof)
    def owned(value,*,saved=False):
        raw=Path(value);raw=raw if raw.is_absolute() else root/raw
        p=raw.resolve(strict=True)
        if not p.is_relative_to(root):
            # Only path fields read from this selected saved pilot may use the
            # explicit, pinned two-mount proof. No caller/input resolver changes.
            if not saved or proof is None or '..' in raw.parts or raw!=p:
                raise ValueError('Selected-pilot media input escaped owned root')
            aliases=[Path(proof[k]) for k in ('logical_root','compute_root')]
            prefixes=[prefix for prefix in aliases if raw.is_relative_to(prefix)]
            if len(prefixes)!=1:raise ValueError('Saved pilot path left the exact proved prefixes')
            p=(root/raw.relative_to(prefixes[0])).resolve(strict=True)
            # The verified container mounts expose different st_dev values;
            # retain the proof's inode-plus-bytes convention, not samefile().
            if (not p.is_relative_to(root) or raw.stat().st_ino!=p.stat().st_ino
                    or raw.is_file()!=p.is_file() or raw.is_dir()!=p.is_dir()
                    or (p.is_file() and pilot.sha(raw)!=pilot.sha(p))):
                raise ValueError('Saved pilot path inode or bytes differ across proved mounts')
        return p
    def add(value,expected=None,*,saved=False):
        p=owned(value,saved=saved)
        if not p.is_file():raise ValueError('Selected-pilot media input must be a file')
        actual=pilot.sha(p)
        if expected is not None and actual!=expected:raise ValueError('Selected-pilot media bound bytes changed')
        found[str(p.relative_to(root))]=actual;return p
    def exact(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:add(value['path'],value['sha256'])
            else:
                for child in value.values():exact(child)
        elif isinstance(value,list):
            for child in value:exact(child)
    def load(path):return json.loads(Path(path).read_text())
    if proof is not None:
        for item in [proof['proof'],proof['nonce'],proof['source_verifier_ancestry'],*proof['alias_bindings']]:
            add(item['path'],item['sha256'])
    run=owned(inputs['run']);panel=add(inputs['panel'])
    if (not run.is_dir() or not run.is_relative_to(root/'outputs/task10b') or panel.parent.parent!=run
            or panel.name!='summary.json' or re.fullmatch(r'validation_(00000|00008|00016)',panel.parent.name) is None):
        raise ValueError('One exact declared D0 validation summary required')
    update=int(panel.parent.name.removeprefix('validation_'))
    launch=load(add(run/'launch.json'))
    if (launch.get('schema')!=pilot.VERSION or launch.get('mode')!='timing_pilot'
            or not launch.get('slurm_job_id')):
        raise ValueError('Do not relabel a selected pilot as a frozen control')
    plan_path=add(launch['plan']['path'],launch['plan']['sha256'],saved=True);plan=load(plan_path)
    paths,rows,_=pilot.validate_plan(plan)
    if paths['release'].parent!=owned(release) or paths['predicate']!=owned(predicate):
        raise ValueError('Media release or predicate differs from the selected pilot')
    copied=add(run/'plan.json')
    if load(copied)!=plan:raise ValueError('Run plan copy differs from committed pilot plan')
    expected=pilot.launch_receipt(plan_path,plan,runtime=launch['runtime'],clock=launch['clock_postprocessing'],
        slurm_job_id=launch['slurm_job_id'],progress=launch['planned_validation'])
    def comparable_launch(value):
        value=deepcopy(value)
        value['plan']['path']=str(owned(value['plan']['path'],saved=True))
        for key in ('catalog','predicate'):value[key]=str(owned(value[key],saved=True))
        sources={str(owned(path,saved=True)):digest for path,digest in value['sources'].items()}
        if len(sources)!=len(value['sources']):raise ValueError('Duplicate saved source aliases')
        value['sources']=sources
        return value
    if comparable_launch(expected)!=comparable_launch(launch):
        raise ValueError('Selected pilot launch metadata differs from its immutable plan')
    clock_source=launch['clock_postprocessing'].get('source',{})
    expected_clock=plan['sources']['src/human_punching/phase2_d0_clock_v2.py']
    clock_parent=launch['clock_postprocessing'].get('parent_source',{})
    expected_parent=plan['sources']['src/human_punching/phase2_learning.py']
    if (launch['clock_postprocessing'].get('schema')!='task10b_phase2_d0_postprocessing_clock_v2'
            or set(clock_source)!={'path','sha256'} or clock_source['sha256']!=expected_clock['sha256']
            or add(clock_source['path'],clock_source['sha256'],saved=True)!=pilot.bound(expected_clock)
            or set(clock_parent)!={'path','sha256'} or clock_parent['sha256']!=expected_parent['sha256']
            or add(clock_parent['path'],clock_parent['sha256'],saved=True)!=pilot.bound(expected_parent)):
        raise ValueError('Selected pilot must retain its explicit clock-v2 source binding')
    exact(plan)
    exact(load(paths['route_receipt']));exact(load(paths['video_receipt']))
    candidates=plan['validation_candidate_ids']
    for row in rows:
        for name in ('reference','legacy_runtime'):add(row[name+'_path'],row[name+'_sha256'])
        if row.get('diagnostic_screen_path'):add(row['diagnostic_screen_path'])
    summary=load(panel);trials=summary.get('trials',[])
    expected_ids={f'{candidate}_bank_{i:02d}' for candidate in candidates for i in range(8)}
    if (summary.get('schema')!='task10b_phase2_d0_panel_v1' or summary.get('planned_trials')!=len(expected_ids)
            or len(trials)!=len(expected_ids) or {r['trial_id'] for r in trials}!=expected_ids
            or any(r.get('candidate_id') not in candidates or not r['trial_id'].startswith(r['candidate_id']+'_bank_') for r in trials)):
        raise ValueError('Selected pilot media must retain every exact requested candidate/bank row')
    training=run/'training';training_plan=load(add(training/'training_plan.json'))
    executed_budget=training_plan.get('budget',{})
    wall=executed_budget.get('maximum_wall_s')
    if (isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall)
            or not 0<wall<=plan['budget']['maximum_wall_s']
            or {k:v for k,v in executed_budget.items() if k!='maximum_wall_s'}
            !={k:v for k,v in plan['budget'].items() if k!='maximum_wall_s'}):
        raise ValueError('Original training budget differs from finite selected pilot')
    ancestry=training_plan.get('ancestry',{});accepted=plan['release_inputs']['checkpoint']
    if (owned(ancestry['checkpoint'],saved=True)!=owned(accepted['path']) or ancestry.get('sha256')!=accepted['sha256']
            or ancestry.get('seed')!=plan['seed'] or ancestry.get('fresh_optimizer') is not True
            or ancestry.get('fresh_simulation') is not True or ancestry.get('independent_scratch_replication') is not False
            or training_plan.get('transition_cap')!=131072 or training_plan.get('validation_updates')!=[0,8,16]):
        raise ValueError('Original fresh transferred training ancestry or cap changed')
    training_rows=rows[1:]
    commitment=hashlib.sha256(json.dumps(training_rows,sort_keys=True).encode()).hexdigest()
    if training_plan.get('reference_commitment')!=commitment:
        raise ValueError('Training rows differ from selected exact timings')
    checkpoint=add(accepted['path'],accepted['sha256']) if update==0 else add(training/f'checkpoint_{update:05d}.pt')
    if checkpoint.stat().st_size>32*1024**2:raise ValueError('Bounded specialist checkpoint payload exceeded')
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('observations')!=369 or saved.get('actions')!=48:raise ValueError('Original specialist dimensions changed')
    h=hashlib.sha256()
    for name,value in sorted(saved['policy'].items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    actor=h.hexdigest()
    if update:
        parity=load(add(training/f'reload_{update:05d}.json'))
        if (saved.get('schema')!='task10b_phase2_original_ppo_v1' or saved.get('updates')!=update
                or saved.get('fresh_transitions')!=update*128*64 or saved.get('ancestry')!=ancestry
                or saved.get('reference_rows')!=training_rows or saved.get('reference_commitment')!=commitment
                or saved.get('actor_parameters_sha256')!=actor
                or parity.get('parameters_equal') is not True or parity.get('actions_equal') is not True
                or parity.get('maximum_action_error')!=0):
            raise ValueError('Learned checkpoint ancestry, exact rows, extent or restored parity differs')
    elif ancestry.get('initial_policy_digest')!=actor:
        raise ValueError('Initial validation must use unchanged accepted weights')
    seen=set()
    for item in summary['panels']:
        directory=owned(item['path'],saved=True);candidate=item['candidate_id']
        if directory.parent!=panel.parent or directory.name!=candidate or candidate not in candidates or candidate in seen:
            raise ValueError('Selected pilot native panel escaped exact candidate binding')
        seen.add(candidate)
        native=load(add(directory/'plan.json'))
        if (native.get('schema')!='task10b_phase2_execution_plan_v1' or native.get('actor_sha256')!=actor
                or native.get('trials')!=[f'{candidate}_bank_{i:02d}' for i in range(8)]
                or native.get('candidates')!=[candidate]*8):
            raise ValueError('Native panel actor or candidate/bank binding differs')
    native_required={r['candidate_id'] for r in trials
                     if r.get('disposition') in ('executed','incomplete_hard_failure')}
    if seen!=native_required:
        raise ValueError('Native panel evidence differs from executed or incomplete trial ledger')
    add(run/'runtime.xml')
    # Preserve the available training receipt/checkpoint closure without treating
    # a failed final checkpoint or an unattempted later panel as a successful run.
    for p in training.iterdir():
        if not p.is_file() or not re.fullmatch(r'(training_plan|summary)\.json|training\.jsonl|reload_000(08|16)\.json|checkpoint_(00008|00016|final)\.pt',p.name):
            raise ValueError('Unexpected selected-pilot training evidence')
        add(p)
    for p in panel.parent.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(panel.parent)
        if len(relative.parts)>2 or (len(relative.parts)==2 and relative.parts[0] not in candidates):
            raise ValueError('Unexpected selected-pilot panel directory')
        if p.suffix=='.json':
            name=p.name
            if name=='summary.json':
                if load(p).get('schema') not in ('task10b_phase2_d0_panel_v1','task10b_phase2_native_panel_v1'):raise ValueError('Unknown selected-pilot summary')
            elif name=='plan.json':
                if load(p).get('schema')!='task10b_phase2_execution_plan_v1':raise ValueError('Unknown selected-pilot native plan')
            elif name=='launch_commitments.json':
                if load(p).get('schema')!='task10b_phase2_launch_commitments_v1':raise ValueError('Unknown selected-pilot launch commitment')
            elif not re.fullmatch(r'trial_\d{4}\.json',name):raise ValueError('Unknown selected-pilot panel JSON')
        elif p.suffix!='.npz' or (p.name!='raw_trace.npz' and not re.fullmatch(r'trial_\d{4}\.npz',p.name)):
            raise ValueError('Unknown selected-pilot trace file')
        add(p)
    for name in ('validation_declared.json','execution.json',f'validation_{update:05d}_started.json',f'validation_{update:05d}_disposition.json'):
        if (run/name).exists():add(run/name)
    return found
