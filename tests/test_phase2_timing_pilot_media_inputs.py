"""Saved selected-pilot closure with actual CPU checkpoint tensors; no physics."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_phase2_timing_pilot import fixture, dump
from human_punching import phase2_timing_pilot as pilot
from human_punching.phase2_timing_pilot_media_inputs import files
from human_punching.phase2_learning import policy_digest, save_policy
from human_punching.jab_learning import ActorCritic


@pytest.fixture
def saved_pilot(fixture):
    root,prior=fixture
    torch.manual_seed(21);policy=ActorCritic(369,48)
    checkpoint=Path(prior['release_inputs']['checkpoint']['path'])
    torch.save(dict(observations=369,actions=48,policy=policy.state_dict()),checkpoint)
    release=Path(prior['release']['path']);manifest=pilot.read(release)
    manifest['artifacts']['cross/checkpoint.pt']['sha256']=pilot.sha(checkpoint);dump(release,manifest)
    plan=pilot.create_plan(prior['catalog']['path'],release.parent,prior['predicate']['path'],
        prior['route_receipt']['path'],prior['video_receipt']['path'],'cross')
    planpath=root/'outputs/pilot_plan.json';dump(planpath,plan)
    run=root/'outputs/task10b/synthetic_pilot/run';run.mkdir(parents=True)
    dump(run/'plan.json',plan)
    progress={str(u):dict(disposition='unattempted',planned_trials=plan['planned_trials_per_panel']) for u in (0,8,16)}
    clock=dict(schema='task10b_phase2_d0_postprocessing_clock_v2',source=plan['sources']['src/human_punching/phase2_d0_clock_v2.py'],
        parent_source=plan['sources']['src/human_punching/phase2_learning.py'])
    launch=pilot.launch_receipt(planpath,plan,runtime={'synthetic':True},clock=clock,slurm_job_id='synthetic_no_job',progress=progress)
    dump(run/'launch.json',launch);(run/'runtime.xml').write_text('<synthetic-no-model-loaded/>')
    training=run/'training';training.mkdir()
    ancestry=dict(checkpoint=str(checkpoint),sha256=pilot.sha(checkpoint),seed=21021,
        fresh_optimizer=True,fresh_simulation=True,independent_scratch_replication=False,initial_policy_digest=policy_digest(policy))
    _,rows,_=pilot.validate_plan(plan);training_rows=rows[1:]
    commitment=hashlib.sha256(json.dumps(training_rows,sort_keys=True).encode()).hexdigest()
    dump(training/'training_plan.json',dict(budget=plan['budget'],ancestry=ancestry,transition_cap=131072,validation_updates=[0,8,16],reference_commitment=commitment))
    with torch.no_grad():policy.actor[-1].bias[0]+=.01
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    save_policy(training/'checkpoint_00008.pt',policy,optimizer,ancestry=ancestry,updates=8,fresh_transitions=8*128*64,
        library=SimpleNamespace(rows=training_rows,commitment=commitment))
    dump(training/'reload_00008.json',dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.))
    actor=policy_digest(policy);panel=run/'validation_00008';panel.mkdir();trials=[];panels=[]
    for candidate in plan['validation_candidate_ids']:
        ids=[f'{candidate}_bank_{i:02d}' for i in range(8)];folder=panel/candidate;folder.mkdir()
        # These are deliberately negative synthetic cycles: media closure is not
        # allowed to discard failures or insist that learned policies succeeded.
        trials.extend(dict(trial_id=t,candidate_id=candidate,disposition='executed',numeric_complete_cycle=False) for t in ids)
        panels.append(dict(candidate_id=candidate,path=str(folder),status='executed'))
        dump(folder/'plan.json',dict(schema='task10b_phase2_execution_plan_v1',actor_sha256=actor,trials=ids,candidates=[candidate]*8))
        dump(folder/'summary.json',dict(schema='task10b_phase2_native_panel_v1'))
        (folder/'raw_trace.npz').write_bytes(b'synthetic opaque raw evidence; no decoding or rendering')
    dump(panel/'summary.json',dict(schema='task10b_phase2_d0_panel_v1',planned_trials=24,trials=trials,panels=panels))
    return root,run,panel/'summary.json',plan


def closure(saved):
    root,run,panel,plan=saved
    return files(root,{'run':str(run),'panel':str(panel)},Path(plan['release']['path']).parent,
        'unused_base_bundle_role',plan['predicate']['path'])


def test_exact_learned_panel_binds_selected_rows_original_ancestry_checkpoint_and_failed_traces(saved_pilot):
    root,run,panel,plan=saved_pilot;found=closure(saved_pilot)
    for p in [run/'launch.json',run/'plan.json',run/'runtime.xml',run/'training/checkpoint_00008.pt',
              run/'training/reload_00008.json',panel,*panel.parent.glob('*/raw_trace.npz')]:
        assert found[str(p.relative_to(root))]==pilot.sha(p)
    catalog=pilot.read(Path(plan['catalog']['path']))
    assert len(catalog['cells'])==4 and len(plan['validation_candidate_ids'])==3
    unused=next(r for r in catalog['cells'] if r['candidate_id'] not in plan['validation_candidate_ids'])
    assert str(Path(unused['reference_path']).relative_to(root)) not in found
    assert len(pilot.read(panel)['trials'])==24 and not any(t['numeric_complete_cycle'] for t in pilot.read(panel)['trials'])
    assert pilot.read(run/'launch.json')['mode']=='timing_pilot'


@pytest.mark.parametrize('change',['frozen_relabel','wrong_catalog','changed_plan_copy','extra_candidate','missing_trial',
    'duplicate_trial','native_actor','native_bank','checkpoint_extent','checkpoint_reference','checkpoint_ancestry',
    'reload_parity','training_commitment','initial_ancestry','escape_trace','missing_native_panel','training_budget'])
def test_mutations_fail_closed_without_requiring_positive_physical_outcomes(saved_pilot,change,tmp_path):
    root,run,panel,plan=saved_pilot
    if change in ('frozen_relabel','wrong_catalog'):
        p=run/'launch.json';d=pilot.read(p)
        if change=='frozen_relabel':d.update(schema='task10b_phase2_original_route_launch_v1',mode='frozen')
        else:d['catalog_sha256']='0'*64
        dump(p,d)
    elif change=='changed_plan_copy':
        p=run/'plan.json';d=pilot.read(p);d['training_candidate_ids'].reverse();dump(p,d)
    elif change in ('extra_candidate','missing_trial','duplicate_trial'):
        d=pilot.read(panel)
        if change=='extra_candidate':d['trials'][0]['candidate_id']='cross_uniform_0p900'
        elif change=='missing_trial':d['trials'].pop()
        else:d['trials'][1]=d['trials'][0]
        dump(panel,d)
    elif change in ('native_actor','native_bank'):
        p=next(panel.parent.glob('*/plan.json'));d=pilot.read(p)
        if change=='native_actor':d['actor_sha256']='0'*64
        else:d['trials'][0]=d['trials'][1]
        dump(p,d)
    elif change=='missing_native_panel':
        d=pilot.read(panel);d['panels'].pop();dump(panel,d)
    elif change.startswith('checkpoint_'):
        p=run/'training/checkpoint_00008.pt';d=torch.load(p,map_location='cpu',weights_only=True)
        if change=='checkpoint_extent':d['fresh_transitions']+=1
        elif change=='checkpoint_reference':d['reference_rows'].reverse()
        else:d['ancestry']['fresh_optimizer']=False
        torch.save(d,p)
    elif change=='reload_parity':
        p=run/'training/reload_00008.json';d=pilot.read(p);d['actions_equal']=False;dump(p,d)
    elif change in ('training_commitment','initial_ancestry','training_budget'):
        p=run/'training/training_plan.json';d=pilot.read(p)
        if change=='training_commitment':d['reference_commitment']='0'*64
        elif change=='training_budget':d['budget']['num_envs']=256
        else:d['ancestry']['sha256']='0'*64
        dump(p,d)
    else:
        outside=tmp_path/'outside.npz';outside.write_bytes(b'outside')
        (panel.parent/plan['validation_candidate_ids'][0]/'trial_0000.npz').symlink_to(outside)
    with pytest.raises(ValueError):closure(saved_pilot)


def test_other_validation_directory_is_not_silently_added(saved_pilot):
    root,run,panel,plan=saved_pilot
    unrelated=run/'validation_00016';unrelated.mkdir();(unrelated/'unrelated.json').write_text('{}')
    assert str((unrelated/'unrelated.json').relative_to(root)) not in closure(saved_pilot)
    with pytest.raises(ValueError,match='summary|validation|file'):
        files(root,{'run':str(run),'panel':str(unrelated)},Path(plan['release']['path']).parent,'unused',plan['predicate']['path'])


@pytest.fixture
def saved_host_namespace(saved_pilot,monkeypatch):
    """Actual dual-mounted synthetic saved files; no model or native execution."""
    from human_punching import phase2_runtime_namespace as ns
    root,run,panel,plan=saved_pilot
    local=Path('/home/kaisyun/Azizan_Lab/HumanBox/human-punching')
    host=Path('/orcd/home/002/kaisyun/Azizan_Lab/HumanBox/human-punching')
    if not root.is_relative_to(local) or not host.exists():pytest.skip('Actual dual mount unavailable')
    alias=host/root.relative_to(local)
    assert root.stat().st_ino==alias.stat().st_ino
    to_host=lambda value:str(alias/Path(value).relative_to(root))
    p=run/'launch.json';d=pilot.read(p)
    d['plan']['path']=to_host(d['plan']['path'])
    for key in ('catalog','predicate'):d[key]=to_host(d[key])
    d['sources']={to_host(path):digest for path,digest in d['sources'].items()}
    for key in ('clock_postprocessing','postprocessing_clock_adapter'):
        d[key]['source']['path']=to_host(d[key]['source']['path'])
        d[key]['parent_source']['path']=to_host(d[key]['parent_source']['path'])
    dump(p,d)
    d=pilot.read(panel)
    for item in d['panels']:item['path']=to_host(item['path'])
    dump(panel,d)
    p=run/'training/training_plan.json';d=pilot.read(p);d['ancestry']['checkpoint']=to_host(d['ancestry']['checkpoint']);dump(p,d)
    p=run/'training/checkpoint_00008.pt';d=torch.load(p,map_location='cpu',weights_only=True)
    d['ancestry']['checkpoint']=to_host(d['ancestry']['checkpoint']);torch.save(d,p)
    proof_path=root/'outputs/namespace_test_evidence.json';dump(proof_path,dict(synthetic=True))
    descriptor=dict(path=str(proof_path.relative_to(root)),sha256=pilot.sha(proof_path))
    calls=[]
    def evidence(selected,path):
        assert selected==root and path==proof_path;calls.append((selected,path))
        return dict(logical_root=str(root),compute_root=str(alias),proof=descriptor,nonce=descriptor,
            source_verifier_ancestry=descriptor,alias_bindings=[])
    # The actual pinned verifier has separate real-proof tests. This fixture
    # isolates our per-saved-field inode/SHA checks on the real dual mount.
    monkeypatch.setattr(ns,'evidence',evidence)
    return saved_pilot,proof_path,alias,calls


def host_closure(context):
    saved,proof,alias,calls=context;root,run,panel,plan=saved
    return files(root,{'run':str(run),'panel':str(panel)},Path(plan['release']['path']).parent,
        'unused',plan['predicate']['path'],namespace_proof=proof)


def test_saved_host_fields_require_explicit_proof_and_same_inode_bytes(saved_host_namespace):
    saved,proof,alias,calls=saved_host_namespace
    with pytest.raises(ValueError,match='escaped'):closure(saved)
    found=host_closure(saved_host_namespace);root,run,panel,plan=saved
    assert calls and str(proof.relative_to(root)) in found
    assert str((run/'training/checkpoint_00008.pt').relative_to(root)) in found
    with pytest.raises(ValueError,match='escaped'):
        files(root,{'run':str(alias/run.relative_to(root)),'panel':str(panel)},Path(plan['release']['path']).parent,
            'unused',plan['predicate']['path'],namespace_proof=proof)


@pytest.mark.parametrize('change',['inode','bytes','duplicate_source_alias'])
def test_saved_host_alias_mismatch_is_not_hidden_by_projection(saved_host_namespace,monkeypatch,change):
    saved,proof,alias,calls=saved_host_namespace;root,run,panel,plan=saved
    if change=='inode':
        old=Path.stat
        def changed_stat(self,*args,**kwargs):
            result=old(self,*args,**kwargs)
            if self.is_relative_to(alias):
                fields=list(result);fields[1]+=1;return os.stat_result(fields)
            return result
        monkeypatch.setattr(Path,'stat',changed_stat)
    elif change=='bytes':
        old=pilot.sha
        monkeypatch.setattr(pilot,'sha',lambda p:'0'*64 if Path(p).is_relative_to(alias) else old(p))
    else:
        p=run/'launch.json';d=pilot.read(p);source,digest=next(iter(d['sources'].items()))
        d['sources'][str(root/Path(source).relative_to(alias))]=digest;dump(p,d)
    with pytest.raises(ValueError):host_closure(saved_host_namespace)
