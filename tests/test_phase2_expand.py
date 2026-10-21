"""Source/ledger/command tests use fake files and mocked scheduler calls only."""
import importlib.util
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC=importlib.util.spec_from_file_location('phase2_expand_launcher',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_expand.py')
launcher=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(launcher)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


@pytest.mark.parametrize('seconds,expected',[(1,60),(420,420),(421,480),(14399,14400),(14400,14400)])
def test_slurm_wall_is_rounded_up_and_charged_exactly(seconds,expected):
    assert launcher.rounded_wall(seconds)==expected


@pytest.mark.parametrize('seconds',[0,-1,float('inf'),float('nan'),True,14400.1])
def test_invalid_or_over_four_hour_rounded_reservations_are_rejected(seconds):
    with pytest.raises(ValueError):launcher.rounded_wall(seconds)


def test_long_output_paths_are_rejected_before_compiler_work():
    with pytest.raises(ValueError,match='shallow'):launcher.shallow_output('/'+'a'*130)


@pytest.fixture
def owned(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'ROOT',tmp_path);monkeypatch.setattr(launcher.base,'ROOT',tmp_path)
    prep=tmp_path/'outputs/task10b/p2prep_20261021_01';prep.mkdir(parents=True)
    dump(prep/'authorization.json',{'scope':'synthetic test only'})
    python=tmp_path/'.venv-task10b/bin/python';python.parent.mkdir(parents=True);python.write_text('fake interpreter bytes; never executed')
    monkeypatch.setattr(launcher.base,'PYTHON',python);monkeypatch.setattr(launcher.base,'PREP',prep)
    for file in ('task10b_phase2_expand.py','task10b_phase2_expand.sbatch','task10b_phase2_target_references.py','task10b_phase2_references.py','task10b_phase2_media.py','task10b_phase2_audit.py'):
        p=tmp_path/'scripts'/file;p.parent.mkdir(exist_ok=True);p.write_text('# source-bound synthetic fixture\n')
    monkeypatch.setattr(launcher.base,'required_inputs',lambda record:set())
    monkeypatch.setattr(launcher.base,'verify_runtime',lambda:None)
    # Synthetic project isolation nests under a pytest output; it is not a
    # native compiler path. The real path-length gate has its own check above.
    monkeypatch.setattr(launcher,'shallow_output',lambda out:None)
    monkeypatch.setattr(launcher,'runtime_record',lambda:{'declared_pins_match':True})
    monkeypatch.setattr(launcher.subprocess,'check_output',lambda *a,**kw:'wbc\n')
    run=tmp_path/'outputs/task10b/source_run'
    dump(run/'launch.json',{'family':'jab','seed':21021})
    panel=dump(run/'panel.json',{'trials':[]})
    return tmp_path,run,panel


def test_recursive_input_closure_rejects_changed_payload_and_symlink_escape(owned):
    root,run,_=owned;payload=run/'trace.npz';payload.write_bytes(b'physical trace fixture')
    parent=dump(run/'bound.json',{'trace':{'path':str(payload),'sha256':launcher.sha256_file(payload)}})
    files=launcher.semantic_files([parent]);assert str(payload.relative_to(root)) in files
    payload.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='changed'):launcher.semantic_files([parent])
    escaped=root.parent/'external_fixture.json';escaped.write_text('{}')
    link=run/'escaped.json';link.symlink_to(escaped)
    with pytest.raises(ValueError):launcher.semantic_files([link])


def test_unpacked_ancestry_is_verified_provenance_and_runtime_arrays_remain_archived(owned):
    root,run,_=owned;ancestor=run/'offline_full_precision.npz';ancestor.write_bytes(b'full precision diagnostic fixture')
    active=run/'runtime.npz';active.write_bytes(b'active runtime fixture')
    row=dump(run/'packed.json',dict(unpacked_reference_path=str(ancestor),unpacked_reference_sha256=launcher.sha256_file(ancestor),
        reference_path=str(active),reference_sha256=launcher.sha256_file(active)))
    provenance={};files=launcher.semantic_files([row],provenance_files=provenance)
    assert str(active.relative_to(root)) in files and str(ancestor.relative_to(root)) not in files
    assert provenance[str(ancestor.relative_to(root))]['bytes']==ancestor.stat().st_size
    ancestor.write_bytes(b'tampered ancestor')
    with pytest.raises(ValueError,match='ancestor provenance'):launcher.semantic_files([row])


def test_packet_archives_every_input_and_rejects_live_and_archived_edits(owned):
    root,run,panel=owned;out=root/'outputs/task10b/p2fixture'
    result=launcher.prepare(out,'media',{'run':str(run),'panel':str(panel)},allocation_wall_seconds=421,application_wall_seconds=400)
    record=launcher.verify(result['packet'])
    assert record['allocation_wall_seconds']==480
    assert '--time=8' in record['submission_command']
    assert record['submission_command'][-2].endswith('task10b_phase2_expand.sbatch')
    assert record['gpus']==1 and record['fresh_transition_cap']==0
    assert not (out/'submission_intent.json').exists()
    archived=out/'source_snapshot'/panel.relative_to(root)
    tampered=archived.with_name('tampered.json');tampered.write_text('{"trials":["forged"]}');os.replace(tampered,archived)
    with pytest.raises(ValueError,match='Archived'):launcher.verify(result['packet'])
    second=launcher.prepare(root/'outputs/task10b/p2livefixture','media',{'run':str(run),'panel':str(panel)},allocation_wall_seconds=480,application_wall_seconds=400)
    panel.write_text('{"trials":["changed live source"]}')
    with pytest.raises(ValueError,match='closure'):launcher.verify(second['packet'])


def test_exact_allowed_commands_cannot_inject_a_module_or_extra_flag(owned):
    root,run,panel=owned
    with pytest.raises(ValueError):launcher.validate_spec('media',{'run':str(run),'panel':str(panel),'command':'arbitrary'},480,400)
    with pytest.raises(ValueError):launcher.validate_spec('shell',{},480,400)
    with pytest.raises(ValueError):launcher.validate_spec('media',{'run':str(run),'panel':str(panel)},480,451)
    r=dict(mode='media',inputs={'run':str(run),'panel':str(panel)},output='outputs/task10b/p2fixture/run',application_wall_seconds=400)
    command=launcher.native_command(r)
    assert command==[str(launcher.base.PYTHON),'-B',str(root/'scripts/task10b_phase2_media.py'),'--run',str(run),'--panel',str(panel),'--output',str(root/r['output']),'--wall-seconds','370']
    with pytest.raises(ValueError):launcher.native_command(r,371)


def submission_fixture(owned):
    root,_,_=owned;out=root/'outputs/task10b/p2path_fixture';(out/'logs').mkdir(parents=True)
    dump(out/'packet.json',{'synthetic':True})
    command=launcher.submission_command(dict(partition='mit_normal_gpu',allocation_wall_seconds=480,mode='frozen-control'),out)
    return root,out,command


def test_submission_guard_accepts_only_the_five_owned_nfs_path_aliases(owned):
    root,out,expected=submission_fixture(owned);alias=root.parent/'nfs_alias';alias.symlink_to(root,target_is_directory=True)
    actual=[item.replace(str(root),str(alias),1) for item in expected]
    result=launcher.submission_comparison(actual,expected)
    assert result['passed'] and not result['differences']
    assert len(result['path_slots'])==5 and result['canonical_actual']==result['canonical_expected']
    assert result['actual_root']==str(root) and actual!=expected
    # The literal %j log basename is retained for Slurm expansion.
    assert next(s for s in result['path_slots'] if s['kind']=='stdout')['actual_canonical']==str(out/'logs/%j.out')


@pytest.mark.parametrize('index',list(range(13)))
def test_submission_alias_guard_never_normalizes_nonpath_arguments(owned,index):
    _,_,expected=submission_fixture(owned);actual=list(expected);actual[index]+='-changed'
    result=launcher.submission_comparison(actual,expected)
    assert not result['passed'] and any(row.get('index')==index for row in result['differences'])


@pytest.mark.parametrize('tamper',['script','packet','log_parent','log_basename','log_missing_parent','relative',
    'escaped_script','escaped_root','reordered_flags','extra_argument','missing_argument'])
def test_submission_alias_guard_rejects_changed_paths_ownership_order_and_log_names(owned,tamper):
    root,out,expected=submission_fixture(owned);actual=list(expected)
    if tamper=='script':
        alternative=root/'scripts/other.sbatch';alternative.write_text('# alternative');actual[-2]=str(alternative)
    elif tamper=='packet':actual[-1]=str(dump(out/'other.json',{}))
    elif tamper=='log_parent':
        different=out/'otherlogs';different.mkdir();actual[14]='--output='+str(different/'%j.out')
    elif tamper=='log_basename':actual[14]='--output='+str(out/'logs/%j.log')
    elif tamper=='log_missing_parent':actual[14]='--output='+str(out/'absent/%j.out')
    elif tamper=='relative':actual[-1]='outputs/task10b/p2path_fixture/packet.json'
    elif tamper=='escaped_script':
        outside=root.parent/'outside.sbatch';outside.write_text('# outside');alias=root/'scripts/escaped.sbatch';alias.symlink_to(outside);actual[-2]=str(alias)
    elif tamper=='escaped_root':actual[13]='--chdir='+str(root.parent)
    elif tamper=='reordered_flags':actual[14],actual[15]=actual[15],actual[14]
    elif tamper=='extra_argument':actual.insert(13,'--exclusive')
    else:actual.pop(3)
    assert not launcher.submission_comparison(actual,expected)['passed']


def test_execute_saves_canonical_mismatch_and_actual_root_before_runtime_or_cuda(owned,monkeypatch):
    root,run,panel=owned;out=root/'outputs/task10b/p2start_receipt_fixture'
    prepared=launcher.prepare(out,'media',dict(run=str(run),panel=str(panel)),allocation_wall_seconds=480,application_wall_seconds=400)
    path=Path(prepared['packet']);record=json.loads(path.read_text());record['submission_command'][9]='--time=9';dump(path,record)
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-no-job');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','synthetic-no-gpu')
    def forbidden(*args):raise AssertionError('No CUDA health, child or GPU call is allowed in this CPU mismatch test')
    monkeypatch.setattr(launcher.base,'old_helper',forbidden)
    assert launcher.execute(path)==2
    start=json.loads((out/'execution_started.json').read_text());terminal=json.loads((out/'execution.json').read_text())
    assert start['runtime_path_context']['root']==str(root) and start['runtime_path_context']['resolved_root']==str(root)
    assert not start['submission_path_comparison']['passed']
    mismatch=start['submission_path_comparison']['differences'][0]
    assert mismatch['index']==9 and mismatch['actual']=='--time=9' and mismatch['expected']=='--time=8'
    assert terminal['native_processes_started']==terminal['actual_physics_steps']==0
    assert terminal['exception']['type']=='ValueError' and not (out/'cuda_health.json').exists()


def d0_catalog_fixture(owned):
    """Synthetic original-control metadata; no policy or physics is executed."""
    import numpy as np
    root,run,_=owned;schedules={};clips=[];rows=[]
    for index,family in enumerate(launcher.base.FAMILIES):
        schedule=dict(launch_actor_tick=4,launch_time_s=.02,outward_completion_tick=160,outward_completion_time_s=.8)
        schedules[family]=schedule;duration=1.+index*.01
        clips.append(dict(family=family,source_duration_s=duration,legacy_runtime_path='legacy_runtime/'+family+'.npz'))
        reference=root/'outputs/four-punch-milestone'/family/'reference.npz';reference.parent.mkdir(parents=True,exist_ok=True)
        active=0 if family in ('jab','left_hook') else 1
        np.savez(reference,head_center=np.array([1.,0.,1.2]),active_hand_index=np.array(active),guard_hold_start=np.array(.9))
        legacy=root/'outputs/task10/release_baseline_02/legacy_runtime'/f'{family}.npz';legacy.parent.mkdir(parents=True,exist_ok=True);legacy.write_bytes(b'synthetic legacy runtime')
        rows.append(dict(candidate_id=family+'_original',kind='original_control',family=family,requested_ttc_s=None,
            reference_path=str(reference),reference_sha256=launcher.sha256_file(reference),legacy_runtime_path=str(legacy),
            legacy_runtime_sha256=launcher.sha256_file(legacy),duration_s=duration,episode_duration_s=duration+1.,
            phase_denominator_s=duration+1.,launch_tick=4,launch_time_s=.02,outward_end_tick=160,outward_completion_time_s=.8,
            kinematic_admission=True,preserved_original_control=True,events_identity='unchanged_phase1_reviewed_schedule',
            target_w=[1.,0.,1.2],head_center=[1.,0.,1.2],active_hand_index=active,guard_time_s=.9))
    dump(root/'outputs/task10/release_baseline_02/manifest.json',dict(clips=clips))
    predicate=dump(root/'outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',dict(schedule=schedules))
    return dump(run/'controls.json',dict(schema='task10b_phase2_reference_catalog_v1',complete=True,
        predicate_sha256=launcher.sha256_file(predicate),cells=rows))


def test_new_d0_modes_keep_fixed_initial_caps_and_original_frozen_dispatch(owned):
    root,_,_=owned;catalog=d0_catalog_fixture(owned)
    assert launcher.validate_spec('timing-references',{},900,840)==dict(family=None,seed=21021,fresh_transition_cap=0)
    inputs=dict(catalog=str(catalog),family='jab')
    assert launcher.validate_spec('frozen-control',inputs,480,390)==dict(family='jab',seed=21021,fresh_transition_cap=0)
    record=dict(mode='frozen-control',inputs=inputs,family='jab',application_wall_seconds=390,
        output='outputs/task10b/p2test/run',release='outputs/four-punch-milestone',bundle='outputs/task10/release_baseline_02',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    command=launcher.native_command(record)
    assert command[:6]==[str(launcher.base.PYTHON),'-B','-m','human_punching.phase2_learning','--mode','frozen']
    assert command[command.index('--max-updates')+1]=='0' and float(command[command.index('--wall-seconds')+1])==360.
    assert command[command.index('--family')+1]=='jab'
    assert float(launcher.native_command(record,350.)[command.index('--wall-seconds')+1])==350.
    record.update(mode='timing-references',inputs={},family=None,application_wall_seconds=840)
    command=launcher.native_command(record)
    assert command==[str(launcher.base.PYTHON),'-B',str(root/'scripts/task10b_phase2_references.py'),
        '--bundle',str(root/record['bundle']),'--release',str(root/record['release']),
        '--output',str(root/record['output']),'--predicate',str(root/record['predicate'])]


@pytest.mark.parametrize('mode',['timing-references','frozen-control'])
def test_new_d0_modes_cannot_spend_conditional_hours_or_expand_caps(owned,mode):
    catalog=d0_catalog_fixture(owned);inputs={} if mode=='timing-references' else dict(catalog=str(catalog),family='jab')
    wall,app=(900,840) if mode=='timing-references' else (480,390)
    with pytest.raises(ValueError,match='Conditional hours'):launcher.validate_spec(mode,inputs,wall,app,proof={'synthetic':True})
    with pytest.raises(ValueError):launcher.validate_spec(mode,inputs,wall+60,app)
    with pytest.raises(ValueError):launcher.validate_spec(mode,inputs,wall,app+1)


@pytest.mark.parametrize('tamper',['retimed_row','duplicate_family','missing_family','horizon','launch','target','active_hand','guard','predicate','reference','legacy'])
def test_frozen_control_d0_catalog_cannot_change_original_requests_or_physical_inputs(owned,tamper):
    root,_,_=owned;path=d0_catalog_fixture(owned);catalog=json.loads(path.read_text());row=next(r for r in catalog['cells'] if r['family']=='jab')
    if tamper=='retimed_row':catalog['cells'][-1]['kind']='uniform'
    elif tamper=='duplicate_family':catalog['cells'].append(dict(row))
    elif tamper=='missing_family':catalog['cells']=[r for r in catalog['cells'] if r['family']!='jab']
    elif tamper=='horizon':row['episode_duration_s']-=.02
    elif tamper=='launch':row['launch_tick']+=4
    elif tamper=='target':row['target_w'][0]+=.01
    elif tamper=='active_hand':row['active_hand_index']=1
    elif tamper=='guard':row['guard_time_s']+=.01
    elif tamper=='predicate':catalog['predicate_sha256']='e'*64
    elif tamper in ('reference','legacy'):
        key='reference' if tamper=='reference' else 'legacy_runtime';original=Path(row[key+'_path'])
        other=root/'outputs/task10b/alternate.npz';other.write_bytes(original.read_bytes());row[key+'_path']=str(other)
    dump(path,catalog)
    with pytest.raises(ValueError):launcher.validate_spec('frozen-control',dict(catalog=str(path),family='jab'),480,390)


def test_d0_frozen_source_closure_reuses_original_frozen_dependency_contract(owned,monkeypatch):
    root,_,_=owned;catalog=d0_catalog_fixture(owned);calls=[]
    monkeypatch.setattr(launcher.base,'required_inputs',lambda record:(calls.append(record) or set()))
    launcher.required_inputs(dict(mode='frozen-control',inputs=dict(catalog=str(catalog),family='jab'),
        bundle='outputs/task10/release_baseline_02',release='outputs/four-punch-milestone',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json'))
    assert calls[0]['mode']=='frozen' and calls[0]['catalog']==str(catalog.relative_to(root))


def test_bare_success_boolean_never_unlocks_conditional_hours(owned):
    _,run,_=owned
    proof=dump(run/'proof.json',{'successful_route_frozen':True})
    with pytest.raises(ValueError,match='Four-family'):launcher.successful_route_freeze(proof)
    with pytest.raises(ValueError,match='Conditional hours'):launcher.validate_spec('target-references',{},900,800,proof={'families':{}})


def test_rounded_submission_uses_shared_accounting_schema_and_failed_response_retains_intent(owned,monkeypatch):
    root,run,panel=owned;out=root/'outputs/task10b/p2submit_fixture'
    result=launcher.prepare(out,'media',{'run':str(run),'panel':str(panel)},allocation_wall_seconds=421,application_wall_seconds=400)
    monkeypatch.setattr(launcher.base,'ledger',lambda:[])
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='123456\n',stderr=''))
    receipt=launcher.submit(result['packet'])
    assert receipt['schema']=='task10b_phase2_submission_v1' and receipt['reserved_gpu_seconds']==480
    assert receipt['budget_before']['ceiling_gpu_seconds']==14400
    assert json.loads((out/'submission_intent.json').read_text())['reserved_gpu_seconds']==480
    with pytest.raises(ValueError,match='durable'):launcher.submit(result['packet'])


def test_partition_capacity_and_initial_budget_reject_before_scheduler_contact(owned,monkeypatch):
    root,run,panel=owned;out=root/'outputs/task10b/p2capacity_fixture'
    result=launcher.prepare(out,'media',{'run':str(run),'panel':str(panel)},allocation_wall_seconds=480,application_wall_seconds=400)
    for i in range(2):dump(root/f'outputs/task10b/p2existing{i}/submission_intent.json',{'command':['--partition=mit_normal_gpu']})
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:pytest.fail('Scheduler must remain untouched'))
    with pytest.raises(ValueError,match='Concurrent'):launcher.submit(result['packet'])
    assert not (out/'submission_intent.json').exists()


def test_initial_budget_overflow_and_ambiguous_scheduler_reply_preserve_correct_charge(owned,monkeypatch):
    root,run,panel=owned;out=root/'outputs/task10b/p2budget_fixture'
    result=launcher.prepare(out,'media',{'run':str(run),'panel':str(panel)},allocation_wall_seconds=480,application_wall_seconds=400)
    original_ledger=launcher.base.ledger
    from human_punching.phase2_budget import Allocation
    monkeypatch.setattr(launcher.base,'ledger',lambda:[Allocation('prior',14000)])
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:pytest.fail('No budget for scheduler contact'))
    with pytest.raises(ValueError,match='remaining'):launcher.submit(result['packet'])
    assert not (out/'submission_intent.json').exists()
    monkeypatch.setattr(launcher.base,'ledger',original_ledger)
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='Possible job123\n',stderr='lost acknowledgement'))
    with pytest.raises(RuntimeError,match='uncertain'):launcher.submit(result['packet'])
    assert (out/'submission_uncertain.json').exists() and not (out/'submission_rejected.json').exists()
    assert launcher.base.ledger()[0].charged_seconds==480


def test_native_execution_requires_an_actual_allocation_before_any_process(owned,monkeypatch):
    _,run,panel=owned
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(RuntimeError,match='Slurm'):launcher.execute(panel)
    assert not (run/'execution_started.json').exists()


def test_native_parity_receipt_requires_actual_passing_xml_and_current_source(owned):
    root,run,_=owned
    source=root/'src/human_punching/phase2_native_state.py';source.parent.mkdir(parents=True);source.write_text('synthetic source')
    xml=run/'cpu.xml';xml.write_text('<testsuites><testsuite tests="23" failures="0" errors="0" skipped="0"/></testsuites>')
    inventory=dump(run/'inventory.json',{'state_spec':16383,'state_size':410})
    binding=lambda p:{'path':str(p),'sha256':launcher.sha256_file(p)}
    receipt=dump(run/'implementation.json',dict(schema='task10b_phase2_native_state_implementation_v1',source=binding(source),
        native_execution_performed=False,native_parity_qualified=False,compiled_state_spec=16383,compiled_state_size=410,
        focused_tests=binding(xml)|dict(tests=23,failures=0,errors=0,skipped=0),accepted_cpu_model_inventory=binding(inventory)))
    assert launcher.parity_implementation(receipt)['compiled_state_size']==410
    xml.write_text('<testsuites><testsuite tests="23" failures="1" errors="0" skipped="0"/></testsuites>')
    bad=json.loads(receipt.read_text());bad['focused_tests']['sha256']=launcher.sha256_file(xml);dump(receipt,bad)
    with pytest.raises(ValueError,match='did not pass'):launcher.parity_implementation(receipt)


def test_test_bank_capture_freeze_fails_before_any_physical_payload_read(owned,monkeypatch):
    _,run,_=owned;freeze=dump(run/'freeze.json',{'not':'a measured freeze'})
    from human_punching import phase2_native_state
    def fail_freeze(path):raise ValueError('Frozen route is absent')
    monkeypatch.setattr(phase2_native_state,'validate_test_bank_freeze',fail_freeze)
    monkeypatch.setattr(launcher,'read',lambda *a:pytest.fail('Physical inputs must remain unopened'))
    with pytest.raises(ValueError,match='Frozen route'):
        launcher.validate_spec('shared-state-capture',dict(split='test',catalog='unopened',domain='unopened',geometry='unopened',
            implementation_receipt='unopened',pre_test_freeze=str(freeze)),480,400)


def test_shared_entrypoints_are_selected_from_fixed_schemas(owned):
    root,run,panel=owned
    configuration=dump(run/'shared_training.json',{'schema':'task10b_phase2_shared_training_configuration_v1'})
    r=dict(mode='grouped-pilot',inputs={'configuration':str(configuration)},output='outputs/task10b/p2r/run',application_wall_seconds=400,
        release='outputs/four-punch-milestone',
        files={str(configuration.relative_to(root)):launcher.sha256_file(configuration)})
    assert launcher.native_command(r)[3]=='human_punching.phase2_shared_learning'
    plan=dump(run/'shared_reference.json',{'schema':'task10b_phase2_shared_catalog_plan_v1'})
    r.update(mode='target-references',inputs={'plan':str(plan)})
    assert launcher.native_command(r)[2].endswith('task10b_phase2_shared_references.py')
    declaration=dump(run/'shared_evaluation.json',{'start_mode':'shared_jab_saved_state'})
    r.update(mode='expanded-evaluation',inputs={k:str(declaration if k=='declaration' else panel) for k in launcher.EVAL_KEYS})
    assert launcher.native_command(r)[3]=='human_punching.phase2_shared_evaluation'
    r.update(mode='shared-state-capture',inputs=dict(catalog=str(panel),domain=str(panel),geometry=str(panel),split='train'))
    command=launcher.native_command(r)
    assert command[3]=='human_punching.phase2_native_state' and '--capture-bank-domain' in command
    assert command[command.index('--family')+1]=='jab'


def test_source_pool_copies_once_and_live_edits_cannot_modify_past_archives(owned):
    root,run,_=owned;source=run/'source.txt';source.write_text('first immutable bytes')
    expected=launcher.sha256_file(source);before=source.stat()
    first=launcher.snapshot_object(source,root/'archive_one/source.txt',expected)
    second=launcher.snapshot_object(source,root/'archive_two/source.txt',expected)
    after=source.stat();obj=Path(first['object'])
    assert first['new_physical_object'] and not second['new_physical_object']
    assert first['object']==second['object']
    assert source.stat().st_ino!=obj.stat().st_ino
    assert (root/'archive_one/source.txt').stat().st_ino==(root/'archive_two/source.txt').stat().st_ino==obj.stat().st_ino
    assert (before.st_ino,before.st_nlink,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_nlink,after.st_mtime_ns,after.st_ctime_ns)
    assert obj.stat().st_mode&0o222==0
    source.write_text('new live source version')
    assert (root/'archive_one/source.txt').read_text()=='first immutable bytes'
    third=launcher.snapshot_object(source,root/'archive_three/source.txt',launcher.sha256_file(source))
    assert third['object']!=first['object'] and obj.read_text()=='first immutable bytes'


def test_source_closure_follows_lazy_local_imports_without_importing_modules(owned):
    root,_,_=owned;folder=root/'src/human_punching';folder.mkdir(parents=True)
    (folder/'entry.py').write_text('def later():\n from .phase2_packing import pack\n')
    (folder/'phase2_packing.py').write_text('from human_punching.phase2_extra import helper\nraise RuntimeError("must not import")\n')
    (folder/'phase2_extra.py').write_text('helper=42\n')
    paths=launcher.local_import_closure({'src/human_punching/entry.py'})
    assert paths=={'src/human_punching/entry.py','src/human_punching/phase2_packing.py','src/human_punching/phase2_extra.py'}


def synthetic_freeze(owned):
    """Consistent fake receipts for testing the source gate, never native evidence."""
    import torch
    from dataclasses import asdict
    from human_punching.phase2_domain import fingerprint
    from human_punching.wbc.splits import assign_trajectory_groups
    from human_punching.phase2_qualification import qualify_panel
    spec=importlib.util.spec_from_file_location('qualification_fixture',Path(__file__).with_name('test_phase2_qualification.py'))
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    root,run,_=owned;assignment=assign_trajectory_groups(range(1000));groups=[]
    for split,count,replica in (('train',32,0),('validation',8,1),('test',16,2)):
        selected=[i for i,s in assignment.items() if s==split][:count]
        groups.extend(dict(split=split,snapshot=dict(command_id=i,replica_id=replica)) for i in selected)
    domain=dump(run/'domain.json',dict(schema='Task10BPhase2Domain/v1',selected_command_groups=groups));domain_sha=launcher.sha256_file(domain)
    split_rows=[dict(command_id=g['snapshot']['command_id'],replica_id=g['snapshot']['replica_id'],split=g['split']) for g in groups]
    bind=lambda p:dict(path=str(p),sha256=launcher.sha256_file(p));family_bindings={}
    for family in launcher.base.FAMILIES:
        folder=run/family;folder.mkdir();parent=root/'outputs/four-punch-milestone'/family/'checkpoint.pt'
        parent.parent.mkdir(parents=True,exist_ok=True);parent.write_bytes(('accepted_fixture_'+family).encode())
        ancestry=dict(sha256=launcher.sha256_file(parent),seed=21021,fresh_optimizer=True,fresh_simulation=True,independent_scratch_replication=False)
        checkpoint=folder/'checkpoint.pt'
        torch.save(dict(schema='task10b_phase2_original_ppo_v1',observations=369,actions=48,fresh_transitions=8192,
            actor_parameters_sha256='f'*64,ancestry=ancestry),checkpoint)
        common=dict(recipe_sha256='1'*64,domain_sha256=domain_sha,target_geometry_sha256='e'*64)
        training=dump(folder/'training.json',dict(fresh_transitions=8192,weight_changed=True,status='bounded_stop',exception=None,
            hard_faults=[],final_policy_digest='f'*64,checkpoint=bind(checkpoint),reload_parity=dict(parameters_equal=True,actions_equal=True),
            ancestry=ancestry,trained_timings_s=[.6,.75],interpolation_used_for_training=False,**common))
        decl=replace(fixtures.declaration(family=family,split='validation'),checkpoint_sha256=launcher.sha256_file(checkpoint),
            parent_checkpoint_sha256=launcher.sha256_file(parent),training_receipt_sha256=launcher.sha256_file(training))
        report=dump(folder/'qualification.json',qualify_panel(decl,fixtures.success_rows(decl)))
        launch=dump(folder/'launch.json',dict(schema='task10b_phase2_expanded_native_launch_v1',declaration_identity=decl.identity,
            slurm_job_id='123456',native_execution_requested=True,release_inputs=dict(checkpoint=bind(parent)),**common))
        trace=folder/'batch_0000/raw_trace.npz';trace.parent.mkdir();trace.write_bytes(b'not a native trace; synthetic gate fixture')
        audit=dump(folder/'audit.json',dict(schema='task10b_phase2_independent_trace_audit_v1',passed=True,
            closed_protocol_launch_audited=True,outcome_resolution_passed=True,
            trial_outcome_resolutions=[dict(trial_id=b.trial_id,request_identity=b.identity,outcome_resolution='audited_complete',
                resolved=True,attempted=True,execution_complete=True,raw_trace_sha256=launcher.sha256_file(trace),
                prefix_integrity_pass=True,exact_counter_pass=True,safe_actor_boundary_pass=True,hard_faults_absent=True) for b in decl.bindings],
            run=str(folder),panels=[dict(path=str(trace.parent),raw_sha256=launcher.sha256_file(trace),
                passed=True,execution_complete=True,planned_trials=96,trials=[dict(trial_id=b.trial_id,passed=True) for b in decl.bindings])]))
        route=dump(folder/'route.json',dict(schema='task10b_phase2_evaluation_route_v1',family=family,state='frozen',selection_splits=['train','validation'],test_used_for_selection=False,
            interpolation_used_for_training=False,predicate_sha256=decl.predicate_sha256,declared_mask_sha256=decl.declared_mask_sha256,
            successful_validation={k:bind(p) for k,p in dict(qualification=report,trace_audit=audit,launch=launch,training=training).items()},**common))
        family_bindings[family]=bind(route)
    proof=dump(run/'freeze.json',dict(schema='task10b_phase2_successful_route_freeze_v1',selection_splits=['train','validation'],
        test_used_for_selection=False,domain=bind(domain),split_identity=fingerprint(split_rows),families=family_bindings))
    return proof


def test_conditional_gate_accepts_complete_consistent_four_family_source_receipts(owned):
    proof=synthetic_freeze(owned)
    result=launcher.successful_route_freeze(proof)
    assert set(result['families'])==set(launcher.base.FAMILIES)
    assert launcher.budget_status([],successful_route_frozen=True)['ceiling_gpu_seconds']==100800
    record=json.loads(proof.read_text());del record['families']['right_hook'];dump(proof,record)
    with pytest.raises(ValueError,match='Four-family'):launcher.successful_route_freeze(proof)


def test_conditional_gate_rejects_failed_report_with_an_unchanged_old_report_identity(owned):
    proof=synthetic_freeze(owned);record=json.loads(proof.read_text());route_path=Path(record['families']['jab']['path'])
    route=json.loads(route_path.read_text());report_path=Path(route['successful_validation']['qualification']['path']);report=json.loads(report_path.read_text())
    report['successes']=0;dump(report_path,report)
    route['successful_validation']['qualification']['sha256']=launcher.sha256_file(report_path);dump(route_path,route)
    record['families']['jab']['sha256']=launcher.sha256_file(route_path);dump(proof,record)
    with pytest.raises(ValueError,match='evidence is incomplete'):launcher.successful_route_freeze(proof)


def test_conditional_gate_preserves_a_resolved_failed_prefix_in_the96_attempt_denominator(owned):
    from human_punching.phase2_qualification import PanelDeclaration,TrialBinding,qualify_panel
    proof=synthetic_freeze(owned);record=json.loads(proof.read_text());route_path=Path(record['families']['jab']['path'])
    route=json.loads(route_path.read_text());report_path=Path(route['successful_validation']['qualification']['path']);report=json.loads(report_path.read_text())
    audit_path=Path(route['successful_validation']['trace_audit']['path']);audit=json.loads(audit_path.read_text())
    resolution=audit['trial_outcome_resolutions'][0]
    resolution.update(outcome_resolution='audited_failed_prefix',execution_complete=False)
    audit['panels'][0]['execution_complete']=False;dump(audit_path,audit)
    declaration=PanelDeclaration(**{**report['declaration'],'bindings':tuple(TrialBinding(**b) for b in report['declaration']['bindings'])})
    outcomes=[row['source_outcome'] for row in report['trials']]
    outcomes[0].update(status='timeout',audit_resolution=resolution,visual_review='reviewed_failure',execution_complete=False,
        observed_active_glove_rms_m=.01,observed_outward_active_glove_rms_m=.01,
        observed_physical_metrics=dict(execution_complete=False,scope='retained_prefix',active_sample_count=101,
            active_squared_error_sum_m2=.0101,outward_active_sample_count=31,outward_active_squared_error_sum_m2=.0031))
    corrected=qualify_panel(declaration,outcomes);assert corrected['passed'] and corrected['successes']==95
    dump(report_path,corrected)
    route['successful_validation']['qualification']['sha256']=launcher.sha256_file(report_path)
    route['successful_validation']['trace_audit']['sha256']=launcher.sha256_file(audit_path);dump(route_path,route)
    record['families']['jab']['sha256']=launcher.sha256_file(route_path);dump(proof,record)
    assert launcher.successful_route_freeze(proof)['families']['jab']['start_mode']=='family_guard'
