"""Immutable measured offline catalog projections; no actor/physics execution."""
import importlib.util
import json
from pathlib import Path

import pytest

from human_punching import phase2_timing_control as timing

ROOT=timing.ROOT
SPEC=importlib.util.spec_from_file_location('timing_driver',ROOT/'scripts/task10b_phase2_expand.py')
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


@pytest.fixture(scope='module')
def catalog(tmp_path_factory):
    folder=tmp_path_factory.mktemp('portable_timing')
    result=timing.project(ROOT/'outputs/task10b/p2r05n/run/manifest.json',ROOT/'outputs/task10b/p2hostalias01/probe.json',
        ROOT/'outputs/task10b/p2r05n/execution.json',folder/'view')
    return timing.owned(result['path'])


def test_projection_retains_all_scientific_metadata_arrays_and_uniform_derivatives(catalog):
    data,evidence=timing.validate(catalog)
    assert len(data['cells'])==len(evidence)==16
    assert sum(e.get('derivative_reconstruction_passed') is True for e in evidence)==12
    assert sum(e.get('unchanged_original_control') is True for e in evidence)==4
    assert data['timing_control_projection']['scientific_metadata_or_arrays_changed'] is False
    assert all(not Path(item['path']).is_absolute() for e in evidence for item in e['files'].values())


def test_projection_and_evidence_are_identical_in_both_proved_root_namespaces(catalog):
    spec=importlib.util.spec_from_file_location('human_punching.foreign_timing_control',
        Path('/orcd/home/002/kaisyun/Azizan_Lab/HumanBox/human-punching/src/human_punching/phase2_timing_control.py'))
    foreign=importlib.util.module_from_spec(spec);spec.loader.exec_module(foreign)
    relative=catalog.relative_to(ROOT)
    assert timing.validate(catalog)==foreign.validate(foreign.ROOT/relative)


@pytest.mark.parametrize('change',['ttc','event','duration','source_hash','admission','target','drop','duplicate','new_path','mapping','source_catalog_hash'])
def test_projection_rejects_any_scientific_identity_or_mapping_change(catalog,tmp_path,change):
    data=json.loads(catalog.read_text());row=next(r for r in data['cells'] if r['kind']=='uniform')
    if change=='ttc':row['requested_ttc_s']+=.05
    elif change=='event':row['launch_tick']+=4
    elif change=='duration':row['phase_denominator_s']-=.02
    elif change=='source_hash':row['source_reference_sha256']='d'*64
    elif change=='admission':row['kinematic_admission']=False
    elif change=='target':row['target_w'][0]+=.01
    elif change=='drop':data['cells'].pop()
    elif change=='duplicate':data['cells'][-1]=dict(row)
    elif change=='new_path':row['unreviewed_path']=str(ROOT/'outputs/task10b/p2r05n/run/manifest.json')
    elif change=='mapping':data['timing_control_projection']['mapped_files'][0]['sha256']='e'*64
    else:data['unpacked_catalog_sha256']='f'*64
    with pytest.raises(ValueError):timing.validate(save(tmp_path/'manifest.json',data))


def test_timing_mode_has_no_training_or_conditional_authority_and_keeps_original_only_mode_strict(catalog):
    inputs=dict(catalog=str(catalog),family='jab');scope=driver.validate_spec('timing-control',inputs,480,390)
    assert scope['family']=='jab' and scope['fresh_transition_cap']==0 and len(scope['timing_control_evidence'])==16
    for wall,app in ((540,390),(480,391)):
        with pytest.raises(ValueError):driver.validate_spec('timing-control',inputs,wall,app)
    with pytest.raises(ValueError,match='Conditional hours'):driver.validate_spec('timing-control',inputs,480,390,proof={'invalid':True})
    with pytest.raises(ValueError):driver.validate_spec('frozen-control',inputs,480,390)
    record=dict(mode='timing-control',inputs=inputs,application_wall_seconds=390,family='jab',output='outputs/task10b/synthetic/run',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    command=driver.native_command(record)
    assert command[command.index('--mode')+1]=='frozen' and command[command.index('--max-updates')+1]=='0'
    assert float(command[command.index('--wall-seconds')+1])==360.


def test_source_closure_binds_all_screen_search_event_arrays_without_following_foreign_receipt_paths(catalog):
    inputs=dict(catalog=str(catalog),family='jab');scope=driver.validate_spec('timing-control',inputs,480,390)
    record=dict(mode='timing-control',inputs=inputs,bundle='outputs/task10/release_baseline_02',release='outputs/four-punch-milestone',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',**scope)
    provenance={};files=driver.required_inputs(record,provenance_files=provenance)
    for cell in scope['timing_control_evidence']:
        for item in cell['files'].values():assert files[item['path']]==item['sha256']
    assert 'outputs/task10b/p2r05n/run/manifest.json' in provenance
    assert 'outputs/task10b/p2hostalias01/nonce.json' in files
    assert not any(name.startswith('/orcd') for name in files)


def test_explicit_generation_failure_remains_in_the_sixteen_cell_denominator(tmp_path):
    original=json.loads((ROOT/'outputs/task10b/p2r05n/run/manifest.json').read_text())
    row=next(r for r in original['cells'] if r['kind']=='uniform')
    row.update(reference_path=None,kinematic_admission=False,rejection_reasons=['reference_generation_failure'],generation_disposition='failed')
    for key in ('reference_sha256','legacy_runtime_path','legacy_runtime_sha256','diagnostic_screen_path'):row.pop(key,None)
    packet=save(tmp_path/'packet.json',dict(mode='timing-references',synthetic_negative_fixture=True))
    source=save(tmp_path/'run/manifest.json',original);save(tmp_path/'run'/row['candidate_id']/'failure.json',row)
    execution=json.loads((ROOT/'outputs/task10b/p2r05n/execution.json').read_text());execution['packet_sha256']=timing.sha256_file(packet)
    execution=save(tmp_path/'execution.json',execution)
    view=timing.project(source,ROOT/'outputs/task10b/p2hostalias01/probe.json',execution,tmp_path/'view')
    data,evidence=timing.validate(timing.owned(view['path']))
    failed=next(e for e in evidence if e['candidate_id']==row['candidate_id'])
    assert len(data['cells'])==16 and failed['offline_rejection'] and failed['native_attempt'] is False
