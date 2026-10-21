"""Actual saved catalog closure and mocked dispatch; no runtime packing/native."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from human_punching import phase2_sequential_pipeline_inputs_v1 as inputs
from human_punching import phase2_runtime_pack_entry_v3 as entry
from human_punching.wbc.provenance import PROJECT_ROOT as ROOT,sha256_file


def binding(path):return dict(path=str(path),sha256=sha256_file(path))


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    path=tmp_path_factory.mktemp('plans')/'plan.json'
    source=json.loads((ROOT/'outputs/task10b/p2sequentialfullprep02/train_plan.json').read_text())
    from human_punching.phase2_runtime_pack import VERSION,CAP
    data=dict(schema='task10b_phase2_runtime_pack_plan_v1',storage_version=VERSION,byte_cap=CAP,
        catalogs=[binding(ROOT/f'outputs/task10b/p2sequentialcatalogfull01{s}/manifest.json') for s in ('t','v')],
        recipes=[source['recipe']],previous_packed_catalogs=[],
        sequential_reference_audits=[binding(ROOT/f'outputs/task10b/p2sequentialfullaudit01{s}/audit.json') for s in ('t','v')])
    path.write_text(json.dumps(data));return path,data


@pytest.fixture(scope='module')
def driver():
    spec=importlib.util.spec_from_file_location('test_sequential_pipeline',ROOT/'scripts/task10b_phase2_sequential_pipeline_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_real_88_source_closure_and_opaque_archive_preservation(plan):
    path,data=plan;provenance={};original=inputs.parent.Inputs
    result=inputs.files('runtime-pack',{'plan':str(path)},provenance_files=provenance)
    assert inputs.parent.Inputs is original
    for item in [*data['catalogs'],*data['recipes'],*data['sequential_reference_audits']]:
        assert result[str(Path(item['path']).relative_to(ROOT))]==item['sha256']
    assert len([n for n in result if n.endswith('/reference.npz') and '/p2sg01' in n])==88
    assert provenance and all('verified_sequential_generation_archive' in x['roles'] for x in provenance.values())
    assert all('source_snapshot' in Path(n).parts for n in provenance)
    assert not any('gridsan' in n for n in result)


@pytest.mark.parametrize('change',['missing','old_target','timing','probe_schema','changed_view'])
def test_no_schema_masquerade_or_projection_substitution(tmp_path,monkeypatch,change):
    catalog=dict(sequential_catalog_projection={})
    receipt=dict(packet_schema=inputs.FULL_SCHEMA)
    if change=='missing':catalog={}
    elif change=='old_target':catalog['target_catalog_projection']={}
    elif change=='timing':catalog['timing_control_projection']={}
    elif change=='probe_schema':receipt['packet_schema']='task10b_phase2_sequential_reference_packet_v3'
    view=deepcopy(catalog)
    if change=='changed_view':view['cells']=[]
    monkeypatch.setattr(inputs,'verify_projection',lambda _: (view,receipt))
    monkeypatch.setattr(inputs,'projection_files',lambda _: pytest.fail('must reject before traversing'))
    with pytest.raises(ValueError):inputs.Inputs(tmp_path).projection(tmp_path/'not_read.json',catalog)


def test_dependency_restored_after_failure(monkeypatch):
    original=inputs.parent.Inputs
    def fail(*a,**k):raise ValueError('synthetic metadata failure')
    monkeypatch.setattr(inputs.parent,'files',fail)
    with pytest.raises(ValueError):inputs.files('runtime-pack',{})
    assert inputs.parent.Inputs is original


def test_runtime_and_grouped_dispatch_stay_bounded(plan,driver,tmp_path):
    path,data=plan
    scope=driver.validate_spec('runtime-pack',{'plan':str(path)},300,240)
    assert scope['fresh_transition_cap']==0 and scope['pipeline_inputs_version']==inputs.VERSION
    with pytest.raises(ValueError):driver.validate_spec('runtime-pack',{'plan':str(path)},360,300)
    (tmp_path/'packet.json').write_text('{}')
    record=dict(mode='runtime-pack',inputs={'plan':str(path)},output=str((tmp_path/'run').relative_to(ROOT)),
        allocation_wall_seconds=300,application_wall_seconds=240,
        runtime_namespace_proof=binding(ROOT/'outputs/task10b/p2hostalias01/probe.json'))
    command=driver.native_command(record,180)
    assert 'human_punching.phase2_runtime_pack_entry_v3' in command and float(command[command.index('--wall-seconds')+1])==180.
    assert str(ROOT/'outputs/task10b/p2hostalias01/probe.json') in command
    with pytest.raises(ValueError):driver.native_command(record,119.)
    grouped=deepcopy(record);grouped.update(mode='grouped-pilot',inputs={'configuration':str(path)},files={str(path.relative_to(ROOT)):sha256_file(path)})
    assert 'human_punching.phase2_grouped_entry_v2' in driver.native_command(grouped,180)


def test_actual_driver_closure_captures_new_sources_and_precise_ancestry(plan,driver):
    path,data=plan;proof=ROOT/'outputs/task10b/p2hostalias01/probe.json'
    from human_punching.phase2_runtime_namespace import evidence
    record=dict(mode='runtime-pack',inputs={'plan':str(path)},bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',
        successful_route_freeze=None,runtime_namespace_proof=dict(path=str(proof.relative_to(ROOT)),sha256=sha256_file(proof)),
        runtime_namespace_evidence=evidence(ROOT,proof))
    provenance={};result=driver.required_inputs(record,provenance_files=provenance)
    for name in ('src/human_punching/phase2_runtime_pack_entry_v3.py',
                 'src/human_punching/phase2_sequential_projection_v1.py',
                 'scripts/task10b_phase2_sequential_pipeline_v1.sbatch'):
        assert result[name]==sha256_file(ROOT/name)
    assert provenance and all(n not in result for n in provenance)


def test_runtime_adapter_refuses_old_packet_schema_before_pack(tmp_path,monkeypatch):
    packet=tmp_path/'packet.json';packet.write_text(json.dumps(dict(schema='task10b_phase2_expansion_packet_v2',output=str((tmp_path/'run').relative_to(ROOT)))))
    plan=tmp_path/'plan.json';plan.write_text('{}');proof=tmp_path/'proof.json';proof.write_text('{}')
    monkeypatch.setattr(entry.packing,'pack',lambda *a,**k:pytest.fail('no packing'))
    with pytest.raises(ValueError,match='immutable'):
        entry.execute(plan,tmp_path/'run',wall_seconds=180,logical_project_root=ROOT,runtime_namespace_proof=proof,
            packet_path=packet,packet_sha256=sha256_file(packet))


@pytest.mark.parametrize('change',['missing','failed','native_mismatch','physics','lost_cell','bad_cell','reverse_splits','extra_recipe'])
def test_exact_saved_audit_gate(plan,tmp_path,monkeypatch,change):
    _,data=plan;data=deepcopy(data)
    monkeypatch.setattr(inputs.parent,'files',lambda *a,**k:{})
    if change=='missing':data['sequential_reference_audits'].pop()
    elif change=='reverse_splits':data['catalogs'].reverse()
    elif change=='extra_recipe':data['recipes']*=2
    else:
        audit=json.loads(Path(data['sequential_reference_audits'][0]['path']).read_text())
        if change=='failed':audit['passed']=False
        elif change=='native_mismatch':audit['manifest_sha256']='0'*64
        elif change=='physics':audit['physics_steps']=1
        elif change=='lost_cell':audit['cells'].pop()
        elif change=='bad_cell':audit['cells'][0]['all_checks_passed']=False
        path=tmp_path/'audit.json';path.write_text(json.dumps(audit));data['sequential_reference_audits'][0]=binding(path)
    path=tmp_path/'plan.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError):inputs.files('runtime-pack',{'plan':str(path)})


def test_grouped_source_factory_and_exact_selection_edges(tmp_path,monkeypatch):
    monkeypatch.setattr(inputs.parent,'files',lambda *a,**k:{})
    factory=ROOT/'outputs/task10b/p2jabgroupedseqprep01/create_configuration.py'
    selection_path=ROOT/'outputs/task10b/p2jabgroupedseqprep01/selection.json'
    path=tmp_path/'configuration.json';path.write_text(json.dumps(dict(configuration_factory=binding(factory),
        sequential_selection=binding(selection_path))))
    closure=inputs.files('grouped-pilot',{'configuration':str(path)})
    assert closure[str(factory.relative_to(ROOT))]==sha256_file(factory)
    selection=json.loads(selection_path.read_text())
    for item in (selection['original_selection'],selection['full_recipe'],*selection['full_plans'].values(),
                 *selection['projected_catalogs'].values(),*selection['saved_audits'].values()):
        assert closure[str(Path(item['path']).relative_to(ROOT))]==item['sha256']
    data=json.loads(path.read_text());data['configuration_factory']['sha256']='0'*64;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='SHA'):inputs.files('grouped-pilot',{'configuration':str(path)})
