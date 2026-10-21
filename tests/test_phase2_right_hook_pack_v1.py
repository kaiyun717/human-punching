"""Metadata, deadline and unchanged precision-conversion checks; no native run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from human_punching import phase2_right_hook_pack_inputs_v1 as inputs
from human_punching import phase2_right_hook_pack_v1 as entry
from human_punching.wbc.provenance import PROJECT_ROOT as ROOT, sha256_file


def save(path, data):
    path.write_text(json.dumps(data)); return dict(path=str(path), sha256=sha256_file(path))


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs.parent, 'runtime_pack_files', lambda *a: None)
    catalogs=[];audits=[];recipe=save(tmp_path/'recipe.json',{})
    for split,count,ts in [('train',64,[1.,1.2]),('validation',24,[1.,1.1,1.2])]:
        rows=[dict(candidate_id=f'{split}{i}', family='right_hook',split=split,
                   requested_ttc_s=ts[i%len(ts)],recipe_sha256=recipe['sha256']) for i in range(count)]
        audit=save(tmp_path/f'{split}_audit.json',dict(schema=inputs.AUDIT,passed=True,
            family='right_hook',split=split,physics_steps=0,manifest_sha256='b'*64,
            cells=[dict(candidate_id=r['candidate_id'],all_checks_passed=True) for r in rows]))
        catalog=save(tmp_path/f'{split}.json',dict(cells=rows,recipe_sha256=recipe['sha256'],
            unpacked_catalog_sha256='b'*64,right_hook_full_catalog_projection=dict(saved_audit=dict(
                relative_name=str(Path(audit['path']).relative_to(ROOT)),sha256=audit['sha256']))))
        catalogs.append(catalog);audits.append(audit)
    data=dict(packing_inputs_version=inputs.VERSION,family='right_hook',catalogs=catalogs,
        right_hook_reference_audits=audits,recipes=[recipe],previous_packed_catalogs=[],
        physical_qualification=False,training_authorized_by_packing=False)
    path=tmp_path/'plan.json';save(path,data);return path,data


def test_exact_88_common_recipe_saved_audit_closure(plan):
    path,data=plan;closure=inputs.files(inputs.MODE,{'plan':str(path)})
    for item in [*data['catalogs'],*data['right_hook_reference_audits']]:
        assert closure[str(Path(item['path']).relative_to(ROOT))]==item['sha256']


@pytest.mark.parametrize('change',['train_interior','reverse','failed_audit','wrong_ids','different_recipe',
    'wrong_family','physics','bad_cell','missing_audit','previous_pool','old_version','promotion'])
def test_wrong_science_or_audit_fails_closed(plan,change):
    path,data=plan
    if change=='reverse':data['catalogs'].reverse()
    elif change=='missing_audit':data['right_hook_reference_audits'].pop()
    elif change=='previous_pool':data['previous_packed_catalogs']=[{}]
    elif change=='old_version':data['packing_inputs_version']='old'
    elif change=='promotion':data['training_authorized_by_packing']=True
    elif change in ('train_interior','different_recipe','wrong_family'):
        item=data['catalogs'][0];p=Path(item['path']);view=json.loads(p.read_text())
        key,value={'train_interior':('requested_ttc_s',1.1),'different_recipe':('recipe_sha256','a'*64),
                   'wrong_family':('family','jab')}[change]
        view['cells'][0][key]=value;data['catalogs'][0]=save(p,view)
    else:
        item=data['right_hook_reference_audits'][0];p=Path(item['path']);audit=json.loads(p.read_text())
        if change=='failed_audit':audit['passed']=False
        elif change=='wrong_ids':audit['cells'][0]['candidate_id']='different'
        elif change=='physics':audit['physics_steps']=1
        elif change=='bad_cell':audit['cells'][0]['all_checks_passed']=False
        data['right_hook_reference_audits'][0]=save(p,audit)
    save(path,data)
    with pytest.raises(ValueError):inputs.files(inputs.MODE,{'plan':str(path)})


def test_projection_opaque_archive_and_schema(tmp_path,monkeypatch):
    catalog={'right_hook_full_catalog_projection':{}}
    archive=tmp_path/'source_snapshot';archive.mkdir();source=archive/'source.py';source.write_text('old')
    name=str(source.relative_to(ROOT));digest=sha256_file(source)
    monkeypatch.setattr(inputs,'verify_projection',lambda _: (catalog,dict(packet_schema=inputs.FULL_SCHEMA)))
    monkeypatch.setattr(inputs,'projection_files',lambda _: {name:digest})
    provenance={};collector=inputs.Inputs(ROOT,provenance);collector.projection(tmp_path/'view.json',catalog)
    assert collector.finish()=={} and provenance[name]['roles']==['verified_right_hook_generation_archive']
    with pytest.raises(ValueError):collector.projection(tmp_path/'view.json',dict(catalog,family_catalog_projection={}))


@pytest.fixture(scope='module')
def driver():
    spec=importlib.util.spec_from_file_location('rh_pack_tests_driver',ROOT/'scripts/task10b_phase2_right_hook_pack_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def record(tmp_path,driver):
    p=tmp_path/'plan.json';p.write_text('{}');(tmp_path/'packet.json').write_text('{}')
    proof=ROOT/'outputs/task10b/p2hostalias01/probe.json'
    return dict(mode=driver.MODE,inputs={'plan':str(p)},family='right_hook',training=False,
        fresh_transition_cap=0,packing_child_wall_seconds=60,evaluation_reserve_seconds=4500,
        allocation_wall_seconds=240,application_wall_seconds=210,physical_qualification=False,
        training_authorized_by_packing=False,output=str((tmp_path/'run').relative_to(ROOT)),
        runtime_namespace_proof=dict(path=str(proof.relative_to(ROOT)),sha256=sha256_file(proof)))


def test_fixed_deadline_and_no_learning_dispatch(driver,tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'files',lambda *a,**k:{})
    scope=driver.validate_spec(driver.MODE,{'plan':str(tmp_path)},240,210)
    assert scope['packing_child_wall_seconds']==60 and scope['fresh_transition_cap']==0
    for mode,wall,app in [('grouped-pilot',240,210),(driver.MODE,180,120),(driver.MODE,240,180)]:
        with pytest.raises(ValueError):driver.validate_spec(mode,{'plan':str(tmp_path)},wall,app)
    r=record(tmp_path,driver);cmd=driver.native_command(r,60)
    assert cmd[cmd.index('--wall-seconds')+1]=='60.0' and 'human_punching.phase2_right_hook_pack_v1' in cmd
    for remaining in (59.999,181,float('nan'),True):
        with pytest.raises(RuntimeError):driver.native_command(r,remaining)


def test_budget_reserve_never_reduced(driver,monkeypatch):
    seen=[];monkeypatch.setattr(driver,'_admit_reservation',lambda *a,**k:seen.append((a,k)))
    driver.protected_reservation({},240,evaluation_reserve_seconds=0)
    assert seen[0][1]['evaluation_reserve_seconds']==4500
    with pytest.raises(ValueError):driver.protected_reservation({},180)
    with pytest.raises(ValueError):driver.protected_reservation({},240,successful_route_frozen=True)


@pytest.fixture
def packet(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json';plan.write_text('{}');proof=tmp_path/'proof.json';proof.write_text('{}')
    names=['src/human_punching/phase2_right_hook_pack_v1.py','src/human_punching/phase2_runtime_pack.py',
           'scripts/task10b_phase2_right_hook_pack_v1.py',str(plan.relative_to(ROOT)),str(proof.relative_to(ROOT))]
    files={n:sha256_file(ROOT/n) for n in names}
    for name in names:
        target=tmp_path/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    p=tmp_path/'packet.json';data=dict(schema='task10b_phase2_right_hook_pack_packet_v1',mode=inputs.MODE,
        training=False,allocation_wall_seconds=240,application_wall_seconds=210,packing_child_wall_seconds=60,
        evaluation_reserve_seconds=4500,fresh_transition_cap=0,gpus=1,inputs={'plan':str(plan)},
        output=str((tmp_path/'run').relative_to(ROOT)),runtime_namespace_proof=dict(path=str(proof.relative_to(ROOT)),sha256=sha256_file(proof)),
        runtime_namespace_evidence={'exact':'proof'},files=files,
        automatic_followup=False,promotion=False,g1_actor_execution=False,combined_scene=False)
    save(p,data)
    monkeypatch.setattr(entry,'logical_output_adapter',lambda *a: (lambda x:Path(x),{'exact':'proof'}))
    def execute():return entry.execute(plan,tmp_path/'run',wall_seconds=60,logical_project_root=ROOT,
        runtime_namespace_proof=proof,packet_path=p,packet_sha256=sha256_file(p))
    return p,data,execute


def test_entry_restores_original_packer_dependencies_after_exception(packet,monkeypatch):
    _,_,execute=packet;original=(entry.packing.writable_path,entry.packing.owned_input)
    def fail(*a,**k):raise RuntimeError('preserved simulated child failure')
    monkeypatch.setattr(entry.packing,'pack',fail)
    with pytest.raises(RuntimeError):execute()
    assert (entry.packing.writable_path,entry.packing.owned_input)==original


@pytest.mark.parametrize('change',['schema','source','archive','wall','reserve'])
def test_entry_requires_exact_packet_source_and_caps(packet,monkeypatch,change):
    p,data,execute=packet
    if change=='schema':data['schema']='task10b_phase2_sequential_pipeline_packet_v1'
    elif change=='wall':data['packing_child_wall_seconds']=61
    elif change=='reserve':data['evaluation_reserve_seconds']=0
    elif change=='source':data['files']['src/human_punching/phase2_right_hook_pack_v1.py']='a'*64
    elif change=='archive':(p.parent/'source_snapshot'/next(iter(data['files']))).write_text('changed')
    save(p,data);monkeypatch.setattr(entry.packing,'pack',lambda *a,**k:pytest.fail('must not pack'))
    with pytest.raises(ValueError):execute()


def test_unchanged_packer_precision_contract():
    n=4;source=dict(t=np.arange(n)*.005,qpos=np.ones((n,55))*.123456789,
        qvel=np.ones((n,54))*.234567891,torque=np.ones((n,48))*.345678912,
        qacc=np.ones((n,54)),head_center=np.array([1.,2.,3.]),active_hand_index=np.array(1),guard_hold_start=np.array(.01))
    packed,receipt=entry.packing.runtime_arrays(source)
    for key in ('qpos','qvel','torque'):assert np.array_equal(packed[key],source[key].astype(np.float32))
    assert packed['t'].dtype==np.float64 and np.array_equal(packed['t'],source['t'])
    assert 'qacc' not in packed and receipt['physical_qualification'] is False
