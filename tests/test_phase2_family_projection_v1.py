"""Saved-only alias views preserve failures and exact family source identity."""
import copy,importlib.util,json,shutil
from pathlib import Path
import pytest
from human_punching import phase2_family_projection_v1 as projection


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,sort_keys=True))
    return projection.sha256_file(path)


@pytest.fixture
def data(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('family_projection_fixture_source',Path(__file__).with_name('test_phase2_sequential_projection_v1.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    d=old.fixture_files(tmp_path,monkeypatch);monkeypatch.setattr(projection,'ROOT',d.root)
    d.plan.update(schema='task10b_phase2_family_target_probe_plan_v1',entry_version='task10b_phase2_family_target_probe_v1',
        family='cross',maximum_child_seconds=210,physical_qualification=False,new_family_engineering_bound_unqualified=True,
        native_control_required_before_learning=True)
    d.plan['retarget_settings'].update(active_side='Right',trunk_joint_names=[],optimized_joint_names=['Right'+s for s in
        ('Shoulder_x','Shoulder_y','Shoulder_z','Arm_x','Arm_y','Arm_z','ForeArm_z','ForeArm_x','Hand_x','Hand_y','Hand_z')])
    extra=['timing_selection','selected_d0_plan']
    for key in extra:
        path=d.root/(key+'.json');digest=write(path,{'opaque_ancestry_path':'/foreign/read-only'})
        d.plan[key]=dict(path=str(path),sha256=digest);d.catalog['input_hashes'][key]=digest
    names=[projection.GENERATOR,projection.SOLVER,projection.WRAPPER,'src/human_punching/phase2_target_reference.py','src/human_punching/phase2_target_catalog.py']
    for name in names+[projection.CLI]:
        p=d.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic exact owned source\n')
    d.plan['sources']={name:dict(path=str(d.root/name),sha256=projection.sha256_file(d.root/name)) for name in names}
    for row in d.catalog['cells']:row['family']='cross'
    write(d.plan_path,d.plan)
    for name in names+[projection.CLI,'plan.json']+[k+'.json' for k in extra]:
        d.packet['files'][name]=projection.sha256_file(d.root/name)
        destination=d.archive/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(d.root/name,destination)
    d.packet.update(schema=projection.PROBE_SCHEMA,mode='family-target-references',family='cross',sequential_child_wall_seconds=210)
    write(d.packet_path,d.packet);digest=projection.sha256_file(d.packet_path)
    d.execution['packet_sha256']=d.terminal['packet_sha256']=digest
    d.execution['native_command'][3]='human_punching.phase2_family_target_probe_v1';d.execution['native_command'][-1]='210.0'
    write(d.execution_path,d.execution);write(d.terminal_path,d.terminal)
    d.catalog['source_sha256']=d.packet['files'][projection.GENERATOR];d.catalog['plan_sha256']=projection.sha256_file(d.plan_path)
    write(d.catalog_path,d.catalog)
    d.sidecar=dict(schema='task10b_phase2_family_target_probe_v1',plan_sha256=projection.sha256_file(d.plan_path),
        manifest_sha256=projection.sha256_file(d.catalog_path),wrapper_source_sha256=d.packet['files'][projection.WRAPPER],
        frozen_generator_source_sha256=d.packet['files'][projection.GENERATOR],family='cross',unchanged_solver_version=projection.SOLVER_VERSION,
        engineering_cap_rad=.205,new_family_bound_unqualified=True,solver_changed=False,physical_limits_changed=False,
        physical_qualification=False,native_control_required_before_learning=True)
    d.sidecar_path=d.run/'family_probe_entry.json';write(d.sidecar_path,d.sidecar)
    return d


def build(data):return projection.build_projection(data.catalog_path,data.execution_path,data.probe_path)


def test_exact_family_projection_roundtrip_and_complete_closure(data):
    before={p:projection.sha256_file(p) for p in data.root.rglob('*') if p.is_file()}
    result=projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/'view')
    path=data.root/result['relative_name'];view,receipt=projection.verify_projection(path)
    restored=copy.deepcopy(view)
    for name in projection.ANNOTATIONS:restored.pop(name)
    for change in receipt['changes']:
        _,index,key=change['json_location'].split('/');restored['cells'][int(index)][key]=change['old']
    assert restored==data.catalog
    closure=projection.files(path)
    assert str(data.sidecar_path.relative_to(data.root)) in closure
    assert all(name in closure for name in data.plan['sources'])
    assert all(projection.sha256_file(p)==digest for p,digest in before.items())
    assert view['admitted_candidate_cells']==3 and view['physical_qualification'] is False


@pytest.mark.parametrize('field,value',[('family','jab'),('engineering_cap_rad',.21),('new_family_bound_unqualified',False),
    ('physical_limits_changed',True),('manifest_sha256','0'*64),('wrapper_source_sha256','0'*64),('solver_changed',True)])
def test_sidecar_cannot_relabel_family_or_engineering_claim(data,field,value):
    data.sidecar[field]=value;write(data.sidecar_path,data.sidecar)
    with pytest.raises(ValueError,match='sidecar'):build(data)


@pytest.mark.parametrize('where',[projection.GENERATOR,projection.WRAPPER,'plan.json','timing_selection.json','selected_d0_plan.json'])
def test_live_or_archived_ancestry_change_rejected(data,where):
    (data.root/where).write_text('changed')
    with pytest.raises(ValueError):build(data)


def test_nonalias_identical_bytes_rejected(data):
    data.compute.unlink();shutil.copytree(data.root,data.compute)
    with pytest.raises(ValueError,match='same owned inode'):build(data)


@pytest.mark.parametrize('field,value',[('requested_ttc_s',.9),('family','jab'),('kinematic_admission',True),('target_w',[0,0,0])])
def test_no_projected_science_mutation(data,field,value):
    projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/'view')
    path=data.root/'view/manifest.json';view=json.loads(path.read_text());view['cells'][3][field]=value;write(path,view)
    with pytest.raises(ValueError,match='changed science'):projection.verify_projection(path)


def test_real_completed_cross_two_failures_preserved_without_array_load():
    root=Path(__file__).resolve().parents[1]
    view,receipt=projection.build_projection(root/'outputs/task10b/p2ft01c/run/manifest.json',root/'outputs/task10b/p2ft01c/execution.json',root/'outputs/task10b/p2hostalias01/probe.json')
    assert view['planned_cells']==view['executed_cells']==4 and view['generated_cells']==2
    assert [r['generation_disposition'] for r in view['cells']]==['failed','failed','generated','generated']
    assert all(r['physical_qualification'] is False for r in view['cells'])
    assert receipt['reference_bytes_changed'] is False and receipt['scientific_metadata_changed'] is False


def test_real_all_negative_left_hook_remains_four_explicit_failures():
    root=Path(__file__).resolve().parents[1]
    view,receipt=projection.build_projection(root/'outputs/task10b/p2ft01l/run/manifest.json',root/'outputs/task10b/p2ft01l/execution.json',root/'outputs/task10b/p2hostalias01/probe.json')
    assert view['planned_cells']==view['executed_cells']==4
    assert view['generated_cells']==view['admitted_candidate_cells']==0
    assert all(r['generation_disposition']=='failed' and r['kinematic_admission'] is False for r in view['cells'])
    assert receipt['protocol']['family']=='left_hook'


def test_incomplete_run_cannot_gain_projection_by_dropping_unattempted_rows(data):
    data.catalog.update(complete=False,executed_cells=3,unattempted_cells=[data.plan['cells'][-1]])
    write(data.catalog_path,data.catalog)
    with pytest.raises(ValueError,match='Complete unqualified'):build(data)
