"""New source-bounded initial candidates; no supplied state is corrected."""
from pathlib import Path
import ast
import importlib.util
import json
from dataclasses import asdict

import numpy as np
import pytest

from human_punching import phase2_shared_catalog_v2 as geometry
from human_punching import phase2_native_bank_geometry_v2 as capture
from human_punching.wbc.provenance import PROJECT_ROOT


def inputs():
    with np.load(PROJECT_ROOT/'outputs/four-punch-milestone/jab/reference.npz',allow_pickle=False) as f:q=f['qpos'][0].copy()
    metadata=json.loads((PROJECT_ROOT/'outputs/four-punch-milestone/jab/body_metadata.json').read_text())
    domain=json.loads((PROJECT_ROOT/'outputs/task10b/p2domain01/domain.json').read_text())
    return q,metadata,domain


@pytest.mark.parametrize('split',['train','validation'])
def test_actual_candidate_bank_is_inward_exact_and_source_bounded(split):
    nominal,metadata,domain=inputs();q,v,receipt=geometry.construct_shared_starts(nominal,metadata,domain,split)
    addresses=np.asarray(metadata['qpos_addresses']);limits=np.asarray(metadata['joint_limits_rad'])
    assert q.dtype==v.dtype==np.float32 and q.shape==(4,55) and v.shape==(4,54)
    assert np.all(q[:,addresses]>=limits[:,0]) and np.all(q[:,addresses]<=limits[:,1])
    assert np.max(np.abs(q[:,addresses].astype(float)-nominal[addresses]))<=.015
    assert np.max(np.abs(v.astype(float)))<=.06 and not np.any(v[:,:6])
    assert np.array_equal(q[:,:7],np.tile(nominal[:7].astype(np.float32),(4,1)))
    assert [r['qpos_address'] for r in receipt['native_nominal_outward_rounding']]==[19,47,54]
    assert [r['joint_name'] for r in receipt['native_nominal_outward_rounding']]==[metadata['joint_names'][i] for i in (12,40,47)]
    assert all(abs(r['delta_from_native_rad'])==float(np.spacing(np.float32(abs(r['native_value_rad'])))) for r in receipt['native_nominal_outward_rounding'])
    assert len(geometry.validate_start_bank(q,v,nominal,metadata,domain,split))==4


def test_original_reference_is_not_mutated_and_real_limit_errors_are_rejected():
    nominal,metadata,domain=inputs();before=nominal.copy();geometry.construct_shared_starts(nominal,metadata,domain,'train')
    assert np.array_equal(nominal,before)
    nominal[19]+=.001
    with pytest.raises(ValueError,match='Original accepted source'):
        geometry.construct_shared_starts(nominal,metadata,domain,'train')


def test_tampered_or_shifted_start_is_not_silently_reconstructed():
    nominal,metadata,domain=inputs();q,v,_=geometry.construct_shared_starts(nominal,metadata,domain,'train')
    q[0,0]+=.01
    with pytest.raises(ValueError,match='differs'):
        geometry.validate_start_bank(q,v,nominal,metadata,domain,'train')


def test_resolve_actual_accepted_jab_preserves_source_nominal_before_quantization(tmp_path):
    """Real source/reference/domain metadata through the complete resolve path."""
    from human_punching.phase2_shared_reference import SharedPreparationBounds,RECIPE_VERSION
    from human_punching.phase2_target_reference import RetargetSettings
    from human_punching.wbc.provenance import sha256_file
    nominal,metadata,domain=inputs();q,v,_=geometry.construct_shared_starts(nominal,metadata,domain,'train')
    starts=tmp_path/'actual_jab_starts.npz';np.savez(starts,qpos=q,qvel=v)
    domain_path=PROJECT_ROOT/'outputs/task10b/p2domain01/domain.json'
    target_path=PROJECT_ROOT/'outputs/task10b/p2targets01/target_geometry.json'
    parent_path=PROJECT_ROOT/'outputs/task10b/p2nr01/run/manifest.json'
    parent=json.loads(parent_path.read_text());targets=json.loads(target_path.read_text())
    group=next(row['snapshot']['command_id'] for row in domain['selected_command_groups'] if row['split']=='train')
    target=next(row for row in targets['geometry_candidates'] if row['command_id']==group and row['split']=='train')
    rows=[next(row for row in parent['cells'] if row['candidate_id']==candidate) for candidate in ('jab_uniform_0p600','jab_uniform_0p750')]
    def bound(path):return dict(path=str(path),sha256=sha256_file(path))
    def save(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data));return bound(path)
    settings=json.loads(json.dumps(asdict(RetargetSettings('Left',('LeftShoulder_z',)))))
    schedules={row['candidate_id']:dict(preparation_complete_tick=0,launch_tick=row['launch_tick'],
        outward_complete_tick=row['outward_end_tick'],guard_hold_tick=int(np.ceil(row['guard_time_s']/.005-1e-10)),
        reference_end_tick=round(row['duration_s']/.005),semantics_version='task10b_explicit_connector_phases_v1') for row in rows}
    # Other-split hashes are explicit synthetic fixture values. No held-out
    # geometry is constructed/read, and this plan is never submitted/generated.
    plan=dict(schema=geometry.SCHEMA,family='jab',split='train',start_mode='shared_jab_saved_state',
        coordinate_precision='native_float32_canonical',domain_scope='candidate_subset',maximum_cells=8,
        reference_byte_cap=2*1024**3,resident_byte_cap=256*1024**2,preparation_s=.4,
        preparation_bounds=asdict(SharedPreparationBounds()),retarget_settings=settings,shared_target_settings={},
        command_groups=[group],timings_s=[.6,.75],domain=bound(domain_path),target_geometry=bound(target_path),
        parent_catalog=bound(parent_path),start_bank=bound(starts),release=str(PROJECT_ROOT/'outputs/four-punch-milestone'),
        start_bank_sha256_by_split=dict(train=sha256_file(starts),validation='1'*64,test='2'*64),source_schedules=schedules,
        cells=[dict(candidate_id=f'actual_source_resolve_{j}_{i}',parent_candidate_id=row['candidate_id'],
            target_candidate_identity=target['candidate_identity'],start_index=i) for j,row in enumerate(rows) for i in range(4)])
    recipe=dict(schema=geometry.RECIPE_SCHEMA,family='jab',shared_recipe_version=RECIPE_VERSION,
        start_quantization_version=geometry.START_QUANTIZATION,physics_dt_s=.005,actor_dt_s=.020,
        **{key:plan[key] for key in ('start_mode','coordinate_precision','preparation_s','preparation_bounds',
            'retarget_settings','shared_target_settings','source_schedules','start_bank_sha256_by_split')},
        **{key+'_sha256':plan[key]['sha256'] for key in ('domain','target_geometry','parent_catalog')},
        accepted_jab_reference_sha256=sha256_file(PROJECT_ROOT/'outputs/four-punch-milestone/jab/reference.npz'),
        accepted_jab_body_metadata_sha256=sha256_file(PROJECT_ROOT/'outputs/four-punch-milestone/jab/body_metadata.json'))
    plan['recipe']=save('actual_source_recipe.json',recipe)
    resolved,_,context=geometry.resolve_cells(plan)
    assert len(resolved)==8 and np.array_equal(context['nominal'],nominal)
    assert context['nominal'].dtype==nominal.dtype
    assert np.array_equal(context['qpos'].astype(np.float32),q)


def test_capture_body_preserves_original_calls_and_changes_only_geometry_and_receipt_version():
    old=PROJECT_ROOT/'src/human_punching/phase2_native_state.py';new=Path(capture.__file__)
    def function(source):
        tree=ast.parse(source)
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_shared_start_bank')
    original=ast.unparse(function(old.read_text())).replace('phase2_shared_catalog import','phase2_shared_catalog_v2 import').replace(
        'task10b_phase2_native_shared_bank_capture_v1','task10b_phase2_native_shared_geometry_bank_capture_v2')
    assert ast.dump(ast.parse(original))==ast.dump(ast.parse(ast.unparse(function(new.read_text()))))


def test_capture_driver_has_zero_steps_finite_allocation_and_fixed_child_module():
    path=PROJECT_ROOT/'outputs/task10b/p2initialcaptureprep01/driver.py'
    spec=importlib.util.spec_from_file_location('initial_capture_transport_v2',path);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    result=driver.native_command(dict(mode='initial-state-capture-v2',output='outputs/task10b/p2sc01/run'),269.)
    assert result[2:4]==['-m','human_punching.phase2_initial_capture_v2']
    assert result[4:6]==['--packet',str(PROJECT_ROOT/'outputs/task10b/p2sc01/packet.json')]
    for value in [120,270.001,float('nan'),float('inf')]:
        with pytest.raises(ValueError):driver.native_command(dict(mode='initial-state-capture-v2'),value)
    source=path.read_text()
    assert 'planned_explicit_integration_world_steps=0,captured_worlds=4,capture_split="train"' in source
    assert 'allocation_wall_seconds=360' in source and 'application_wall_seconds=300.' in source
    assert 'num_envs=4,rollout_steps=0' in source
