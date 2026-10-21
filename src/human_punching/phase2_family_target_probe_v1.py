"""Four exact training target cells per new family, unchanged sequential v3 IK.

The .205rad search cap is a declared engineering hypothesis for these families,
not inherited jab feasibility. No trunk/root/leg changes, learned generator,
reference substitution, interpolation training, or physical qualification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util,json
from pathlib import Path
from . import phase2_target_catalog as original
from . import phase2_sequential_target_catalog_v3 as engine_source
from .phase2_target_reference import RetargetSettings
from .phase2_target_path_reference_v3 import SequentialSettings,VERSION as SOLVER_VERSION
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json

VERSION='task10b_phase2_family_target_probe_v1'
SCHEMA='task10b_phase2_family_target_probe_plan_v1'
FAMILIES=('cross','left_hook','right_hook')
TIMINGS={'cross':(.6,.75),'left_hook':(1.1,1.25),'right_hook':(1.,1.2)}
GROUPS=(4184,4251)
PLACEMENT=dict(forward_m=0.,left_m=0.,yaw_rad=0.)
MAX_OUTPUT_BYTES=128*1024**2
CHILD_SECONDS=210
SOURCES=(*engine_source.SOURCES,'src/human_punching/phase2_family_target_probe_v1.py')


def binding(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError('Owned family probe input required')
    return dict(path=str(path),sha256=sha256_file(path))


def read(value):return json.loads(original.owned_input(value).read_text())
def source_bindings():return {name:binding(PROJECT_ROOT/name) for name in SOURCES}


def family_settings(family):
    if family not in FAMILIES:raise ValueError('Only three newly declared families; jab probe already preserved')
    side='Left' if family=='left_hook' else 'Right'
    names=tuple(side+name for name in ('Shoulder_x','Shoulder_y','Shoulder_z','Arm_x','Arm_y','Arm_z',
        'ForeArm_z','ForeArm_x','Hand_x','Hand_y','Hand_z'))
    return SequentialSettings(RetargetSettings(side,names,maximum_iterations=80,maximum_joint_delta_rad=.205))


def selection_for(plan):
    selection=read(plan['timing_selection']);family=plan['family']
    record=selection['families'][family];probe=selection['target_probe']
    if (selection.get('schema')!='task10b_phase2_all_family_target_development_selection_v1'
            or record['trained_timings_s']!=list(TIMINGS[family]) or probe['command_groups']!=list(GROUPS)
            or probe['placement']!=PLACEMENT or record['source_catalog']!=plan['parent_catalog']
            or len(record['training_candidate_ids'])!=2 or plan['selected_d0_plan']!=record['d0_plan']
            or plan['retarget_settings']!=record['retarget_settings_proposal']):
        raise ValueError('Exact sealed timing selection and two training target groups required')
    d0=read(plan['selected_d0_plan'])
    if (d0.get('schema')!='task10b_phase2_selected_d0_timing_pilot_plan_v1'
            or d0.get('family')!=family or d0.get('training_candidate_ids')!=record['training_candidate_ids']):
        raise ValueError('Explicit prior selected D0 recipe must bind the exact trained rows')
    return selection,record


def validate_plan(plan):
    family=plan.get('family')
    if (family not in FAMILIES or plan.get('schema')!=SCHEMA or plan.get('entry_version')!=VERSION
            or plan.get('split')!='train' or plan.get('maximum_cells')!=4 or len(plan.get('cells',[]))!=4
            or plan.get('maximum_child_seconds')!=CHILD_SECONDS or plan.get('output_byte_cap')!=MAX_OUTPUT_BYTES
            or plan.get('path_solver_version')!=SOLVER_VERSION or set(plan.get('sources',{}))!=set(SOURCES)
            or plan.get('physical_qualification') is not False
            or plan.get('new_family_engineering_bound_unqualified') is not True
            or plan.get('native_control_required_before_learning') is not True):
        raise ValueError('Exact four-cell new-family engineering probe contract required')
    settings=family_settings(family)
    if asdict(RetargetSettings(**plan['retarget_settings']))!=asdict(settings.base):
        raise ValueError('Unchanged commonv3 settings and explicit active-side11-joint mask required')
    for name,item in plan['sources'].items():
        if original.owned_input(item)!=(PROJECT_ROOT/name).resolve(strict=True):
            raise ValueError('Solver source must be the exact owned module')
    expected_body=(PROJECT_ROOT/'outputs/four-punch-milestone'/family/'body_metadata.json').resolve(strict=True)
    if original.owned_input(plan['body_metadata'])!=expected_body:
        raise ValueError('Exact accepted family body metadata required')
    selection,record=selection_for(plan)
    for name in ('domain','target_geometry'):
        if plan[name]!=selection[name]:raise ValueError('Declared canonical geometry changed')
    common=deepcopy(plan);common['schema']=original.SCHEMA
    original.validate_plan(common)
    return settings,common,record


def resolve_cells(plan):
    settings,common,record=validate_plan(plan)
    _,cells,paths=original.resolve_cells(common)
    actual=[(target['command_id'],parent['requested_ttc_s']) for _,target,parent,_,_ in cells]
    expected=[(group,timing) for group in GROUPS for timing in TIMINGS[plan['family']]]
    if actual!=expected:raise ValueError('Four exact training group/timing cells in declared order required')
    for entry,target,parent,_,_ in cells:
        timing=parent['requested_ttc_s'];index=TIMINGS[plan['family']].index(timing)
        expected_id=f"{plan['family']}_train_c{target['command_id']}_{timing:.3f}".replace('.','p')+'_family_v3_probe'
        if (parent['candidate_id']!=record['training_candidate_ids'][index] or entry['candidate_id']!=expected_id
                or parent!=record['selected_source_rows'][index] or target['placement']!=PLACEMENT
                or parent['active_hand_index']!=(0 if plan['family']=='left_hook' else 1)):
            raise ValueError('Exact trained parent, active hand, nominal placement and new candidate identity required')
    recipe=read(plan['recipe'])
    if (recipe.get('family_probe_entry_version')!=VERSION or recipe.get('path_solver_version')!=SOLVER_VERSION
            or recipe.get('solver_sources')!=plan['sources'] or recipe.get('timing_selection')!=plan['timing_selection']
            or recipe.get('selected_d0_plan')!=plan['selected_d0_plan']
            or recipe.get('comparison_parent_flags_preserved') is not True
            or recipe.get('new_family_engineering_bound_unqualified') is not True
            or recipe.get('native_control_required_before_learning') is not True
            or recipe.get('interior_training_eligible') is not False or recipe.get('sequential_preference_weight')!=.002):
        raise ValueError('Recipe must explicitly bind unqualified new-family solver/selection/ancestry')
    paths['timing_selection']=original.owned_input(plan['timing_selection'])
    paths['selected_d0_plan']=original.owned_input(plan['selected_d0_plan'])
    return settings,cells,paths


def _engine():
    spec=importlib.util.spec_from_file_location('human_punching._family_target_probe_engine_v1',Path(engine_source.__file__))
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    return engine


def generate(plan_path,output,*,wall_seconds):
    plan_path=original.owned_input(binding(plan_path));plan=json.loads(plan_path.read_text())
    resolve_cells(plan)
    if type(wall_seconds) not in (int,float) or wall_seconds!=CHILD_SECONDS:raise ValueError('Exact210second child required')
    engine=_engine()
    engine.resolve_cells=resolve_cells
    result=engine.generate(plan_path,output,wall_seconds=wall_seconds)
    folder=writable_path(output)
    write_json(folder/'family_probe_entry.json',dict(schema=VERSION,plan_sha256=sha256_file(plan_path),
        manifest_sha256=sha256_file(folder/'manifest.json'),wrapper_source_sha256=sha256_file(__file__),
        frozen_generator_source_sha256=sha256_file(engine_source.__file__),family=plan['family'],
        unchanged_solver_version=SOLVER_VERSION,engineering_cap_rad=.205,new_family_bound_unqualified=True,
        solver_changed=False,physical_limits_changed=False,physical_qualification=False,native_control_required_before_learning=True))
    return result


def create_plans(selection_path,output):
    """Metadata only; all three finite packets can be prepared without submission."""
    selection_binding=binding(selection_path);selection=read(selection_binding)
    geometry=read(selection['target_geometry']);sources=source_bindings()
    targets={group:next(t for t in geometry['geometry_candidates'] if t['command_id']==group and t['split']=='train' and t['placement']==PLACEMENT) for group in GROUPS}
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False);result={}
    for family in FAMILIES:
        record=selection['families'][family];settings=family_settings(family);retarget=asdict(settings.base)
        # JSON recipe lists must match immutable plan lists exactly.
        retarget=json.loads(json.dumps(retarget))
        recipe=dict(schema='task10b_phase2_target_recipe_v1',family=family,start_mode='family_guard',
            physics_dt_s=.005,actor_dt_s=.020,retarget_settings=retarget,
            domain_sha256=selection['domain']['sha256'],target_geometry_sha256=selection['target_geometry']['sha256'],
            parent_catalog_sha256=record['source_catalog']['sha256'],family_probe_entry_version=VERSION,
            path_solver_version=SOLVER_VERSION,solver_sources=sources,timing_selection=selection_binding,
            selected_d0_plan=record['d0_plan'],comparison_parent_flags_preserved=True,
            sequential_preference_weight=.002,interior_training_eligible=False,
            new_family_engineering_bound_unqualified=True,native_control_required_before_learning=True,
            optimized_trunk_joints=[],nominal_registered_placement=PLACEMENT,
            purpose='Four training-only target construction cells; common.205 engineering probe, no inherited jab/native or target competence.',
            source_contract='Exact family reference/root/feet/opposite arm/source holds retained; newly rederived5ms tangent derivatives, original feedforward and event/contact clocks; no G1 actor.',
            continuity_semantics='Sequential branch preference only; .025rad solver increment is not a between-knot bound. Preserve all correction-step/effort/support/base-equation diagnostics and failures.')
        recipe_path=folder/(family+'_recipe.json');write_json(recipe_path,recipe)
        cells=[]
        for group in GROUPS:
            for timing,parent_id in zip(TIMINGS[family],record['training_candidate_ids'],strict=True):
                name=f'{family}_train_c{group}_{timing:.3f}'.replace('.','p')+'_family_v3_probe'
                cells.append(dict(candidate_id=name,parent_candidate_id=parent_id,target_candidate_identity=targets[group]['candidate_identity']))
        plan=dict(schema=SCHEMA,entry_version=VERSION,family=family,split='train',start_mode='family_guard',cells=cells,
            maximum_cells=4,maximum_child_seconds=CHILD_SECONDS,output_byte_cap=MAX_OUTPUT_BYTES,
            reference_byte_cap=2*1024**3,resident_byte_cap=256*1024**2,
            retarget_settings=retarget,path_solver_version=SOLVER_VERSION,sources=sources,
            domain=selection['domain'],target_geometry=selection['target_geometry'],parent_catalog=record['source_catalog'],
            body_metadata=binding(PROJECT_ROOT/'outputs/four-punch-milestone'/family/'body_metadata.json'),
            timing_selection=selection_binding,selected_d0_plan=record['d0_plan'],recipe=binding(recipe_path),physical_qualification=False,
            new_family_engineering_bound_unqualified=True,native_control_required_before_learning=True)
        resolve_cells(plan);path=folder/(family+'_plan.json');write_json(path,plan)
        write_json(folder/(family+'_inputs.json'),dict(plan=str(path)));result[family]=binding(path)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True)
    a=p.parse_args(argv);result=generate(a.plan,a.output,wall_seconds=a.wall_seconds)
    print(json.dumps({k:result[k] for k in ('complete','planned_cells','executed_cells','generated_cells','exception')}))
    return 0 if result['complete'] and result['exception'] is None else 1

if __name__=='__main__':raise SystemExit(main())
