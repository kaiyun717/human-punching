"""Four exact training target cells per new family, unchanged sequential v3 IK.

The .210rad search cap is a declared engineering hypothesis for these families,
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
from .phase2_family_target_path_v2 import SequentialSettings,VERSION as SOLVER_VERSION
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json

VERSION='task10b_phase2_family_target_probe_v2'
SCHEMA='task10b_phase2_family_target_probe_plan_v2'
FAMILIES=('cross','left_hook')
TIMINGS={'cross':(.6,.75),'left_hook':(1.1,1.25),'right_hook':(1.,1.2)}
GROUPS=(4184,4251)
PLACEMENT=dict(forward_m=0.,left_m=0.,yaw_rad=0.)
MAX_OUTPUT_BYTES=128*1024**2
CHILD_SECONDS=210
SOURCES=(*engine_source.SOURCES,'src/human_punching/phase2_family_target_path_v2.py','src/human_punching/phase2_family_target_probe_v2.py')


def binding(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError('Owned family probe input required')
    return dict(path=str(path),sha256=sha256_file(path))


def read(value):return json.loads(original.owned_input(value).read_text())
def source_bindings():return {name:binding(PROJECT_ROOT/name) for name in SOURCES}


def family_settings(family):
    if family not in FAMILIES:raise ValueError('Only cross/left-hook cap experiment; jab and right-hook unchanged')
    side='Left' if family=='left_hook' else 'Right'
    names=tuple(side+name for name in ('Shoulder_x','Shoulder_y','Shoulder_z','Arm_x','Arm_y','Arm_z',
        'ForeArm_z','ForeArm_x','Hand_x','Hand_y','Hand_z'))
    return SequentialSettings(RetargetSettings(side,names,maximum_iterations=80,maximum_joint_delta_rad=.210))


def selection_for(plan):
    selection=read(plan['timing_selection']);family=plan['family']
    record=selection['families'][family];probe=selection['target_probe']
    if (selection.get('schema')!='task10b_phase2_all_family_target_development_selection_v1'
            or record['trained_timings_s']!=list(TIMINGS[family]) or probe['command_groups']!=list(GROUPS)
            or probe['placement']!=PLACEMENT or record['source_catalog']!=plan['parent_catalog']
            or len(record['training_candidate_ids'])!=2 or plan['selected_d0_plan']!=record['d0_plan']
            or {**plan['retarget_settings'],'maximum_joint_delta_rad':.205}!=record['retarget_settings_proposal']):
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
    evidence_paths(plan)
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
        expected_id=f"{plan['family']}_train_c{target['command_id']}_{timing:.3f}".replace('.','p')+'_family_0210_v2_probe'
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
    paths.update(evidence_paths(plan))
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
    from .phase2_family_target_path_v2 import retarget_sequential
    engine.retarget_sequential=retarget_sequential
    engine.VERSION=SOLVER_VERSION
    result=engine.generate(plan_path,output,wall_seconds=wall_seconds)
    folder=writable_path(output)
    write_json(folder/'family_probe_entry.json',dict(schema=VERSION,plan_sha256=sha256_file(plan_path),
        manifest_sha256=sha256_file(folder/'manifest.json'),wrapper_source_sha256=sha256_file(__file__),
        frozen_generator_source_sha256=sha256_file(engine_source.__file__),family=plan['family'],
        solver_source_sha256=sha256_file(PROJECT_ROOT/'src/human_punching/phase2_family_target_path_v2.py'),
        solver_change='Engineering correction cap .205 to .210 only; distinct solver version; same original math and physical limits',
        unchanged_solver_version=SOLVER_VERSION,engineering_cap_rad=.210,new_family_bound_unqualified=True,
        solver_changed=True,solver_math_other_than_cap_changed=False,physical_limits_changed=False,physical_qualification=False,native_control_required_before_learning=True))
    return result


def diagnostic_input(item):
    path=Path(item['path'])
    if not path.is_absolute():path=PROJECT_ROOT/path
    return original.owned_input(dict(path=str(path),sha256=item['sha256']))


def evidence_paths(plan):
    """Close the six saved-pose diagnostic and every byte it actually read."""
    evidence=read(plan['bound_sensitivity']);previous=read(plan['previous_probe_plan'])
    if (evidence.get('schema')!='task10b_phase2_six_failed_family_pose_bound_sensitivity_v1'
            or evidence.get('status')!='completed_saved_pose_algebra_only'
            or evidence.get('proposed_caps_rad')!=[.205,.21,.225,.25] or evidence.get('tolerance_m')!=.0005
            or evidence.get('accepted_constants_unchanged') is not True or evidence.get('physical_joint_limits_relaxed') is not False
            or evidence.get('new_reference_generation') is not False or len(evidence.get('cases',[]))!=6
            or previous.get('schema')!='task10b_phase2_family_target_probe_plan_v1'
            or previous.get('family')!=plan['family'] or previous['retarget_settings']['maximum_joint_delta_rad']!=.205
            or previous['parent_catalog']!=plan['parent_catalog'] or previous['timing_selection']!=plan['timing_selection']
            or previous['selected_d0_plan']!=plan['selected_d0_plan']):
        raise ValueError('Exact completed six-pose sensitivity and retained.205 family plan required')
    if [(r['parent_candidate_id'],r['target_candidate_identity']) for r in previous['cells']]!=[(r['parent_candidate_id'],r['target_candidate_identity']) for r in plan['cells']]:
        raise ValueError('Cap-only experiment keeps the same four source/target requests')
    required={name:dict(path=name,sha256=digest) for name,digest in evidence['input_bindings'].items()}
    required.update({item['path']:item for item in (evidence['script'],evidence['independent_fk_source'],evidence['alias_proof'])})
    if plan.get('bound_sensitivity_inputs')!=required:raise ValueError('Exact diagnostic input byte inventory required')
    for item in required.values():diagnostic_input(item)
    prior_path=original.owned_input(plan['previous_probe_plan'])
    if evidence['input_bindings'].get(str(prior_path.relative_to(PROJECT_ROOT)))!=plan['previous_probe_plan']['sha256']:
        raise ValueError('The previous family plan must be an exact saved diagnostic input')
    expected=2 if plan['family']=='cross' else 4
    cases=[r for r in evidence['cases'] if r['family']==plan['family']]
    if len(cases)!=expected:raise ValueError('All failed family frames must remain present')
    for case in cases:
        selected=next(r for r in case['samples'] if r['engineering_cap_rad']==.21)
        if (selected['physical_joint_ranges_retained'] is not True or selected['nonlinear_single_pose_inside_05mm'] is not True
                or selected['nonlinear_single_pose_residual_m']>.0005 or case['jacobian_finite_difference_max_error']>=1e-9):
            raise ValueError('Selected.210 cap lacks the reviewed local-pose diagnostic')
    recipe=read(plan['recipe'])
    if (recipe.get('bound_sensitivity')!=plan['bound_sensitivity'] or recipe.get('previous_probe_plan')!=plan['previous_probe_plan']
            or recipe.get('selected_engineering_cap_rad')!=.21 or recipe.get('other_solver_math_changed') is not False
            or recipe.get('larger_caps_selected') is not False):
        raise ValueError('New cap selection must be explicit in the separately versioned recipe')
    return {name:original.owned_input(plan[name]) for name in ('bound_sensitivity','previous_probe_plan')}


def create_plans(sensitivity_path,output):
    """Two plans only; saved-pose evidence does not establish full-path success."""
    sensitivity=binding(sensitivity_path);evidence=read(sensitivity)
    inventory={name:dict(path=name,sha256=digest) for name,digest in evidence['input_bindings'].items()}
    inventory.update({item['path']:item for item in (evidence['script'],evidence['independent_fk_source'],evidence['alias_proof'])})
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False);results={}
    for family in FAMILIES:
        prior_path=PROJECT_ROOT/'outputs/task10b/p2familytargetprep01'/(family+'_plan.json');prior=read(binding(prior_path))
        plan=deepcopy(prior);plan.update(schema=SCHEMA,entry_version=VERSION,path_solver_version=SOLVER_VERSION,
            sources=source_bindings(),bound_sensitivity=sensitivity,previous_probe_plan=binding(prior_path),bound_sensitivity_inputs=inventory)
        plan['retarget_settings']['maximum_joint_delta_rad']=.210
        for row in plan['cells']:row['candidate_id']=row['candidate_id'].replace('_family_v3_probe','_family_0210_v2_probe')
        recipe=read(prior['recipe']);recipe.update(family_probe_entry_version=VERSION,path_solver_version=SOLVER_VERSION,
            retarget_settings=plan['retarget_settings'],solver_sources=plan['sources'],bound_sensitivity=sensitivity,
            previous_probe_plan=binding(prior_path),selected_engineering_cap_rad=.210,other_solver_math_changed=False,larger_caps_selected=False,
            purpose='One bounded.210 cap-only full-path construction probe following six saved-pose diagnostics; no local-pose-to-full-path or physical qualification inference.')
        recipe_path=folder/(family+'_recipe.json');write_json(recipe_path,recipe);plan['recipe']=binding(recipe_path)
        resolve_cells(plan);path=folder/(family+'_plan.json');write_json(path,plan)
        write_json(folder/(family+'_inputs.json'),dict(plan=str(path)));results[family]=binding(path)
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True)
    a=p.parse_args(argv);result=generate(a.plan,a.output,wall_seconds=a.wall_seconds)
    print(json.dumps({k:result[k] for k in ('complete','planned_cells','executed_cells','generated_cells','exception')}))
    return 0 if result['complete'] and result['exception'] is None else 1

if __name__=='__main__':raise SystemExit(main())
