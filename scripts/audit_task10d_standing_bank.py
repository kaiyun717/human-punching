"""Independent post-run checks for the new one-fixture Phase 3C bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from human_punching.task10d.g1 import PROJECT
from human_punching.task10d.placement_phase3 import PRIMITIVE_IDS, cache_library
from human_punching.task10d.replay import Placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.bank_directory.resolve(strict=True), args.output.resolve()
    if not output.is_relative_to(PROJECT.resolve()) or output.exists():
        raise ValueError('Use a new audit output inside human-punching')
    with np.load(directory/'bank.npz', allow_pickle=False) as data:
        bank = {key: data[key] for key in data.files}
    summary = json.loads((directory/'bank_summary.json').read_text())
    landmarks = json.loads((directory/'fixed_landmarks.json').read_text())
    invocation = json.loads((directory/'invocation.json').read_text())
    # Slurm and the interactive host expose the same project with different
    # absolute mount prefixes; fixture identity below verifies the local join.
    fixture = json.loads((directory.parent/'fixture'/'fixture.json').read_text())
    checks = {}

    def equal(name, left, right):
        np.testing.assert_array_equal(left, right, err_msg=name)
        checks[name] = True

    equal('all_eight_ids', bank['primitive_ids'], PRIMITIVE_IDS)
    equal('fixture_identity',str(bank['fixture_id']),fixture['fixture_id'])
    equal('native_measured_camera_rotation_used',bank['camera_rotation'],fixture['geometry']['camera_rotation'])
    equal('grid_shape', bank['outward_hit'].shape, (8,506,72))
    equal('yaw_full_circle', bank['yaw_deg'], np.arange(-180,180,5))
    expected_xy = np.array([(x/100,y/100) for x in range(45,151,5)
                            for y in range(-55,56,5)])
    equal('xy_candidate_rectangle', bank['xy_G'], expected_xy)
    x,y = bank['xy_G'].T
    radius = np.hypot(x,y)
    primary = ((radius >= .7-1e-12) & (radius <= 1.5+1e-12) & (x >= .65-1e-12)
               & (np.abs(y) <= .55+1e-12) & (np.abs(np.arctan2(y,x)) <= np.deg2rad(35)+1e-12))
    equal('primary_envelope',bank['primary_xy'],primary)
    equal('initial_clearance',bank['initial_clearance'],(bank['initial_glove_distances_m'] >= .16).all(-1))
    equal('outward_hit_threshold',bank['outward_hit'],bank['outward_min_distance_m'] <= .14)
    equal('all_eligible_native_checked',bank['native_checked'],bank['geometric_initial_valid'])
    equal('native_threshold',bank['native_valid'],bank['native_checked'] &
          (bank['native_min_convex_distance_capped_m'] >= -.03))
    equal('all_native_valid_clean_rendered',bank['clean_checked'],bank['geometric_initial_valid'] & bank['native_valid'])
    clean = ((bank['glove_rgb_valid_source_pixels'] > 0).all(-1)
             & (bank['glove_acquired_min_cells'] > 0).all(-1) & bank['clean_checked'])
    equal('clean_processed_hand_admission',bank['clean_valid'],clean)
    scene = bank['initial_clearance'] & bank['initial_frustum'] & bank['native_valid']
    equal('scene_valid_distinct_from_clean',bank['initial_scene_valid'],scene)
    equal('raw_valid',bank['raw_valid'],scene & clean)
    equal('raw_aimed',bank['raw_aimed'],bank['raw_valid'] & bank['outward_hit'])
    # Independent graph traversal on the circle; no production retention helper.
    expected = np.zeros_like(bank['raw_valid'])
    component_count = np.zeros((8,506),int)
    component_start = np.full((8,506),-1,int)
    component_length = np.zeros((8,506),int)
    for p in range(8):
        for xy in range(506):
            mask = bank['raw_valid'][p,xy]
            unseen = set(np.flatnonzero(mask).tolist())
            components=[]
            while unseen:
                seed=min(unseen)
                queue=[seed]
                connected=set()
                while queue:
                    at=queue.pop()
                    if at not in unseen:
                        continue
                    unseen.remove(at)
                    connected.add(at)
                    queue.extend([(at-1)%72,(at+1)%72])
                start=0 if len(connected)==72 else next(i for i in connected if (i-1)%72 not in connected)
                components.append((len(connected),start,connected))
            component_count[p,xy]=len(components)
            if components:
                length,start,indices=min(components,key=lambda row:(-row[0],row[1]))
                expected[p,xy,list(indices)]=True
                component_start[p,xy],component_length[p,xy]=start,length
    equal('largest_valid_circular_component',bank['retained_valid'],expected)
    equal('component_count',bank['valid_component_count'],component_count)
    equal('component_start_tie_rule',bank['retained_component_start'],component_start)
    equal('component_length',bank['retained_component_length'],component_length)
    equal('retained_aimed_intersection_after_retention',bank['retained_aimed'],expected & bank['outward_hit'])
    equal('aimed_losses',bank['aimed_lost_by_retention'],bank['raw_aimed'] & ~bank['retained_aimed'])
    equal('sampler_support',bank['primary_sampler_xy'],primary & bank['retained_aimed'].any(axis=(0,2)))
    for domain,selected in [('primary',primary),('exploratory',~primary)]:
        expected_counts={key:int(bank[key][:,selected].sum()) for key in
                         ('raw_valid','raw_aimed','retained_valid','retained_aimed','aimed_lost_by_retention')}
        for key,value in expected_counts.items():
            equal(f'{domain}_{key}_reported_count',summary['domains'][domain][key],value)
    # Check each fixed annotation with unchanged production FK at its saved
    # bracket and root, and verify it is an actual admitted retained bank entry.
    paths = cache_library()
    calibrated=[]
    for p,(path,row) in enumerate(zip(paths,landmarks['landmarks'],strict=True)):
        equal(f'landmark_{p}_identity',row['primitive_id'],PRIMITIVE_IDS[p])
        equal(f'landmark_{p}_reference_hash',row['reference_sha256'],path.primitive.reference.sha256)
        if not row['supported']:
            equal(f'landmark_{p}_unsupported',bank['retained_aimed'][p,primary].any(),False)
            continue
        choice=row['calibration_choice']
        xy,yaw=choice['xy_index'],choice['yaw_index']
        equal(f'landmark_{p}_admitted',primary[xy] and bank['retained_aimed'][p,xy,yaw],True)
        placement=Placement(**row['native_placement'])
        entry=row['s_j_star']
        previous,first=row['crossing_bracket_s']
        assert 0 <= previous < entry <= first <= row['accepted_outward_end_s']+1e-12
        distance=lambda t:float(np.linalg.norm(path.primitive.reference.sample(t,placement).glove_pos[path.active_index]-bank['head_world']))
        assert distance(previous) > .14 and distance(first) <= .14
        assert abs(distance(entry)-.14) < 2e-10
        equal(f'landmark_{p}_first_sample',first,bank['first_outward_sample_s'][p,xy,yaw])
        head=np.asarray(row['frozen_head_world'])
        equal(f'landmark_{p}_frozen_head',head,bank['head_world'])
        calibrated.append(dict(primitive_id=PRIMITIVE_IDS[p],s_j_star=entry,
                               first_sample_s=first,bracket_s=[previous,first]))
    report=dict(schema='task10d_standing_bank_independent_audit_v1',passed=True,
        checks=checks,calibrated=calibrated,
        candidate_count=int(bank['outward_hit'].size),
        native_checked=int(bank['native_checked'].sum()),clean_checked=int(bank['clean_checked'].sum()),
        primary_sampler_xy=int(bank['primary_sampler_xy'].sum()),
        inspected_artifact_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ['bank.npz','bank_summary.json','fixed_landmarks.json']})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['passed','candidate_count','native_checked','clean_checked','primary_sampler_xy']},indent=2))


if __name__=='__main__':
    main()
