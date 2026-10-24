"""Assemble the small Phase 2 review index without copying motion arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from human_punching.task10c.robot import ROOT, JOINT_NAMES
from human_punching.task10d.phase2 import load_primitive


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--variant-run', type=Path, required=True)
    parser.add_argument('--visual-review', type=Path,
                        default=ROOT/'outputs/task10d/phase2/visual_review.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/task10d/phase2/library_index.json')
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to((ROOT/'outputs/task10d/phase2').resolve())
    if out.exists():
        raise FileExistsError('Preserve prior index; choose a new filename')
    nominal_path = ROOT/'outputs/task10d/phase2/nominal_selected.json'
    nominal_run = ROOT/'outputs/task10d/phase2/nominal_preview01'
    visual_review = read(args.visual_review)
    records = []
    for manifest, run in ((nominal_path, nominal_run), (args.variants, args.variant_run)):
        for record in read(manifest)['primitives']:
            primitive = load_primitive(record)
            meta = primitive.reference.reference.metadata
            source = meta['source']
            source_id = source.get('candidate_id', meta.get('source_event', {}).get('event_id'))
            item = {key: record[key] for key in ('primitive_id','family','active_hand','role',
                'reference_npz','reference_sha256','placement','target','annotations')}
            item.update(reference_json=str(Path(record['reference_npz']).with_suffix('.json')),
                reference_json_sha256=primitive.metadata_sha256,
                stance=record.get('stance', meta.get('stance', 'not independently classified')),
                source={key: source[key] for key in ('source_path','source_sha256','actor',
                    'source_rate_hz','source_frame_range_inclusive','selected_frame_range_inclusive',
                    'outward_completion_frame','anatomical_hand','coordinate_system') if key in source},
                source_event_id=source_id,
                source_time_convention='Original inclusive 120 Hz frames; crop starts at zero. Adapted source_frames retain lineage; processed recovery timing is declared separately.',
                asset={'xml':'assets/task10c/h2/h2.xml',
                    'xml_sha256':meta['asset_sha256'], 'joint_names':list(JOINT_NAMES),
                    'nq':38,'nv':37,'quaternion_order':'wxyz',
                    'qvel_root_angular_frame':meta['qvel_root_angular_frame']},
                source_to_world_placement=record.get('source_to_world_placement',
                    {'xy':[0.,0.],'yaw':0.,'z':0.,'description':'Approved reference coordinates retained; identity reset placement.'}),
                provenance_class=record.get('provenance_class', meta.get('provenance_category')),
                diversity=record.get('diversity_rationale', meta.get('diversity_rationale')),
                limitations=record.get('limitations', []),
                timing_status='Source and representative clocks are runtime transforms, not separate styles. No TTC interval qualified; inherited Task10C ranges are historical.',
                generation_record=record.get('generation_record', relative(Path(manifest).parent/'generation.json')),
                source_manifest=relative(manifest),
                review_status='Selected for Phase 2 user review; not formal acceptance or canonical artifact promotion.')
            # Retain explicit spatial/recovery processing provenance without
            # embedding solver traces or duplicating spatial arrays.
            for key in ('processing', 'terminal_timing_processing', 'derived_processing', 'derived_terminal_timing',
                        'diagnostic_record', 'source_reference'):
                if key in record:
                    item[key] = record[key]
            if 'derived_terminal_timing' in item:
                item['derived_terminal_timing']['source_frame_mapping'] = (
                    'Hold repeats the last retained source-frame coordinate, possibly fractional '
                    'after 50 Hz resampling; no additional recorded frame is invented.')
            item['visual_review'] = visual_review['primitives'][record['primitive_id']]
            item['limitations'] = list(item['limitations']) + item['visual_review'].get('retained_limitations', [])
            if 'diagnostics' in record:
                item['native_measurements'] = record['diagnostics']
            item['annotations'].setdefault('guard_at_launch_s', 0.)
            item['annotations'].setdefault('extension_center_semantics',
                'Design aim/sweep center, not maximum elbow extension or closest approach.')
            item['annotations'].setdefault('preparation_interval_s',
                [0., item['annotations']['nominal_first_entry_s']])
            item['annotations'].setdefault('withdrawal_interval_s',
                [item['annotations']['withdrawal_start_s'], item['annotations']['terminal_guard_start_s']])
            item['annotations'].setdefault('terminal_guard_interval_s',
                [item['annotations']['terminal_guard_start_s'], item['annotations']['end_s']])
            item['annotations']['original_source_outward_annotation_time_s'] = (
                source['outward_completion_frame'] - source['source_frame_range_inclusive'][0]
                ) / source['source_rate_hz']
            item['previews'] = {}
            for clock in ('source','ttc0700'):
                label = record['primitive_id']+'_'+clock
                summary_path = run/f'{label}.json'
                if not summary_path.exists():
                    continue
                summary = read(summary_path)
                if (summary['reference_sha256'] != primitive.reference.sha256 or
                        summary['reference_json_sha256'] != primitive.metadata_sha256):
                    raise ValueError(f'Stale preview reference identity: {label}')
                for name in ('external.mp4','perception.mp4','sequence_review.jpg'):
                    if not (run/label/name).is_file():
                        raise FileNotFoundError(run/label/name)
                item['previews'][clock] = {
                    'external_video':relative(run/label/'external.mp4'),
                    'camera_video':relative(run/label/'perception.mp4'),
                    'sequence_review':relative(run/label/'sequence_review.jpg'),
                    'summary':relative(summary_path),
                    **{key:summary[key] for key in ('clock_rate','replay_frozen_target_first_entry_s',
                        'moving_head_first_encounter_s','recovery_complete_s','first_sampled_complete_s',
                        'terminal_glove_distances_m','video_frames')}}
            if 'source' not in item['previews']:
                raise ValueError('Every selected motion needs its complete source-clock preview')
            records.append(item)
    payload = {'schema':'task10d_phase2_library_index_v1',
        'status':'Phase 2 complete for user review; acceptance remains the user decision',
        'scope':'Isolated reset scenarios of human-motion-derived kinematic H2; dynamic frozen Fresh96 G1; no training, live switching, impact model or broad coverage claim.',
        'motion_storage':'References point to existing selected NPZ/JSON pairs; no duplicated TTC/snapshot trajectory bank.',
        'runtime_contract':{'physics_dt_s':.005,'g1_control_dt_s':.02,'glove_radius_m':.08,
            'g1_head_radius_m':.06,'encounter_distance_m':.14,'both_glove_safe_clearance_m':.16,
            'h2_mocap_links':32,'h2_contacts':False,'camera_changes':False,
            'camera_revision':'Deferred to Task 11'},
        'source_inventory':'outputs/task10d/phase2/survey/survey.json',
        'source_exclusions':'outputs/task10d/phase2/survey/summary.json',
        'raw_conversion_checks':'outputs/task10d/phase2/variants/raw_checks.json',
        'visual_review':relative(args.visual_review),
        'variant_exclusions':read(args.variants).get('excluded_raw_events', []),
        'primitives':records,
        'stop_boundary':'No Phase 3 or Phase 4 authorization; no automatic expansion or training.'}
    out.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'index':relative(out),'primitives':len(records),
                      'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    main()
