"""Adapt explicit surveyed events, retaining recorded paths and family guards."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np

from human_punching.task10c.references import StoredReference, arrays_from_qpos
from human_punching.task10c.robot import H2_XML, ROOT
from human_punching.task10c.targets import sha256
from .adaptation import diagnostics, solve_family, write_json


def ease_terminal(arrays: dict, config: dict) -> tuple[dict, dict]:
    """Monotone C2 source clock, derivative 1→0, preserves final pose exactly."""
    tail = float(config['source_tail_s'])
    if config['duration_multiplier'] != 2.0:
        raise ValueError('This explicit quartic terminal clock requires duration multiplier 2')
    old_end = float(arrays['time_s'][-1])
    start = old_end-tail
    end = start+2*tail
    times = np.arange(0., end+config['final_pose_hold_s']+1e-9, .02)
    u = np.clip((times-start)/(2*tail), 0., 1.)
    eased = start+tail*(2*u-2*u**3+u**4)
    query = np.where(times <= start, times, eased)
    query = np.clip(query, 0., old_end)
    ref = StoredReference(arrays, {})
    qpos = ref.sample(query)['qpos']
    # Exact endpoint arrays avoid an interpolation ulp looking like drift.
    qpos[times >= end-1e-9] = arrays['qpos'][-1]
    output = arrays_from_qpos(times, qpos)
    output['source_frames'] = np.interp(query, arrays['time_s'], arrays['source_frames'])
    output['pre_easing_reference_time_s'] = query
    return output, {'method': 'C2 monotone terminal source clock, s=start+tail*(2u-2u^3+u^4)',
        'source_tail_interval_s': [start, old_end], 'processed_tail_interval_s': [start, end],
        'processed_terminal_hold_interval_s': [end, float(times[-1])],
        'start_rate': 1., 'end_rate': 0., 'endpoint_clock_accelerations': [0., 0.],
        'original_final_qpos_preserved_exactly': bool(np.array_equal(qpos[-1], arrays['qpos'][-1])),
        'source_frame_mapping': 'Original frame lineage follows monotone source query; the last retained source-frame coordinate (possibly fractional after 50 Hz resampling) repeats during generated hold.',
        'provenance': 'Derived recovery timing processing of this recorded-style variant; not another recorded style or runtime TTC variant.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/task10d/phase2_variants.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--events', nargs='+')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = (args.output or ROOT/config['output']).resolve()
    output.relative_to((ROOT/'outputs/task10d/phase2/variants').resolve())
    if output.exists():
        raise FileExistsError('Preserve previous candidates; choose a new output directory')
    output.mkdir(parents=True)
    target = np.asarray(config['target']['head_world'])
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    selected = args.events or list(config['events'])
    entries, comparisons = [], {}
    for event_id in selected:
        cfg = config['events'][event_id]
        path = ROOT/cfg['reference']
        if sha256(path) != cfg['sha256']:
            raise ValueError(f'Raw event hash differs: {event_id}')
        ref = StoredReference.load(path)
        family, side = cfg['family'], cfg['active_hand']
        if (family, side) != (ref.metadata['family'], ref.metadata['active_hand']):
            raise ValueError('Explicit family/hand differs from source')
        poses, placed, design = solve_family(model, ref, cfg, target, config['contact_center_offset_m'])
        adapted = arrays_from_qpos(ref.time_s.copy(), poses)
        adapted['source_frames'] = ref.arrays['source_frames'].copy()
        arrays, easing = ease_terminal(adapted, config['terminal_easing'])
        before = diagnostics(model, arrays_from_qpos(ref.time_s, placed), target, side)
        after = diagnostics(model, arrays, target, side)
        rebuilt = arrays_from_qpos(arrays['time_s'], arrays['qpos'])
        regeneration = max(float(np.max(abs(value-arrays[key]))) for key, value in rebuilt.items() if value.dtype.kind in 'fc')
        center = design['strike_center_time_s']
        metadata = deepcopy(ref.metadata)
        metadata.update({'schema': 'task10d_h2_reference_v1', 'primitive_id': event_id,
            'strike_time_s': center, 'strike_center_time_s': center,
            'source_strike_annotation_time_s': ref.metadata['strike_time_s'],
            'end_time_s': float(arrays['time_s'][-1]),
            'recovery_duration_s': float(arrays['time_s'][-1]-center),
            'withdrawal_interval_s': [center, easing['processed_tail_interval_s'][1]],
            'terminal_guard_interval_s': easing['processed_terminal_hold_interval_s'],
            'nominal_first_entry_s': after['first_entry_s'], 'nominal_closest_approach_s': after['active_closest_approach_s'],
            'nominal_target_adaptation': design | {'target': config['target'], 'source_npz': str(path),
                'source_npz_sha256': sha256(path), 'source_json_sha256': sha256(path.with_suffix('.json'))},
            'derived_terminal_timing': easing, 'supported_ttc_range_s': None,
            'timing_range_status': 'Only source/basic integration clocks previewed; no supported TTC range established.',
            'annotation_rationale': 'Own-event source projection selects design aim center; first entry/closest approach are measured separately. Original source frames follow processed clock through terminal easing.',
            'runtime_placement': {'xy': [0., 0.], 'yaw': 0.},
            'dependent_arrays': 'Recomputed by native H2 FK after spatial adaptation and terminal clock processing.'})
        destination = output/event_id
        destination.mkdir()
        result = destination/(family+'.npz')
        np.savez_compressed(result, **arrays)
        write_json(result.with_suffix('.json'), metadata)
        checks = {'raw_placed': before, 'candidate': after, 'design': design, 'terminal_easing': easing,
            'dependent_array_regeneration_max_absolute_error': regeneration,
            'first_pose_exactly_matches_placed_source': bool(np.array_equal(arrays['qpos'][0], placed[0])),
            'final_pose_exactly_matches_placed_source': bool(np.array_equal(arrays['qpos'][-1], placed[-1])),
            'pre_easing_root_qpos_exactly_matches_placed_source': bool(np.array_equal(poses[:, :7], placed[:, :7]))}
        write_json(destination/'diagnostics.json', checks)
        comparisons[event_id] = checks
        entries.append({'primitive_id': event_id, 'family': family, 'active_hand': side, 'role': 'variant',
            'reference_npz': str(result.relative_to(ROOT)), 'reference_sha256': sha256(result),
            'reference_json_sha256': sha256(result.with_suffix('.json')),
            'placement': {'xy': [0., 0.], 'yaw': 0.}, 'source_to_world_placement': design['source_to_world_placement'],
            'target': config['target'], 'annotations': {'extension_center_s': center, 'design_aim_center_s': center,
                'nominal_first_entry_s': after['first_entry_s'], 'nominal_closest_approach_s': after['active_closest_approach_s'],
                'withdrawal_start_s': center, 'terminal_guard_start_s': easing['processed_tail_interval_s'][1],
                'end_s': float(arrays['time_s'][-1])},
            'source': ref.metadata['source'], 'source_event': ref.metadata['source_event'],
            'stance': ref.metadata['stance'], 'provenance_class': 'recorded_style_variant_with_derived_spatial_and_terminal_timing_processing',
            'diversity_rationale': ref.metadata['diversity_rationale'], 'derived_terminal_timing': easing,
            'diagnostics': after, 'review_status': 'candidate; pending normal-speed complete visual review',
            'source_reference': {'npz': str(path.relative_to(ROOT)), 'npz_sha256': sha256(path), 'json_sha256': sha256(path.with_suffix('.json'))},
            'limitations': ['Kinematic motion, not physical execution or impact dynamics.',
                'Native source guards retained; no shared guard or exact initial-footprint return.',
                'Source family/event bounds are distinct from processed source clock and frozen-target entry.',
                'Sparse correction fit is capped; convergence is not a prerequisite or claimed guarantee.']})
        print(json.dumps({'event_id': event_id, 'path': str(result), 'diagnostics': after}), flush=True)
    write_json(output/'generation.json', {'config': config, 'config_sha256': sha256(args.config),
        'generator_sha256': sha256(Path(__file__)), 'adaptation_sha256': sha256(Path(__file__).with_name('adaptation.py')),
        'h2_xml_sha256': sha256(H2_XML), 'processed_event_ids': selected, 'training_updates': 0, 'physics_steps': 0})
    write_json(output/'variant_manifest.json', {'schema': 'task10d_phase2_primitives_v1', 'primitives': entries,
        'excluded_raw_events': config['excluded_raw_events'], 'status': 'candidates; visual review pending'})
    write_json(output/'diagnostics.json', comparisons)


if __name__ == '__main__':
    main()
