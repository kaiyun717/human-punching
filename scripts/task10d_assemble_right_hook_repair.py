"""Join disjoint target-strike and late-recovery edits for the one RH candidate."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np

from human_punching.task10c.references import StoredReference, arrays_from_qpos
from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT
from human_punching.task10c.targets import sha256
from human_punching.task10d.adaptation import diagnostics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-draft', type=Path, required=True)
    parser.add_argument('--repaired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to((ROOT/'outputs/task10d/phase2/variants/right_hook_repair01').resolve())
    out.mkdir(parents=True, exist_ok=False)
    target_ref = StoredReference.load(args.target_draft)
    repaired = StoredReference.load(args.repaired)
    source_path = ROOT/'outputs/task10d/phase2/variants/raw/right_hook_014_02074/right_hook.npz'
    raw = StoredReference.load(source_path)
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    arm_names = [n for n in JOINT_NAMES if n.startswith('right_') and
                 any(s in n for s in ('shoulder', 'elbow', 'wrist'))]
    qa = np.array([int(model.joint(n).qposadr[0]) for n in arm_names])
    other = np.setdiff1d(np.arange(model.nq), qa)
    assert np.array_equal(raw.time_s, repaired.time_s)
    assert np.array_equal(raw.arrays['qpos'][:, other], repaired.arrays['qpos'][:, other])
    unchanged = raw.time_s <= .9+1e-9
    assert np.array_equal(raw.arrays['qpos'][unchanged], repaired.arrays['qpos'][unchanged])
    query = target_ref.arrays['pre_easing_reference_time_s']
    delta = repaired.sample(query)['qpos'][:, qa]-raw.sample(query)['qpos'][:, qa]
    # The target fit ends at .90 s and the recovery edit is exactly zero there.
    # This preserves target approach/strike without solving or blending it again.
    qpos = target_ref.arrays['qpos'].copy()
    qpos[:, qa] += delta
    arrays = arrays_from_qpos(target_ref.time_s.copy(), qpos)
    for key in ('source_frames', 'pre_easing_reference_time_s'):
        arrays[key] = target_ref.arrays[key].copy()
    metadata = deepcopy(target_ref.metadata)
    repair_record = {'method':'Disjoint right-arm recovery joint correction applied at retained source queries after target adaptation; dependent arrays rebuilt.',
        'source_repair_npz':str(args.repaired.resolve().relative_to(ROOT)),
        'source_repair_npz_sha256':sha256(args.repaired),
        'source_repair_json_sha256':sha256(args.repaired.with_suffix('.json')),
        'raw_reference_npz_sha256':sha256(source_path),
        'unchanged_through_source_s':.9, 'modified_joint_names':arm_names,
        'revised_final_guard':'Final right-arm pose uses the repaired source guard; original overlapping endpoint is not restored.',
        'terminal_easing':'Existing monotone source clock and stationary endpoint hold retained; repair follows source queries.',
        'provenance':'Derived recovery repair, not an unmodified recorded return.'}
    metadata['derived_recovery_repair'] = repair_record
    # The timing stage still holds its input endpoint, but that endpoint now
    # includes the recovery repair. Do not inherit the raw-guard claim.
    metadata['derived_terminal_timing'].update(
        original_final_qpos_preserved_exactly=False,
        repaired_final_qpos_preserved_by_terminal_easing=True,
        endpoint_scope='Right-arm endpoint is revised by derived_recovery_repair; root and nonactive endpoint coordinates are preserved. Terminal easing holds this repaired endpoint.')
    metadata['source_guard_preparation_interval_s'] = deepcopy(metadata['guard_preparation_interval_s'])
    metadata['guard_preparation_interval_s'] = [0., float(metadata['strike_center_time_s'])]
    metadata['annotation_rationale'] = ('Guard/preparation/approach interval ends at the adapted native aim center. '
        'Original source preparation and reach landmark remain in source_guard_preparation_interval_s and source_event; '
        'first entry and closest approach are measured separately. Recovery repair changes the final right-arm guard.')
    target = np.asarray(metadata['nominal_target_adaptation']['target']['head_world'])
    check = diagnostics(model, arrays, target, 'right')
    metadata.update(nominal_first_entry_s=check['first_entry_s'],
        nominal_closest_approach_s=check['active_closest_approach_s'])
    destination = out/'right_hook.npz'
    np.savez_compressed(destination, **arrays)
    write_json(destination.with_suffix('.json'), metadata)
    target_manifest = json.loads((args.target_draft.parent.parent/'variant_manifest.json').read_text())
    before = deepcopy(target_manifest['primitives'][0])
    before.update(primitive_id='right_hook_014_02074_before', role='diagnostic_before',
        review_status='Unrepaired late recovery; diagnostic comparison only, never accepted.')
    candidate = deepcopy(target_manifest['primitives'][0])
    candidate.update(reference_npz=str(destination.relative_to(ROOT)), reference_sha256=sha256(destination),
        reference_json_sha256=sha256(destination.with_suffix('.json')), role='candidate',
        derived_recovery_repair=repair_record, diagnostics=check,
        derived_terminal_timing=deepcopy(metadata['derived_terminal_timing']),
        review_status='Candidate pending complete visual review and explicit user acceptance; seven accepted motions remain unchanged.')
    candidate['annotations'].update(nominal_first_entry_s=check['first_entry_s'],
        nominal_closest_approach_s=check['active_closest_approach_s'],
        guard_preparation_approach_interval_s=metadata['guard_preparation_interval_s'],
        source_reach_annotation_s=metadata['source_strike_annotation_time_s'])
    candidate['limitations'] = ['Prescribed kinematic H2; no impact, torque, dynamic balance or avoidance claim.',
        'Distinct recorded tighter-hook event with derived strike targeting, repaired right-arm recovery and derived terminal easing.',
        'Right-arm terminal guard is deliberately revised because original retargeted endpoint intersected own head.',
        'One isolated reset/real target and source clock; no coverage or supported TTC range claim.']
    write_json(out/'preview_manifest.json', {'schema':'task10d_phase2_primitives_v1',
        'status':'Authorized focused RH attempt; seven accepted punch references unchanged',
        'primitives':[before,candidate]})
    write_json(out/'candidate_manifest.json', {'schema':'task10d_phase2_primitives_v1',
        'status':'Candidate only; no acceptance or automatic library promotion', 'primitives':[candidate]})
    rebuilt = arrays_from_qpos(arrays['time_s'], arrays['qpos'])
    preservation = {'target_strike_through_source_0_9_s_exact':bool(np.array_equal(
        qpos[query<=.9+1e-9],target_ref.arrays['qpos'][query<=.9+1e-9])),
        'nonactive_coordinates_exact':bool(np.array_equal(qpos[:,other],target_ref.arrays['qpos'][:,other])),
        'dependent_arrays_max_error':max(float(np.max(abs(v-arrays[k]))) for k,v in rebuilt.items() if v.dtype.kind in 'fc'),
        'maximum_terminal_qvel':float(abs(arrays['qvel'][-1]).max()),
        'input_target_npz_sha256':sha256(args.target_draft), 'input_repaired_npz_sha256':sha256(args.repaired),
        'assembler_sha256':sha256(Path(__file__)), 'diagnostics':check}
    write_json(out/'assembly_checks.json',preservation)
    print(json.dumps({'candidate':str(destination),'sha256':sha256(destination),**preservation},indent=2))


if __name__ == '__main__':
    main()
