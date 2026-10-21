"""Pack full sequential candidates with unchanged math and one proven logical root."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import phase2_runtime_pack as packing
from .phase2_runtime_namespace import evidence
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

VERSION = 'task10b_phase2_runtime_pack_logical_entry_v3'


def logical_output_adapter(root, logical_root, proof_path):
    root = Path(root).resolve(strict=True)
    logical = Path(logical_root)
    proof_path = Path(proof_path).resolve(strict=True)
    if not proof_path.is_relative_to(root) or not proof_path.is_file():
        raise ValueError('Owned exact namespace proof required')
    proof = json.loads(proof_path.read_text())
    if (str(logical) != proof.get('local_root') or not logical.is_absolute()
            or logical.resolve(strict=True) != root):
        raise ValueError('Declared logical root must be the exact proven alias of this owned project')
    namespace = evidence(root, proof_path)
    def adapt(path):
        owned = writable_path(path)
        if not owned.is_relative_to(root):
            raise ValueError('Packing output escaped owned root')
        return logical/owned.relative_to(root)
    return adapt, namespace


def execute(plan_path, output, *, wall_seconds, logical_project_root,
            runtime_namespace_proof, packet_path, packet_sha256):
    root = PROJECT_ROOT.resolve(strict=True)
    packet_path = Path(packet_path).resolve(strict=True)
    if not packet_path.is_relative_to(root) or sha256_file(packet_path) != packet_sha256:
        raise ValueError('Exact owned sequential V1 generation packet required')
    packet = json.loads(packet_path.read_text())
    plan_path = Path(plan_path).resolve(strict=True)
    proof_path = Path(runtime_namespace_proof).resolve(strict=True)
    expected_output = writable_path(root/packet['output'])
    if (packet.get('schema') != 'task10b_phase2_sequential_pipeline_packet_v1'
            or packet.get('mode') != 'runtime-pack' or packet.get('training') is not False
            or packet.get('allocation_wall_seconds') != 300 or packet.get('application_wall_seconds') != 240
            or packet.get('fresh_transition_cap') != 0 or packet.get('gpus') != 1
            or any(packet.get(k) is not False for k in
                   ('automatic_followup', 'promotion', 'g1_actor_execution', 'combined_scene'))
            or set(packet['inputs']) != {'plan'}
            or Path(packet['inputs']['plan']).resolve(strict=True) != plan_path
            or writable_path(output) != expected_output
            or packet['runtime_namespace_proof']['sha256'] != sha256_file(proof_path)
            or (root/packet['runtime_namespace_proof']['path']).resolve(strict=True) != proof_path):
        raise ValueError('Runtime packing request differs from immutable sequential V1 packet')
    if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds) or not 120 < wall_seconds <= packet['application_wall_seconds']-30
            or not packet['application_wall_seconds'] <= packet['allocation_wall_seconds']-30 <= 14370):
        raise ValueError('Finite child/application/allocation limits differ')
    required = {str(plan_path.relative_to(root)), str(proof_path.relative_to(root)),
                'src/human_punching/phase2_runtime_pack_entry_v3.py',
                'src/human_punching/phase2_runtime_pack.py', 'scripts/task10b_phase2_sequential_pipeline_v1.py'}
    if (not required <= packet.get('files', {}).keys()
            or packet['files']['src/human_punching/phase2_runtime_pack_entry_v3.py'] != sha256_file(__file__)):
        raise ValueError('Exact entry/packing/launcher source archive closure required')
    for relative, digest in packet['files'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Exact packet-relative archive member required')
        live = (root/rel).resolve(strict=True)
        saved = (packet_path.parent/'source_snapshot'/rel).resolve(strict=True)
        if (not live.is_relative_to(root)
                or not saved.is_relative_to((packet_path.parent/'source_snapshot').resolve(strict=True))
                or sha256_file(live) != digest or sha256_file(saved) != digest):
            raise ValueError('Live or archived runtime packing source/input changed')
    adapter, namespace = logical_output_adapter(root, logical_project_root, proof_path)
    if namespace != packet.get('runtime_namespace_evidence'):
        raise ValueError('Runtime packing namespace evidence differs from the packet')
    # The unchanged packing math sees the same physical folder through its
    # proven logical name. Only str(folder/filename) serialization differs.
    original = packing.writable_path
    original_input = packing.owned_input
    def verified_logical_input(binding):
        return adapter(original_input(binding))
    try:
        packing.writable_path = adapter
        packing.owned_input = verified_logical_input
        result = packing.pack(plan_path, output, wall_seconds=wall_seconds)
    finally:
        packing.writable_path = original
        packing.owned_input = original_input
    write_json(expected_output/'logical_entry_v3.json', dict(version=VERSION,
        packet_sha256=packet_sha256, source_sha256=sha256_file(__file__),
        logical_project_root=str(logical_project_root), resolved_project_root=str(root),
        namespace_evidence=namespace, scientific_packing_math_changed=False,
        reference_arrays_changed_by_adapter=False, underlying_status=result['status'],
        physical_qualification=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output', 'logical-project-root', 'runtime-namespace-proof', 'packet', 'packet-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    args = parser.parse_args()
    # Preserve the existing allocator guard in the original CLI.
    import os
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Runtime packing requires a counted Slurm CUDA allocation')
    result = execute(args.plan, args.output, wall_seconds=args.wall_seconds,
        logical_project_root=args.logical_project_root, runtime_namespace_proof=args.runtime_namespace_proof,
        packet_path=args.packet, packet_sha256=args.packet_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] == 'packed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
