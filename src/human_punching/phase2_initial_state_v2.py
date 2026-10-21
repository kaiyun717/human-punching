"""Exact independent initial states; future bitwise drift stays diagnostic.

The unchanged v1 adapter still owns all native/controller writes. This explicit
v2 contract checks the user-required initial identity after a declared refresh;
it never converts an old future-trajectory parity failure into a v1 pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import phase2_native_state as native
from .phase2_domain import fingerprint
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json
from .wbc.release_baseline_probe import _tensor_view

VERSION='task10b_initial_execution_state_identity_v2'
WITNESS='task10b_initial_execution_state_identity_witness_v2'
CAPTURE='task10b_native_shared_bank_capture_initial_identity_v2'
PARENT_SHA='32ee676197b2de746dd3674680229841095051fee41452f5ce0c0b2a1e6f8758'
DIAGNOSTIC_SHA='b646aa8148c773c88ae728b1ceaa2a9327d4214d2037d876e43e5fd7aca17d33'
MAX_BYTES=64*1024**2


def owned(path):
    p=Path(path);p=(p if p.is_absolute() else PROJECT_ROOT/p).resolve(strict=True)
    if not p.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError('Owned immutable initial-state evidence required')
    return p


def read_bound(item):
    path=owned(item['path'])
    if path.stat().st_size>16*1024**2 or sha256_file(path)!=item['sha256']:
        raise ValueError('Changed or oversized initial-state evidence')
    return path,json.loads(path.read_text())


def binding(path):
    path=owned(path);return dict(path=str(path.relative_to(PROJECT_ROOT.resolve())),sha256=sha256_file(path))


def exact_arrays(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def _saved(path):
    """Read the fixed sibling archive, checking its declared bytes and identity.

    A job's absolute archive spelling may use the proven compute mount. The
    sibling is identified by fixed basename plus SHA; no source prefix rewrite
    or missing-file substitution occurs.
    """
    path=owned(path);meta=json.loads(path.read_text());archive=path.with_name('state.npz')
    if (meta.get('schema')!=native.VERSION or Path(meta['archive_path']).name!='state.npz'
            or sha256_file(archive)!=meta['archive_sha256'] or archive.stat().st_size>1024**2):
        raise ValueError('Saved witness archive changed')
    from .phase2_cache import _preflight
    _preflight(archive,meta['archive_sha256'],1024**2)
    states=[]
    with np.load(archive,allow_pickle=False) as arrays:
        for row in meta['worlds']:
            values={name:({k:arrays[key].copy() for k,key in value.items()} if name=='adapter_state' else
                arrays[value['array']].copy() if isinstance(value,dict) and set(value)=={'array'} else value)
                for name,value in row.items()}
            states.append(native.SavedExecutionState(**values))
    saved=native.NativeComparisonSnapshot(tuple(states),meta['inventory_json'],meta['version'])
    if saved.identity!=meta['identity']:raise ValueError('Saved witness identity differs')
    return saved


def assess_witness(diagnostic_path,diagnostic_sha256):
    """Independently recompute all54 retained snapshots; no native execution."""
    path,d=read_bound(dict(path=diagnostic_path,sha256=diagnostic_sha256));folder=path.parent
    if (sha256_file(native.__file__)!=PARENT_SHA or d.get('schema')!='task10b_native_state_difference_diagnostic_v1'
            or d.get('source_sha256')!=DIAGNOSTIC_SHA or d.get('parent_source_sha256')!=PARENT_SHA
            or d.get('diagnostic_complete') is not True or d.get('exception') is not None
            or len(d.get('snapshots',[]))!=54 or len(d.get('native_calls',[]))!=19
            or d.get('explicit_step_calls_issued')!=12 or d.get('explicit_step_calls_returned')!=12
            or d.get('unknown_extent_step_calls')!=0 or d.get('unattempted_explicit_step_calls')!=0
            or not d.get('slurm_job_id')):
        raise ValueError('Complete unchanged counted twelve-interval diagnostic required')
    packet_path=folder.parent/'packet.json';packet=json.loads(packet_path.read_text())
    execution=json.loads((folder.parent/'execution.json').read_text())
    terminal=json.loads((folder.parent/'terminal_accounting.json').read_text())
    if (packet.get('mode')!='native-state-diagnostic' or packet.get('planned_explicit_integration_world_steps')!=12
            or execution.get('packet_sha256')!=sha256_file(packet_path)
            or terminal.get('packet_sha256')!=sha256_file(packet_path)
            or execution.get('job_id')!=d['slurm_job_id'] or terminal.get('job_id')!=d['slurm_job_id']):
        raise ValueError('Counted diagnostic allocation/packet identity differs')
    for name,digest in packet['files'].items():
        if (sha256_file(owned(name))!=digest
                or sha256_file(owned(folder.parent/'source_snapshot'/name))!=digest):
            raise ValueError('Diagnostic live/archive source/input changed: '+name)
    arrays={};snapshots=[];total=0
    from .phase2_cache import _preflight
    for index,item in enumerate(d['snapshots']):
        p=folder/f'snapshot_{index:03d}.npz'
        if item.get('index')!=index or Path(item['path']).name!=p.name or sha256_file(p)!=item['sha256']:
            raise ValueError('Lossless diagnostic snapshot identity differs')
        header=_preflight(p,item['sha256'],1024**2);total+=header.uncompressed_array_bytes
        if total>MAX_BYTES:raise ValueError('Diagnostic retained snapshot cap exceeded')
        with np.load(p,allow_pickle=False) as archive:values={k:archive[k].copy() for k in archive.files}
        if set(values)!=set(item['arrays']):raise ValueError('Snapshot inventory differs')
        for key,value in values.items():
            if (list(value.shape)!=item['arrays'][key]['shape'] or value.dtype.str!=item['arrays'][key]['dtype']
                    or value.dtype.kind not in 'biuf' or not np.isfinite(value).all()):
                raise ValueError('Snapshot numeric schema differs')
        arrays[index]=values;snapshots.append(binding(p))
    saved=_saved(folder/'saved_start/state.json');inventory=json.loads(saved.inventory_json)
    if len(saved.states)!=1 or saved.states[0].native_state_size!=410:
        raise ValueError('Accepted single-world410 witness required')
    if set(d['environment_counters'])!={'source','restored'}:raise ValueError('Both independent environment counters required')
    for key,value in d['environment_counters'].items():
        if value.get('hard_faults') or value.get('physics_steps')!=8 or value.get('actor_updates')!=1:
            raise ValueError('Diagnostic fault ledger differs')
    if sum(row.get('method')=='step' for row in d['native_calls'])!=12:
        raise ValueError('Exact completed step-call inventory required')
    for index,call in enumerate(d['native_calls']):
        if call.get('index')!=index or call.get('call_returned') is not True:
            raise ValueError('Unknown or inconsistent native-call extent')
        for suffix in ('intent','returned','terminal'):
            recorded=json.loads((folder/f'call_{index:03d}_{suffix}.json').read_text())
            if (recorded.get('index')!=index or recorded.get('method')!=call['method']
                    or (suffix!='intent' and recorded.get('call_returned') is not True)):
                raise ValueError('Durable native-call extent evidence differs')
    required_native={'native.integration','native.time','native.qpos','native.qvel','native.qacc_warmstart',
                     'native.ctrl','native.qfrc_applied','native.xfrc_applied'}
    required_controller={'controller.'+name for name in inventory['physical_controller_fields'] if name!='hard_faults'}
    required_controller|={'controller.phase_denominator_s','controller.held_raw_action'}
    required_static={'static.'+name for name in native.STATIC_CONTROLLER_ARRAYS}
    initial=[];future=[]
    labels=['immediately_after_exact_restore','after_common_forward_refresh']+[f'paired_integration_{i}' for i in range(1,5)]
    if [item['label'] for item in d['comparisons']]!=labels:raise ValueError('Exact declared diagnostic comparisons required')
    for item in d['comparisons']:
        a,b=arrays[item['source_snapshot']],arrays[item['restored_snapshot']]
        differences=[]
        for key in sorted(set(a)|set(b)):
            numeric_equal=key in a and key in b and a[key].shape==b[key].shape and a[key].dtype==b[key].dtype and np.array_equal(a[key],b[key])
            if item['fields'][key]['exact_equal']!=numeric_equal:raise ValueError('Stored comparison does not match lossless evidence')
            if not numeric_equal:
                differences.append(dict(field=key,maximum_absolute_difference=(float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)),initial=0)) if key in a and key in b and a[key].shape==b[key].shape else None)))
        canonical=exact_arrays(a['native.integration'],b['native.integration'])
        if item['label'] in labels[:2]:
            for prefix,expected in [('native.',required_native),('controller.',required_controller),('static.',required_static)]:
                if {k for k in a if k.startswith(prefix)}!=expected or {k for k in b if k.startswith(prefix)}!=expected:
                    raise ValueError('Initial native/controller/static inventory is incomplete')
                if not all(exact_arrays(a[k],b[k]) for k in expected):
                    raise ValueError('Initial canonical/controller/static byte identity failed')
            previous=saved.states[0].previous_raw_action[None];held=saved.states[0].held_raw_action[None]
            previous_native=previous.astype(a['controller.last_action'].dtype)
            held_native=held.astype(a['controller.held_raw_action'].dtype)
            if (not exact_arrays(a['native.integration'],saved.states[0].native_integration_state[None])
                    or not np.array_equal(previous_native.astype(previous.dtype),previous)
                    or not np.array_equal(held_native.astype(held.dtype),held)
                    or not exact_arrays(a['controller.last_action'],previous_native)
                    or not exact_arrays(a['controller.held_raw_action'],held_native)):
                raise ValueError('Initial measured state does not bind the saved physical/controller state')
            initial.append(dict(label=item['label'],canonical410_exact=True,controller_history_exact=True,
                static_controller_exact=True,derived_differences=differences))
        else:future.append(dict(label=item['label'],canonical410_exact=canonical,differences=differences))
    policies=[]
    for item in d['policy_records']:
        p=folder/Path(item['path']).name
        if sha256_file(p)!=item['sha256']:raise ValueError('Recorded actor evidence changed')
        with np.load(p,allow_pickle=False) as data:policies.append({key:data[key].copy() for key in data.files})
    if len(policies)!=3 or any(not exact_arrays(policies[1][k],policies[2][k]) for k in ('observation','raw_action')):
        raise ValueError('Same-state recorded observation/action identity failed')
    restore=d['restore']
    if (restore.get('source_snapshot_identity')!=saved.identity or restore.get('physical_controller_identity')!=saved.physical_controller_identity
            or restore.get('complete_native_integration_equal') is not True or restore.get('physical_state_history_preserved') is not True):
        raise ValueError('Unchanged exact restore receipt does not bind the saved state')
    result=dict(schema=WITNESS,version=VERSION,initial_state_identity_passed=True,
        diagnostic=binding(path),diagnostic_packet=binding(packet_path),diagnostic_execution=binding(folder.parent/'execution.json'),
        diagnostic_terminal=binding(folder.parent/'terminal_accounting.json'),snapshots=snapshots,
        source_bindings_verified=len(packet['files']),saved_snapshot_identity=saved.identity,
        saved_physical_controller_identity=saved.physical_controller_identity,
        compiled_model_sha256=saved.states[0].model_sha256,static_controller_sha256=inventory['static_controller_sha256'],
        native_state_spec=16383,native_state_size=410,controller_inventory=sorted(required_controller),
        initial_comparisons=initial,observations_equal=True,actor_outputs_equal=True,
        future_bitwise_parity_passed=all(row['canonical410_exact'] for row in future),future_state_diagnostics=future,
        old_exact_parity_result_preserved=d['exact_parity_passed'],solver_cause='unknown',
        parent_adapter_sha256=PARENT_SHA,consumer_source_sha256=sha256_file(__file__),
        no_tolerance_used=True,zero_new_physics_steps=True,shared_start_feasibility_qualified=False)
    result['identity']=fingerprint(result);return result


def validate_witness(item):
    path,receipt=read_bound(item)
    if receipt.get('schema')!=WITNESS or receipt.get('version')!=VERSION:
        raise ValueError('Explicit v2 initial-identity witness required; legacy future parity is separate')
    current=assess_witness(receipt['diagnostic']['path'],receipt['diagnostic']['sha256'])
    if fingerprint(current)!=fingerprint(receipt):raise ValueError('Initial witness cannot be independently recomputed')
    return dict(path=str(path),sha256=sha256_file(path),identity=receipt['identity'],version=VERSION,
        initial_state_identity_passed=True,future_bitwise_parity_passed=receipt['future_bitwise_parity_passed'],solver_cause='unknown')


@dataclass(frozen=True)
class InitialRestoreProof:
    receipt:dict
    before:native.NativeComparisonSnapshot
    after:native.NativeComparisonSnapshot
    derived_before:dict
    derived_after:dict


def derived_outputs(env):
    """Retain bounded named solver outputs; differences do not earn parity."""
    data=getattr(env.sim,'wp_data',None)
    if data is None:raise ValueError('Explicit native derived-output inventory required')
    def copy(value,index=None):
        view=_tensor_view(value)
        return (view if index is None else view[index]).detach().cpu().numpy().copy()
    result={name:copy(getattr(data,name)) for name in
        ('qacc','qacc_smooth','qfrc_constraint','qfrc_smooth','actuator_force','solver_niter','nefc')}
    if (result['nefc'].shape!=(env.n,) or result['nefc'].dtype.kind not in 'iu'
            or np.any(result['nefc']<0) or np.any(result['nefc']>1024)):
        raise ValueError('Bounded per-world native constraint counts required')
    for name in ('qacc','qacc_smooth','qfrc_constraint','qfrc_smooth'):
        if result[name].shape!=(env.n,54):raise ValueError('Native derived vector schema differs')
    if result['actuator_force'].shape!=(env.n,48) or result['solver_niter'].shape!=(env.n,):
        raise ValueError('Native derived force/solver count schema differs')
    count=int(np.max(result['nefc'],initial=0))
    result['constraint_force']=copy(data.efc.force,(slice(None),slice(None,count)))
    if (result['constraint_force'].shape!=(env.n,count) or sum(x.nbytes for x in result.values())>1024**2
            or any(x.dtype.kind not in 'biuf' or not np.isfinite(x).all() for x in result.values())):
        raise ValueError('Bounded finite native derived outputs required')
    return result


@native._ordered_comparison
def restore_initial_comparison(env,snapshot,*,trial_id,reference_ids):
    """One fresh independent restore plus a declared no-integration refresh."""
    if sha256_file(native.__file__)!=PARENT_SHA:raise ValueError('Preserved native adapter changed')
    phases={state.reference_phase for state in snapshot.states}
    if len(phases)!=1:raise ValueError('One explicit initial phase required')
    derived_outputs(env)  # Check supported inventory before any restore write.
    receipt=native.restore_comparison(env,snapshot,trial_id=trial_id,reference_ids=reference_ids)
    def capture():
        return native.capture_comparison(env,held_raw_action=np.stack([state.held_raw_action for state in snapshot.states]),
            reference_phase=next(iter(phases)),support_interval_end_s=np.array([state.support_interval_end_s for state in snapshot.states]))
    before=capture();derived_before=derived_outputs(env)
    env.sim.forward();after=capture();derived_after=derived_outputs(env)
    expected=np.stack([state.native_integration_state for state in snapshot.states])
    for actual in (before,after):
        if (actual.physical_controller_identity!=snapshot.physical_controller_identity
                or not exact_arrays(np.stack([state.native_integration_state for state in actual.states]),expected)):
            raise RuntimeError('Exact canonical/controller/history identity failed at independent initialization')
    return InitialRestoreProof(dict(**receipt,initial_identity_version=VERSION,
        initial_state_identity_passed=True,exact_after_declared_forward_refresh=True,
        declared_initial_forward_refreshes=1,legacy_future_bitwise_gate_result='separate_failed_or_unmeasured_diagnostic',
        future_bitwise_parity_qualified=False,solver_cause='unknown',no_within_cycle_restore=True),
        before,after,derived_before,derived_after)


def capture_bank(catalog,domain,split,geometry,release,output,wall_seconds,*,witness,pre_test_freeze=None):
    """Delegate unchanged counted four-world capture; add a separate v2 receipt."""
    proof=validate_witness(witness)
    result=native.capture_shared_start_bank(catalog,domain,split,geometry,release,output,wall_seconds,pre_test_freeze=pre_test_freeze)
    folder=writable_path(output)
    wrapper={**result,'schema':CAPTURE,'source_sha256':sha256_file(__file__),
        'base_capture':binding(folder/'bank_capture.json'),'base_capture_source_sha256':PARENT_SHA,
        'initial_identity_version':VERSION,'initial_identity_witness':proof,
        'old_future_bitwise_result_preserved':True,'shared_start_feasibility_qualified':False}
    write_json(folder/'bank_capture_initial_identity_v2.json',wrapper);return wrapper


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('witness');q.add_argument('--diagnostic',required=True,type=Path);q.add_argument('--diagnostic-sha256',required=True);q.add_argument('--output',required=True,type=Path)
    q=sub.add_parser('capture-bank')
    for name in ('catalog','domain','geometry','release','output','witness'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--witness-sha256',required=True);q.add_argument('--split',choices=('train','validation','test'),required=True)
    q.add_argument('--wall-seconds',type=float,required=True);q.add_argument('--pre-test-freeze',type=Path)
    a=p.parse_args()
    if a.mode=='witness':
        result=assess_witness(a.diagnostic,a.diagnostic_sha256);folder=writable_path(a.output);folder.mkdir(parents=True,exist_ok=False)
        write_json(folder/'initial_identity_witness.json',result);print(json.dumps(dict(identity=result['identity'],initial_state_identity_passed=True,future_bitwise_parity_passed=result['future_bitwise_parity_passed'])));return 0
    result=capture_bank(a.catalog,a.domain,a.split,a.geometry,a.release,a.output,a.wall_seconds,
        witness=dict(path=str(a.witness),sha256=a.witness_sha256),pre_test_freeze=a.pre_test_freeze)
    return 0 if result['complete_capture'] else 1


if __name__=='__main__':raise SystemExit(main())
