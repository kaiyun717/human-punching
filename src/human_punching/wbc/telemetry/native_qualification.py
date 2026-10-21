"""Bounded observer A/A/B qualification; no learning or punch acceptance.

Two six-second fixtures (four-family moving references and a static common
 guard) each run observer off/off/on with identical deterministic actions.
A/A measures baseline determinism. A/B must remain bitwise equal; uncertainty
is reported as a failed qualification, never absorbed into a wider tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import time
import numpy as np
import torch

from human_punching.wbc.native_cycle_probe import (ProbeMetrics, _events,
    build_cycle_probe_cfg, initialize_launch_drive, placed_request, save_compiled_model,
    allocated_device_metadata, _save_npz)
from human_punching.wbc.guard_probe import build_guard_probe_cfg
from human_punching.wbc.cycle_trace import CycleTraceRecorder, TraceBindings, derive_kinematics, classify_contacts
from human_punching.wbc.native_probe import _copy, _configure_caches, _require_allocated_cuda, audit_probe_arrays
from human_punching.wbc.provenance import writable_path, write_json, sha256_file, runtime_record
from human_punching.wbc.runtime.bundle import load_runtime_bundle
from .runtime_telemetry import (ObserverAction, ObserverActionCfg, ObserverOptions,
    configure_observer, get_observer, native_tensor)
from .posthoc_qualification import command_drive_audit, offline_audit
from .streams import verify_native_stream_capture

ENVIRONMENTS=16
POLICY_STEPS=300
PHYSICAL_FIELDS=('qpos','qvel','ctrl','time','actuator_force','qfrc_actuator','qfrc_applied',
    'xfrc_applied','qacc','qfrc_bias','qfrc_constraint','xpos','xquat','geom_xpos','cvel')


def compare_rollouts(a,b):
    result={}
    for key in ('qpos','qvel','ctrl','time','actuator_force','qfrc_actuator'):
        x,y=np.asarray(a[key]),np.asarray(b[key])
        shape=x.shape==y.shape and x.dtype==y.dtype
        exact=shape and x.tobytes()==y.tobytes()
        difference=(x!=y).reshape(x.shape[0],-1).any(-1) if shape else None
        first=int(np.flatnonzero(difference)[0]) if shape and difference.any() else None
        result[key]={'bitwise_equal':bool(exact),
            'initial_bitwise_equal':bool(shape and x[0].tobytes()==y[0].tobytes()),
            'first_unequal_value_tick':first,
            'first_second_maximum_absolute_difference':float(np.max(abs(x[:201].astype(float)-y[:201].astype(float)))) if shape else None,
            'maximum_absolute_difference':float(np.max(abs(x.astype(float)-y.astype(float)))) if shape else None}

    return {'passed':all(row['bitwise_equal'] for row in result.values()),'fields':result}


class NonmutationAudit:
    """Read-only native-array witnesses around observer work, including initial FK."""
    def __init__(self):
        self.arrays=None;self.before=None;self.bad=None;self.calls=0

    def capture(self,env):
        if self.arrays is None:
            import warp as wp
            self.arrays={key:native_tensor(getattr(env.sim.data,key)) for key in PHYSICAL_FIELDS}
            self.arrays['efc.force']=wp.to_torch(env.sim.wp_data.efc.force)
            for key in ('geom','worldid','pos','dist','dim','efc_address'):
                self.arrays['contact.'+key]=wp.to_torch(getattr(env.sim.wp_data.contact,key))
            for key in ('nacon','overflow'):
                self.arrays[key]=wp.to_torch(getattr(env.sim.wp_data,key))
            self.before={key:value.clone() for key,value in self.arrays.items()}
            self.bad={key:torch.zeros((),dtype=torch.bool,device=env.device) for key in self.arrays}
        else:
            for key,value in self.arrays.items():self.before[key].copy_(value)

    def verify(self,env):
        del env
        for key,value in self.arrays.items():
            # Byte comparison preserves NaN payloads in inactive native slots.
            same=(self.before[key].contiguous().view(torch.uint8)==value.contiguous().view(torch.uint8)).all()
            self.bad[key] |= ~same
        self.calls+=1

    def report(self):
        rows={key:bool(value.item()) for key,value in self.bad.items()} if self.bad else {}
        return {'passed':bool(rows) and not any(rows.values()),'checks':self.calls,'mutated_fields':rows}


class AuditedAction(ObserverAction):
    def process_actions(self,actions):
        audit=self._env._telemetry_nonmutation
        audit.capture(self._env)
        super().process_actions(actions)
        audit.verify(self._env)


@dataclass(kw_only=True)
class AuditedActionCfg(ObserverActionCfg):
    def build(self,env):return AuditedAction(self,env)


def instrument_cfg(cfg, *, enabled, auditor, recorded_states):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    if not enabled:return cfg
    configure_observer(cfg,ObserverOptions(enabled=True))
    old=cfg.actions['joint_pos']
    cfg.actions['joint_pos']=AuditedActionCfg(**{f.name:getattr(old,f.name) for f in fields(old) if f.init})
    def before(env):
        auditor.capture(env)
        return torch.zeros(env.num_envs,device=env.device)
    def after(env):
        auditor.verify(env)
        recorded_states.append({key:_copy(value) for key,value in get_observer(env).endpoint_state().items()})
        return torch.zeros(env.num_envs,device=env.device)
    native=cfg.metrics.pop('human_native_substep_valid')
    others=cfg.metrics
    cfg.metrics={'observer_nonmutation_before':MetricsTermCfg(func=before,per_substep=True),
                 'human_native_substep_valid':native,
                 'observer_nonmutation_after':MetricsTermCfg(func=after,per_substep=True),**others}
    return cfg


def offline_parity(model,raw,states,bindings=TraceBindings()):
    """Independent double-precision CPU FK and original CPU contact classifier."""
    maxima={};mismatch={name:0 for name in ('support_contact','support_patch_contact','forbidden_collision','human_g1_collision')}
    offsets=raw['contact_offsets']
    for env_index in range(ENVIRONMENTS):
        expected=derive_kinematics(model,raw['qpos'][:,env_index],raw['qvel'][:,env_index],bindings)
        for name,values in expected.items():
            if name not in states:continue
            err=float(np.max(abs(values[1:]-states[name][:,env_index])))
            maxima[name]=max(maxima.get(name,0.),err)
        for k in range(1,len(raw['qpos'])):
            lo,hi=offsets[k:k+2];owned=raw['contact_worldid'][lo:hi]==env_index
            contact={name:raw['contact_'+name][lo:hi][owned] for name in ('geom','pos','dist','force')}
            classified=classify_contacts(model,contact,expected['support_anchor_position_w'][k-1],bindings)
            for name in mismatch:mismatch[name]+=int(np.count_nonzero(classified[name]!=states[name][k-1,env_index]))
    return {'passed':all(v<=2.e-5 for v in maxima.values()) and not any(mismatch.values()),
            'maximum_absolute_fk_error':maxima,'fk_tolerance':2.e-5,
            'contact_boolean_mismatches':mismatch,
            'contact_semantics':'actual preceding interval at previous endpoint material anchors'}


def _run(bundle,destination,*,kind,enabled):
    from mjlab.envs import ManagerBasedRlEnv
    from .cycle_qualification import native_stream_scope
    import mujoco
    metrics=ProbeMetrics();states=[];audit=NonmutationAudit()
    cfg=build_cycle_probe_cfg(bundle,metrics) if kind=='punch' else build_guard_probe_cfg(bundle,metrics)
    cfg=instrument_cfg(cfg,enabled=enabled,auditor=audit,recorded_states=states)
    cfg.sim.nan_guard.output_dir=str(destination/'nan_guard')
    env=None;begin=time.perf_counter();error=None;wall=[];metadata=None;stream_capture=None
    destination.mkdir(exist_ok=False)
    try:
        with native_stream_scope(), torch.no_grad():
            env=ManagerBasedRlEnv(cfg,device='cuda:0')
            env._telemetry_nonmutation=audit
            model=env.sim.mj_model
            save_compiled_model(model,destination/'compiled_scene.mjb')
            env.reset(seed=10010)
            command=env.command_manager.get_term('motion')
            initialize_launch_drive(env);command.write_ledger.seal()
            clips=[bundle.clips[int(i)] for i in _copy(command.trajectory_ids)]
            contexts=[placed_request(c,o,t) for c,o,t in zip(clips,_copy(env.scene.env_origins),_copy(env.sim.data.time),strict=True)]
            cycle=CycleTraceRecorder(contexts,event_provider=_events)
            metrics.attach(cycle);metrics.native.capture(env);cycle.capture_launch(env)
            # Same deterministic bounded nonzero input probes the action pathway,
            # not policy quality. Shared action array is created once per run.
            base=torch.arange(48,device=env.device,dtype=torch.float32)
            for step in range(POLICY_STEPS):
                action=(.02*torch.sin(base[None,:]*.17+step*.11)).expand(ENVIRONMENTS,-1)
                torch.cuda.synchronize();start=time.perf_counter()
                env.step(action)
                torch.cuda.synchronize();wall.append(time.perf_counter()-start)
            if enabled:
                metadata=get_observer(env).metadata
                stream_capture=verify_native_stream_capture('cuda:0')
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        if env is not None:env.close()
    native=metrics.native.arrays();raw=metrics.cycle.raw_arrays() if metrics.cycle else {}
    hashes={}
    for name,value in (('native_trace',native),('raw_trace',raw)):
        hashes[name]=_save_npz(destination/(name+'.npz'),value)
    observed={name:np.stack([s[name] for s in states]) for name in states[0]} if states else {}
    hashes['observer_trace']=_save_npz(destination/'observer_trace.npz',observed)
    hashes['timing']=_save_npz(destination/'timing.npz',{'step_wall_s':np.asarray(wall)})
    controller,parity=audit_recorded_run(bundle,kind,native,raw,observed,
        destination/'compiled_scene.mjb',policy_steps=len(wall),enabled=enabled)
    report={'schema':'human_wbc_observer_qualification_run_v1','fixture':kind,'observer_enabled':enabled,
        'exception':error,'native_controller_audit':controller,'nonmutation':audit.report() if enabled else None,
        'offline_parity':parity,'observer_metadata':metadata,'stream_capture_roundtrip':stream_capture,'compiled_scene_sha256':sha256_file(destination/'compiled_scene.mjb') if (destination/'compiled_scene.mjb').exists() else None,'hashes':hashes,'policy_steps':len(wall),
        'physics_samples':len(native.get('time',[])),'wall_s':time.perf_counter()-begin,
        'step_wall_median_s':float(np.median(wall)) if wall else None,
        'step_wall_total_s':float(sum(wall)), 'steady_step_wall_mean_s':float(np.mean(wall[10:])) if len(wall)>10 else None,'training':False,'acceptance_qualified':False}
    write_json(destination/'report.json',report)
    return report,native


def audit_recorded_run(bundle,kind,native,raw,observed,model_path,*,policy_steps,enabled):
    """Use the already qualified nonzero-command and SO(3) trace verifiers."""
    import mujoco
    controller=command_drive_audit(native,bundle,kind) if policy_steps==POLICY_STEPS else {'status':'unexecuted'}
    complete_observer=bool(observed) and all(len(value)==POLICY_STEPS*4 for value in observed.values())
    parity=offline_audit(mujoco.MjModel.from_binary_path(str(model_path)),raw,observed) if enabled and complete_observer else None
    return controller,parity


def run_qualification(bundle,output):
    _require_allocated_cuda()
    runtime=runtime_record()
    if not runtime["declared_pins_match"]:raise RuntimeError("Pinned Task10 runtime is required")
    destination=writable_path(output);destination.mkdir(exist_ok=False)
    _configure_caches(destination)
    import warp as wp
    torch.cuda.init();torch.cuda.set_device(0);wp.init()
    checked=load_runtime_bundle(bundle)
    report={'schema':'human_wbc_observer_qualification_v1','status':'running',
        'runtime':runtime,'device':allocated_device_metadata('cuda:0'),'bundle_manifest_sha256':sha256_file(checked.root/'manifest.json'),
        'training':False,'acceptance_qualified':False,'fixtures':{}}
    try:
        for kind in ('guard','punch'):
            rows=[];arrays=[]
            report['fixtures'][kind]={'runs':rows,'completed':False}
            for label,on in (('a',False),('a_repeat',False),('b',True)):
                row,trace=_run(checked,destination/(kind+'_'+label),kind=kind,enabled=on)
                rows.append(row);arrays.append(trace)
                if len(arrays)==2:report['fixtures'][kind]['baseline_determinism']=compare_rollouts(arrays[0],arrays[1])
                if row['exception'] is not None or row['policy_steps']!=POLICY_STEPS:
                    raise RuntimeError(f'{kind}/{label} did not complete: {row["exception"]}')
            aa=compare_rollouts(arrays[0],arrays[1]);ab=compare_rollouts(arrays[0],arrays[2])
            instrumented=rows[2]['nonmutation']['passed'] and rows[2]['offline_parity']['passed']
            instrumented &= all(r['native_controller_audit']['status']=='passed' for r in rows)
            passed=aa['passed'] and ab['passed'] and instrumented
            report['fixtures'][kind]={'passed':passed,'completed':True,'instrumentation_passed':instrumented,'baseline_determinism':aa,'observer_comparison':ab,
                'runs':rows,'step_wall_overhead_ratio':rows[2]['step_wall_total_s']/rows[1]['step_wall_total_s'],
                'steady_step_wall_overhead_ratio':rows[2]['steady_step_wall_mean_s']/rows[1]['steady_step_wall_mean_s'],
                'overhead_semantics':'includes qualification copies/CPU trace collection; not production telemetry timing'}
        report['strict_rollout_status']='passed' if all(r['passed'] for r in report['fixtures'].values()) else 'failed'
        report['instrumentation_status']='passed' if all(r['instrumentation_passed'] for r in report['fixtures'].values()) else 'failed'
        report['status']='passed' if report['strict_rollout_status']=='passed' else ('instrumentation_passed_rollouts_nondeterministic' if report['instrumentation_status']=='passed' else 'failed')
    except Exception as exc:
        report['status']='failed';report['exception']={'type':type(exc).__name__,'message':str(exc)}
    write_json(destination/'report.json',report)
    return report


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();report=run_qualification(args.bundle,args.output)
    print(json.dumps({'status':report['status'],'output':args.output}))
    return 0 if report.get('instrumentation_status')=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
