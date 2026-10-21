"""Prefix-aware metadata audit; a retained prefix receives no cycle credit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import phase2_expanded_clock_audit_v1 as parent

VERSION='task10b_phase2_independent_direct_clock_audit_v2'


def audit_batch(folder):
    folder=Path(folder)
    result=parent.audit_batch(folder)
    plan=json.loads((folder/'plan.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    checks=dict(result['checks']);original_complete=checks.pop('complete_raw')
    planned=plan['planned_physics_steps'];actual=summary.get('actual_physics_steps')
    with np.load(folder/'raw_trace.npz',allow_pickle=False) as raw:
        retained=len(raw['qpos'])-1;clock=raw['time']
        checks['retained_prefix_counter']=(type(planned) is int and planned>0 and
            type(actual) is int and 0<=actual==retained<=planned)
        checks['native_clock_shape']=bool(clock.dtype in (np.dtype('float32'),np.dtype('float64'))
            and clock.shape==(retained+1,len(plan['trials'])) and np.isfinite(clock).all())
        if checks['native_clock_shape']:
            # Compare the actual native per-interval accumulation, preserving
            # its dtype. This avoids inventing tolerance for float32 drift.
            expected=np.zeros_like(clock)
            for index in range(1,len(expected)):
                expected[index]=expected[index-1]+np.asarray(.005,dtype=clock.dtype)
            checks['exact_native_5ms_clock']=np.array_equal(clock,expected)
        else:checks['exact_native_5ms_clock']=False
    return {**result,'schema':VERSION,'passed':all(checks.values()),'checks':checks,
        'planned_physics_steps':planned,'retained_physics_steps':retained,
        'complete_raw_diagnostic':original_complete,'prefix_only':retained<planned,
        'completed_cycle_credit':False,'qualification':False,
        'limits':'Metadata consistency only, including resolved prefixes. The original independent physical/actor audit must determine attempt resolution and all physical metrics; this receipt grants no cycle or qualification credit.'}
