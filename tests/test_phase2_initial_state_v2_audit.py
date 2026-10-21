"""Independent v2 initial-identity audit and fixed callback dispatch tests."""
from dataclasses import replace
from pathlib import Path
import ast
import importlib.util
import json

import numpy as np
import pytest

from human_punching import phase2_initial_state_v2 as initial
from human_punching import phase2_initial_capture_v2 as capture
from human_punching import phase2_native_bank_geometry_v2 as geometry_capture
from human_punching.phase2_native_state import load_snapshot,save_snapshot,NativeComparisonSnapshot

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


audit=module(ROOT/'scripts/task10b_phase2_audit_initial_v2.py','initial_identity_audit_v2')
old_tests=module(Path(__file__).with_name('test_phase2_audit.py'),'preserved_independent_audit_fixtures')


def fixture(tmp_path,monkeypatch):
    evidence=list(old_tests.shared_initialization_fixture(tmp_path));raw,receipt,plan,declaration,metadata,_=evidence
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=audit.digest(path))
    bank=load_snapshot(metadata['native_start_state_bank']['path'])
    witness=dict(path='synthetic-no-actual-witness',sha256='a'*64,identity='b'*64,
        version=initial.VERSION,initial_state_identity_passed=True,future_bitwise_parity_passed=False,solver_cause='unknown')
    monkeypatch.setattr(initial,'validate_witness',lambda item:witness)
    def allocation(item,**kwargs):
        if item!={'synthetic':True}:raise ValueError('Changed synthetic allocation')
        return dict(synthetic=True,no_actual_job=True)
    monkeypatch.setattr(capture,'validate_capture_allocation',allocation)
    base=json.loads(Path(metadata['native_start_state_capture']['path']).read_text())
    base.update(schema=geometry_capture.VERSION,source_sha256=audit.digest(Path(geometry_capture.__file__)))
    base_bound=save('synthetic_base_capture_v2.json',base)
    wrapped={**base,'schema':initial.CAPTURE,'source_sha256':audit.digest(Path(capture.__file__)),
        'base_capture':base_bound,'base_capture_source_sha256':audit.digest(Path(geometry_capture.__file__)),
        'parent_native_adapter_sha256':initial.PARENT_SHA,'initial_identity_consumer_sha256':audit.digest(Path(initial.__file__)),
        'initial_identity_version':initial.VERSION,'initial_identity_witness':witness}
    metadata['native_start_state_capture']=save('synthetic_v2_capture.json',wrapped)
    metadata.pop('native_state_parity');metadata.update(native_initial_identity_witness=witness,native_capture_allocation={'synthetic':True},
        initial_execution_state_identity_version=initial.VERSION,future_bitwise_parity_required=False,
        future_solver_drift_diagnostics_mandatory=True)
    receipt.update(schema='task10b_shared_native_initialization_audit_v2',initial_identity_version=initial.VERSION,
        initial_identity_witness=witness,future_bitwise_parity_qualified=False,solver_cause='unknown',derived_differences_are_diagnostic=True)
    receipt['restore_receipt'].update(initial_identity_version=initial.VERSION,initial_state_identity_passed=True,
        exact_after_declared_forward_refresh=True,declared_initial_forward_refreshes=1,no_within_cycle_restore=True,
        future_bitwise_parity_qualified=False)
    selected=NativeComparisonSnapshot(tuple(replace(bank.states[b['start_index']],reference_id=b['reference_sha256'])
        for b in declaration['bindings']),bank.inventory_json)
    retained={}
    for label in ('before_refresh','after_refresh'):
        path=save_snapshot(selected,tmp_path/label);retained[label]=dict(path=str(path),sha256=audit.digest(path),identity=selected.identity)
    receipt['retained_initial_identity_states']=retained
    payload={side+'__'+key:np.zeros((2,54),np.float32) for side in ('before','after') for key in ('qacc','qacc_smooth','qfrc_constraint','qfrc_smooth')}
    for side in ('before','after'):
        payload[side+'__actuator_force']=np.zeros((2,48),np.float32)
        payload[side+'__solver_niter']=np.zeros(2,np.int32);payload[side+'__nefc']=np.ones(2,np.int32)
        payload[side+'__constraint_force']=np.zeros((2,1),np.float32)
    payload['after__qacc']+=.1  # Explicitly diagnosed output change, never initial-state drift.
    path=tmp_path/'derived.npz';np.savez(path,**payload);receipt['derived_refresh_evidence']=dict(path=str(path),sha256=audit.digest(path))
    receipt['derived_refresh_differences']={key:dict(same_shape=True,exact_equal=np.array_equal(payload['before__'+key],payload['after__'+key]),
        maximum_absolute_difference=float(np.max(np.abs(payload['before__'+key].astype(float)-payload['after__'+key].astype(float)),initial=0)))
        for key in ('qacc','qacc_smooth','qfrc_constraint','qfrc_smooth','actuator_force','solver_niter','nefc','constraint_force')}
    return evidence


def test_independent_initial_audit_retains_derived_difference_without_future_credit(tmp_path,monkeypatch):
    result=audit.shared_initialization_audit(*fixture(tmp_path,monkeypatch))
    assert result['passed'],result
    assert result['full410_initial_state_audited'] and not result['future_bitwise_parity_qualified']
    assert result['derived_refresh_differences']['qacc']['exact_equal'] is False
    assert result['solver_cause']=='unknown'


@pytest.mark.parametrize('tamper',['canonical','history','static','model','derived','allocation','legacy_schema','future_credit','missing_refresh'])
def test_independent_v2_initial_audit_rejects_tampering(tmp_path,monkeypatch,tamper):
    evidence=fixture(tmp_path,monkeypatch);raw,receipt,plan,declaration,metadata,_=evidence
    if tamper in ('canonical','history','static','model'):
        record=receipt['retained_initial_identity_states']['after_refresh'];saved=load_snapshot(record['path']);state=saved.states[0]
        if tamper=='canonical':
            values=state.native_integration_state.copy();values[-1]+=.25;state=replace(state,native_integration_state=values)
        elif tamper=='history':state=replace(state,held_raw_action=np.ones(48))
        elif tamper=='model':
            saved=NativeComparisonSnapshot(tuple(replace(s,model_sha256='f'*64) for s in saved.states),saved.inventory_json)
        else:
            inventory=json.loads(saved.inventory_json);inventory['static_controller_sha256']='f'*64
            saved=NativeComparisonSnapshot(saved.states,json.dumps(inventory))
        if tamper not in ('static','model'):saved=NativeComparisonSnapshot((state,*saved.states[1:]),saved.inventory_json)
        path=save_snapshot(saved,tmp_path/'tampered');record.update(path=str(path),sha256=audit.digest(path),identity=saved.identity)
    elif tamper=='derived':receipt['derived_refresh_differences']['qacc']['exact_equal']=True
    elif tamper=='allocation':metadata['native_capture_allocation']={'synthetic':False}
    elif tamper=='legacy_schema':receipt['schema']='task10b_shared_native_initialization_audit_v1'
    elif tamper=='future_credit':receipt['future_bitwise_parity_qualified']=True
    else:receipt['restore_receipt']['declared_initial_forward_refreshes']=0
    assert not audit.shared_initialization_audit(*evidence)['passed']


def test_unchanged_d0_algorithms_and_clock_have_static_ast_parity():
    old=ast.parse((ROOT/'scripts/task10b_phase2_audit_clock_v2.py').read_text())
    new=ast.parse((ROOT/'scripts/task10b_phase2_audit_initial_v2.py').read_text())
    functions=lambda tree:{item.name:ast.dump(item) for item in tree.body if isinstance(item,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    for name in a:
        if name not in ('shared_initialization_audit','closed_launch_audit','audit_run','main'):assert a[name]==b[name],name
    assert audit.initial_identity_version_receipt()['legacy_future_bitwise_failures_reclassified'] is False
    assert audit.clock_version_receipt()['version']=='task10b_phase2_independent_clock_v2'


def test_callback_and_learning_dispatch_change_only_versioned_consumer_and_configuration():
    for parent,new,changes in [('phase2_shared_validation_callback.py','phase2_shared_validation_callback_v2.py',
        [('from .phase2_shared_evaluation import','from .phase2_shared_evaluation_v2 import'),('shared_validation_configuration_v1','shared_validation_configuration_v2')]),
        ('phase2_shared_learning.py','phase2_shared_learning_v2.py',
        [('from .phase2_shared_validation_callback import','from .phase2_shared_validation_callback_v2 import'),
         ('from .phase2_shared_catalog import','from .phase2_shared_catalog_v2 import'),
         ('shared_training_configuration_v1','shared_training_configuration_v2')])]:
        text=(ROOT/'src/human_punching'/parent).read_text()
        for old,newtext in changes:text=text.replace(old,newtext)
        assert text==(ROOT/'src/human_punching'/new).read_text()


def test_test_capture_freeze_gate_precedes_physical_payload_hashes():
    source=Path(capture.__file__).read_text();function=source[source.index('def check_packet'):source.index('def validate_capture_allocation')]
    assert function.index('validate_test_bank_freeze')<function.index("for name,digest in packet['files'].items()")
    assert "packet['capture_split']=='test'" in function
