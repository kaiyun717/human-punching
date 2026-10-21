"""Exact initial identity and preserved failed future parity; CPU only."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.util
import json

import numpy as np
import pytest
import torch

from human_punching import phase2_initial_state_v2 as initial
from human_punching import phase2_native_state as native
from human_punching import phase2_shared_evaluation_v2 as shared

spec=importlib.util.spec_from_file_location('preserved_native_cpu_fixtures',Path(__file__).with_name('test_phase2_native_state.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
model=fixtures.model


def fake_env(model,**kwargs):
    env=fixtures.fake_env(model,**kwargs)
    env.sim.wp_data=SimpleNamespace(**{name:torch.zeros(env.n,54) for name in
        ('qacc','qacc_smooth','qfrc_constraint','qfrc_smooth')},actuator_force=torch.zeros(env.n,48),
        solver_niter=torch.zeros(env.n,dtype=torch.int32),nefc=torch.ones(env.n,dtype=torch.int32),
        efc=SimpleNamespace(force=torch.zeros(env.n,1)))
    return env


@pytest.fixture(scope='module')
def measured_witness():
    path=initial.PROJECT_ROOT/'outputs/task10b/p2sd01/run/diagnostic.json'
    return initial.assess_witness(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_actual54_snapshot_witness_keeps_both_results_separate(measured_witness):
    assert measured_witness['initial_state_identity_passed'] is True
    assert measured_witness['future_bitwise_parity_passed'] is False
    assert measured_witness['old_exact_parity_result_preserved'] is False
    assert len(measured_witness['snapshots'])==54 and len(measured_witness['future_state_diagnostics'])==4
    assert measured_witness['solver_cause']=='unknown'
    assert measured_witness['shared_start_feasibility_qualified'] is False
    assert all(not row['canonical410_exact'] for row in measured_witness['future_state_diagnostics'])


@pytest.mark.parametrize('change',[{'diagnostic_complete':False},{'exception':{}},{'explicit_step_calls_returned':11},
    {'unknown_extent_step_calls':1},{'unattempted_explicit_step_calls':1},{'source_sha256':'0'*64},
    {'parent_source_sha256':'0'*64},{'slurm_job_id':None}])
def test_partial_or_unbound_diagnostic_cannot_qualify_initial_identity(monkeypatch,change):
    path=initial.PROJECT_ROOT/'outputs/task10b/p2sd01/run/diagnostic.json'; original=initial.read_bound
    def altered(item):
        p,data=original(item)
        return (p,{**data,**change}) if p==path else (p,data)
    monkeypatch.setattr(initial,'read_bound',altered)
    with pytest.raises(ValueError):initial.assess_witness(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_byte_exact_identity_does_not_hide_one_ulp_or_zero_sign():
    a=np.zeros(3,np.float32);b=a.copy();b[0]=-0.
    assert not initial.exact_arrays(a,b)
    b=a.copy();b[1]=np.nextafter(np.float32(0),np.float32(1))
    assert not initial.exact_arrays(a,b)
    assert not initial.exact_arrays(a,a.astype(np.float64))


def test_exact_restore_and_symmetric_refresh_preserve_full_state_and_history(model):
    source=fake_env(model);saved=fixtures.snapshot(source);destination=fake_env(model,source=False)
    calls=[];destination.sim.forward=lambda:calls.append('forward')
    proof=initial.restore_initial_comparison(destination,saved,trial_id='separate_trial',reference_ids=np.ones(2,dtype=int))
    assert calls==['forward','forward']  # Preserved adapter refresh, then declared common refresh.
    assert proof.before.physical_controller_identity==proof.after.physical_controller_identity==saved.physical_controller_identity
    assert proof.receipt['initial_state_identity_passed'] is True
    assert proof.receipt['future_bitwise_parity_qualified'] is False
    assert destination.physics_steps_executed==source.physics_steps_executed
    assert np.array_equal(destination.last_action.numpy(),source.last_action.numpy())
    with pytest.raises(ValueError,match='once'):
        initial.restore_initial_comparison(destination,saved,trial_id='replayed_trial',reference_ids=np.ones(2,dtype=int))


def test_forward_mutating_any_canonical_value_fails_exact_initial_gate(model):
    source=fake_env(model);saved=fixtures.snapshot(source);destination=fake_env(model,source=False)
    # Preserved fixture intentionally alters warmstart at every forward. v1
    # reapplies it; the newly declared final refresh must still match exactly.
    with pytest.raises(RuntimeError,match='Exact canonical'):
        initial.restore_initial_comparison(destination,saved,trial_id='new_trial',reference_ids=np.zeros(2,dtype=int))


def test_restoration_rejects_extra_controller_state_and_changed_static_drive(model):
    source=fake_env(model);saved=fixtures.snapshot(source)
    destination=fake_env(model,source=False);destination.hidden_history=np.zeros(1)
    with pytest.raises(ValueError,match='Uninventoried'):
        initial.restore_initial_comparison(destination,saved,trial_id='extra',reference_ids=np.zeros(2,dtype=int))
    destination=fake_env(model,source=False);destination.kp[0]+=1
    with pytest.raises(ValueError,match='Static native'):
        initial.restore_initial_comparison(destination,saved,trial_id='drive',reference_ids=np.zeros(2,dtype=int))


def test_derived_difference_is_retained_without_relabeling_future_parity(model):
    source=fake_env(model);saved=fixtures.snapshot(source);destination=fake_env(model,source=False)
    destination.sim.forward=lambda:destination.sim.wp_data.qacc.add_(.1)
    proof=initial.restore_initial_comparison(destination,saved,trial_id='new_trial',reference_ids=np.zeros(2,dtype=int))
    assert proof.receipt['initial_state_identity_passed'] and not proof.receipt['future_bitwise_parity_qualified']
    assert not initial.exact_arrays(proof.derived_before['qacc'],proof.derived_after['qacc'])
    assert proof.before.physical_controller_identity==proof.after.physical_controller_identity


def test_unsupported_derived_inventory_fails_before_mutating_destination(model):
    source=fake_env(model);saved=fixtures.snapshot(source);destination=fake_env(model,source=False)
    destination.sim.wp_data.nefc[0]=1025
    with pytest.raises(ValueError,match='constraint counts'):
        initial.restore_initial_comparison(destination,saved,trial_id='bad',reference_ids=np.zeros(2,dtype=int))
    assert not hasattr(destination,'_phase2_comparison_restore_receipt')


def test_v2_launch_entry_rejects_legacy_manifest_without_touching_native_payloads():
    declaration=SimpleNamespace(identity='identity',start_mode='shared_jab_saved_state')
    with pytest.raises(ValueError,match='Full shared'):
        shared.SharedEvaluationProtocol().preflight(declaration,dict(schema='task10b_phase2_shared_expanded_manifest_v1'),{},
            limits=None,mask_decisions={},release=None,route={})


def test_v2_does_not_fabricate_a_legacy_parity_receipt():
    source=Path(shared.__file__).read_text()
    assert 'parity.get("passed") is not True' not in source
    assert 'manifest["native_initial_identity_witness"]' in source
    assert 'manifest.get("initial_execution_state_identity_version")' in source
    assert 'future_bitwise_parity_required=False' in source
    assert 'native_state_parity=dict(' not in source
    assert 'retained_initial_identity_states=retained' in source


@pytest.mark.parametrize('complete',[True,False])
def test_capture_adds_separate_v2_receipt_preserving_original_result(tmp_path,monkeypatch,complete):
    monkeypatch.setattr(initial,'writable_path',lambda path:Path(path))
    monkeypatch.setattr(initial,'validate_witness',lambda item:dict(version=initial.VERSION,initial_state_identity_passed=True,
        future_bitwise_parity_passed=False,identity='witness'))
    monkeypatch.setattr(initial,'binding',lambda path:dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    seen=[]
    def capture(*args,**kwargs):
        seen.append((args,kwargs));folder=Path(args[5]);folder.mkdir()
        result=dict(schema='task10b_phase2_native_shared_bank_capture_v1',source_sha256=initial.PARENT_SHA,
            complete_capture=complete,exception=None if complete else {'type':'Failure'},shared_start_feasibility_qualified=False)
        (folder/'bank_capture.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(native,'capture_shared_start_bank',capture)
    result=initial.capture_bank('catalog','domain','train','geometry','release',tmp_path/'capture',270,witness={})
    original=json.loads((tmp_path/'capture/bank_capture.json').read_text())
    assert len(seen)==1 and original['schema']=='task10b_phase2_native_shared_bank_capture_v1'
    assert result['schema']==initial.CAPTURE and result['complete_capture']==complete
    assert result['initial_identity_witness']['future_bitwise_parity_passed'] is False
    assert result['shared_start_feasibility_qualified'] is False
    assert (tmp_path/'capture/bank_capture_initial_identity_v2.json').exists()
