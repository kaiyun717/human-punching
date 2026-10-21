"""Future entry routing only; mock saved receipts, no native training."""
import importlib.util,json
from pathlib import Path
import pytest
from human_punching import phase2_grouped_clock_entry_v2 as entry
from human_punching import phase2_grouped_entry_v3 as integrity
from human_punching import phase2_expanded_clocks_v1 as clocks


def driver():
    path=Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_grouped_clocks_v2.py'
    spec=importlib.util.spec_from_file_location('grouped_clock_v2_test_driver',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_only_reviewed_integrity_implementation_and_receipt_version_change():
    root=Path(__file__).resolve().parents[1]
    old=(root/'src/human_punching/phase2_grouped_clock_entry_v1.py').read_text()
    expected=old.replace('phase2_grouped_entry_v2 as parent','phase2_grouped_entry_v3 as parent').replace('task10b_phase2_grouped_clock_entry_v1','task10b_phase2_grouped_clock_entry_v2').replace('grouped_integrity_v2','grouped_integrity_v3').replace('grouped_event_clocks_v1.json','grouped_event_clocks_v2.json')
    assert Path(entry.__file__).read_text()==expected and entry.parent is integrity and entry.VERSION==clocks.VERSION


@pytest.mark.parametrize('parity,clocks_pass,expected',[(True,True,True),(False,True,False),(True,False,False)])
def test_tuple_saved_receipt_passes_only_with_exact_parity_and_clock_audit(tmp_path,monkeypatch,parity,clocks_pass,expected):
    config=tmp_path/'configuration.json';config.write_text('{}');out=tmp_path/'run';written=[]
    def original(path,digest,folder):
        (folder/'training').mkdir(parents=True);p=folder/'training/checkpoint_final.pt';p.write_bytes(b'mock checkpoint never loaded')
        batch=folder/'held_out_validation/validation_00000/batch_0000';batch.mkdir(parents=True)
        (batch.parent/'summary.json').write_text(json.dumps({'status':'executed','trials':[{'event_clock_version':clocks.VERSION}]}))
        result={'status':'bounded_stop','grouped_schema':'task10b_phase2_grouped_training_receipt_v1',
            'target_candidate_identities':((7,'a'*64),),'checkpoint':integrity._binding(p),
            'reload_parity':{'parameters_equal':parity,'actions_equal':parity,'maximum_action_error':0.}}
        data=json.dumps(result).encode();(folder/'grouped_training_receipt.json').write_bytes(data);written.append(data)
        return result
    monkeypatch.setattr(integrity,'_original_execute_configuration',original)
    monkeypatch.setattr(entry,'audit_batch',lambda p:{'passed':clocks_pass})
    result=entry.execute_configuration(config,integrity._sha(config),out)
    assert result['status']==('bounded_stop' if expected else 'failed')
    assert result['grouped_integrity_pass'] is parity and result['event_clocks_pass'] is clocks_pass
    assert (out/'grouped_training_receipt.json').read_bytes()==written[0] and len(written)==1
    receipt=json.loads((out/'grouped_event_clocks_v2.json').read_text())
    assert receipt['inherited_integrity']==result['grouped_integrity_v3']
    assert receipt['physics_actor_reference_changes'] is False and receipt['qualification'] is False


def test_validation_hook_restores_if_original_entry_raises(monkeypatch,tmp_path):
    from human_punching import phase2_validation_callback as callback
    saved=callback.evaluate_policy
    def raises(*a):
        assert callback.evaluate_policy is clocks.evaluate_policy
        raise RuntimeError('original failure')
    monkeypatch.setattr(entry.parent,'execute_configuration',raises)
    with pytest.raises(RuntimeError):entry.execute_configuration('unused','bound',tmp_path/'output')
    assert callback.evaluate_policy is saved


def test_transport_changes_only_native_entry_and_closes_v3_source(monkeypatch):
    d=driver();old=['python','-B','-m','human_punching.phase2_grouped_entry_v2','--configuration','source','--output','fresh']
    monkeypatch.setattr(d,'_native',lambda *a:list(old))
    new=d.native_command({'mode':'grouped-pilot'},1600)
    assert new[:3]==old[:3] and new[4:]==old[4:] and new[3]=='human_punching.phase2_grouped_clock_entry_v2'
    monkeypatch.setattr(d,'_inputs',lambda *a,**k:{})
    closure=d.required_inputs({'mode':'grouped-pilot'})
    assert 'src/human_punching/phase2_grouped_entry_v3.py' in closure
    assert 'src/human_punching/phase2_expanded_clocks_v1.py' in closure


@pytest.mark.parametrize('mode,wall,app',[('grouped-finetune',1800,1680),('grouped-pilot',1801,1680),('grouped-pilot',1800,1700)])
def test_no_implicit_scope_or_budget_expansion(mode,wall,app):
    with pytest.raises(ValueError):driver().validate_spec(mode,{},wall,app)


def test_real_configuration_metadata_stays_original_finite_pilot():
    d=driver();path=d.ROOT/'outputs/task10b/p2jgconfig01/configuration.json'
    result=d.validate_spec('grouped-pilot',{'configuration':str(path)},1800,1680)
    assert result['fresh_transition_cap']==262144
    assert result['expanded_event_clock_version']==clocks.VERSION
    assert result['grouped_integrity_gate_version']==integrity.SCHEMA
