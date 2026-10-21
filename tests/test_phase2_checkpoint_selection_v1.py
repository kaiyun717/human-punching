from pathlib import Path
import hashlib
import torch
import pytest
from human_punching import phase2_checkpoint_selection_v1 as s

def save(path,weight,**metadata):
    state={'actor.0.weight':torch.tensor([[weight]],dtype=torch.float32),'log_std':torch.tensor([-.5])}
    torch.save(dict(policy=state,**metadata),path)
    return state,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def test_equivalent_serializations_choose_exact_declared_file_even_when_accepted_first(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    state,a=save(tmp_path/'accepted.pt',1.,optimizer={'state':{},'param_groups':[]})
    _,b=save(tmp_path/'evaluated.pt',1.,physical_validation_only=True)
    assert a['sha256']!=b['sha256']
    path,actual=s.matching_evaluated_checkpoint([a['path'],b['path']],s.policy_digest(state),b)
    assert path==Path(b['path']) and all(torch.equal(actual[k],state[k]) for k in state)

@pytest.mark.parametrize('defect',['weight','bytes','inventory','outside'])
def test_exact_binding_never_repaired_by_another_matching_checkpoint(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    state,a=save(tmp_path/'accepted.pt',1.)
    _,b=save(tmp_path/'evaluated.pt',2. if defect=='weight' else 1.)
    paths=[a['path'],b['path']]
    if defect=='bytes':b['sha256']=a['sha256']
    if defect=='inventory':paths=[a['path']]
    if defect=='outside':monkeypatch.setattr(s,'ROOT',tmp_path/'other');s.ROOT.mkdir()
    with pytest.raises(ValueError):s.matching_evaluated_checkpoint(paths,s.policy_digest(state),b)
