import pytest
import torch

from human_punching.phase2_sampling import (IntegerReferenceSampler,
    IntegerReferencePlayback,future_offset_ticks)


def arrays(n=64):
    t=torch.arange(n,dtype=torch.float32)*.005
    q=torch.zeros(n,55);q[:,3]=1.;q[:,7]=t**3
    return dict(t=t,qpos=q,qvel=torch.stack((3*t*t,torch.arange(n)),dim=1),
        torque=torch.arange(n*48,dtype=torch.float32).reshape(n,48),
        head_center=torch.tensor([1.,2.,3.]),guard_glove_chest_pos=torch.ones(2,3),
        initial_root_height=torch.tensor(1.))


def test_exact_derivative_knots_and_held_terminal_without_source_mutation():
    row=arrays();before={k:v.clone() for k,v in row.items()}
    sample=IntegerReferenceSampler([row])
    ids=torch.zeros(5,dtype=torch.long);ticks=torch.tensor([0,1,20,50,100])
    assert torch.equal(sample.sample("qvel",ids,ticks),row["qvel"][[0,1,20,50,63]])
    assert torch.equal(sample.sample("qpos",ids,ticks),row["qpos"][[0,1,20,50,63]])
    assert torch.equal(sample.sample("head_center",ids,ticks),row["head_center"].expand(5,3))
    assert all(torch.equal(row[k],v) for k,v in before.items())


def test_exact_reference_identity_does_not_pick_nearest_row():
    first,second=arrays(),arrays();second["torque"]+=10000.
    sample=IntegerReferenceSampler([first,second]);ids=torch.tensor([1,0]);ticks=torch.tensor([20,20])
    assert torch.equal(sample.sample("torque",ids,ticks),torch.stack((second["torque"][20],first["torque"][20])))
    with pytest.raises(ValueError):sample.sample("torque",torch.tensor([2]),torch.tensor([20]))


@pytest.mark.parametrize("ticks",[torch.tensor([1.]),torch.tensor([-1]),torch.tensor([[1]])])
def test_noninteger_or_negative_clock_rejected(ticks):
    with pytest.raises(ValueError):IntegerReferenceSampler([arrays()]).sample("qvel",torch.tensor([0]),ticks)


def test_nonmaster_grid_rejected():
    row=arrays();row["t"][5]+=.001
    with pytest.raises(ValueError):IntegerReferenceSampler([row])


def test_future_offsets_are_original_exact_ticks():
    assert [future_offset_ticks(x) for x in (0.,.1,.25)]==[0,20,50]
    for x in (.105,-.1,True):
        with pytest.raises(ValueError):future_offset_ticks(x)


def test_native_mixin_uses_integer_clock_and_suppresses_rejected_future():
    class Base:
        def ref(self,*args):return "initialization_only"
    class Env(IntegerReferencePlayback,Base):pass
    env=Env();assert env.ref("qvel")=="initialization_only"
    row=arrays();env.library=type("Library",(),dict(arrays=[row]))()
    env.ticks=torch.tensor([1,1]);env.ids_ref=torch.tensor([0,0]);env.rejected=torch.tensor([False,True])
    env.install_integer_reference_playback()
    assert torch.equal(env.ref("qvel",.25),row["qvel"][[51,1]])
