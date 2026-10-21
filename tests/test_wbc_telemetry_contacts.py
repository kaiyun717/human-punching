import inspect
import numpy as np
import pytest
import torch
from human_punching.wbc.telemetry.contacts import ContactBindings, ContactReducer


def reducer():
    return ContactReducer(ContactBindings((1,2),(0,),(1,2,3),(4,),((2,3),),5),2,6,'cpu',torch.float64)


def inputs():
    return dict(count=torch.tensor([6]),worldid=torch.tensor([0,0,0,1,1,1]),
        geom=torch.tensor([[0,1],[2,0],[3,0],[1,4],[2,3],[0,2]]),
        pos=torch.zeros(6,3,dtype=torch.float64),distance=torch.full((6,),-.001,dtype=torch.float64),
        force=torch.tensor([[2.,0,0,0,0,0]]*6,dtype=torch.float64),
        overflow=torch.zeros(2,dtype=torch.int64),anchors=torch.zeros(2,2,3,dtype=torch.float64))


def test_floor_human_self_and_g1_distinct():
    a=inputs();r=reducer().reduce(**a)
    assert torch.equal(r['support_contact'],torch.tensor([[True,True],[False,True]]))
    assert torch.equal(r['support_patch_contact'],r['support_contact'])
    assert r['forbidden_collision'].tolist()==[True,False]
    assert r['human_g1_collision'].tolist()==[False,True]
    assert not r['numerical_failure'].any() and not r['contact_overflow'].any()
    assert r['foot_normal_force_n'].tolist()==[[2.,2.],[0.,2.]]


def test_stale_slots_force_clearance_and_patch_at_interval_start():
    a=inputs();a['count'][:]=2;a['force'][0,0]=.999;a['pos'][1,0]=.06501
    r=reducer().reduce(**a)
    assert r['support_contact'].tolist()==[[False,True],[False,False]]
    assert not r['support_patch_contact'].any()
    assert not r['forbidden_collision'].any() and not r['human_g1_collision'].any()
    a['force'][0,0]=1.;a['pos'][1,0]=.065
    r=reducer().reduce(**a)
    assert r['support_patch_contact'].tolist()==[[True,True],[False,False]]


def test_inactive_nan_ignored_but_active_nan_and_bad_ids_fail_closed():
    a=inputs();a['count'][:]=2;a['force'][4]=float('nan');a['worldid'][5]=999
    r=reducer().reduce(**a);assert not r['numerical_failure'].any()
    a['force'][0,0]=float('nan')
    r=reducer().reduce(**a);assert r['numerical_failure'].tolist()==[True,False]
    a['worldid'][1]=-1
    r=reducer().reduce(**a);assert r['numerical_failure'].all()


@pytest.mark.parametrize('count',[-1,7])
def test_global_capacity_fault(count):
    a=inputs();a['count'][:]=count;r=reducer().reduce(**a)
    assert r['numerical_failure'].all() and r['contact_overflow'].all()


def test_world_overflow_and_unloaded_margin_contact():
    a=inputs();a['count'][:]=3;a['distance'][:]=.01;a['force'].zero_();a['overflow'][1]=2
    r=reducer().reduce(**a)
    assert r['contact_overflow'].tolist()==[False,True]
    assert not r['forbidden_collision'].any() and not r['support_contact'].any()


def test_no_hot_path_host_reads():
    source=inspect.getsource(ContactReducer.reduce)
    for token in ('.item(','.numpy(','.cpu(','.tolist('):assert token not in source


def test_decode_mask_blocks_invalid_addresses_and_overflow_before_native_reads():
    from human_punching.wbc.telemetry.contacts import decode_mask
    world=torch.tensor([0,0,1,1,1,99]);count=torch.tensor([6]);dim=torch.tensor([1,3,3,3,3,3])
    adr=torch.full((6,10),-1,dtype=torch.int32);adr[:,0]=torch.tensor([8,4,0,-1,-2,0])
    overflow=torch.tensor([0,0]);ids=torch.arange(6,dtype=torch.int32)
    safe,fault=decode_mask(ids,count,world,dim,adr,overflow,8,True)
    assert not safe.any()  # Unassignable world ID invalidates this decode batch.
    assert fault.all()  # World99 is unassignable; fail the whole batch.
    world[-1]=1;adr[-1,0]=-1
    safe,fault=decode_mask(ids,count,world,dim,adr,overflow,8,True)
    assert fault.tolist()==[True,True]
    adr[0,0]=7;adr[4,0]=-1;overflow[1]=2
    safe,fault=decode_mask(ids,count,world,dim,adr,overflow,8,True)
    assert safe.tolist()==[True,True,False,False,False,False]
    assert not fault.any()  # Separate overflow flag carries the explicit failure.
    overflow.zero_();adr[:,:3]=torch.tensor([0,1,2]);adr[1,2]=-1
    safe,fault=decode_mask(ids,count,world,dim,adr,overflow,8,False)
    assert not safe[1] and fault[0]
