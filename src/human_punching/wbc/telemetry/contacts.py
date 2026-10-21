"""DRAFT GPU contact telemetry from existing native forces, never a new solve.

Classification matches cycle_trace.classify_contacts at interval-start anchors.
Public MuJoCo-Warp contact_force handles pyramidal or elliptic decoding. Outputs
are separate buffers; live contacts, constraints and force arrays are read only.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from .streams import current_native_stream


@dataclass(frozen=True)
class ContactBindings:
    foot_ids: tuple[int, int]
    floor_ids: tuple[int, ...]
    human_ids: tuple[int, ...]
    g1_ids: tuple[int, ...]
    allowed_pairs: tuple[tuple[int, int], ...]
    ngeom: int
    patch_radius_m: float = .065
    minimum_normal_force_n: float = 1.


def resolve_contacts(model, bindings):
    def geom(name):
        try: return int(model.geom(name).id)
        except (KeyError, ValueError) as exc: raise ValueError(f'Missing contact geom {name}') from exc
    result = ContactBindings(tuple(geom(bindings.prefix+n) for n in bindings.foot_geoms),
        tuple(geom(n) for n in bindings.floor_geoms),
        tuple(i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(bindings.human_geom_prefix)),
        tuple(geom(n) for n in bindings.g1_geom_names),
        tuple((geom(a),geom(b)) for a,b in bindings.allowed_self_pairs),
        int(model.ngeom), bindings.patch_radius_m, bindings.minimum_normal_force_n)
    if (len(set(result.foot_ids)) != 2 or not result.floor_ids
            or not set(result.foot_ids) <= set(result.human_ids)
            or set(result.human_ids) & (set(result.g1_ids)|set(result.floor_ids))):
        raise ValueError('Two human feet and disjoint human/G1/floor geom sets required')
    return result


class ContactReducer:
    """Capacity-shaped tensor reducer; inactive slots never contribute stale data."""
    def __init__(self, bindings, worlds, capacity, device, dtype=torch.float32):
        if worlds < 1 or capacity < 1:
            raise ValueError('Positive world and contact capacities required')
        self.bindings, self.n, self.capacity = bindings, worlds, capacity
        self.device, self.dtype = torch.device(device), dtype
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.indices = torch.arange(capacity, device=device)
        self.human = torch.zeros(bindings.ngeom, dtype=torch.bool, device=device)
        self.g1 = torch.zeros_like(self.human); self.floor = torch.zeros_like(self.human)
        self.human[list(bindings.human_ids)] = True
        self.g1[list(bindings.g1_ids)] = True
        self.floor[list(bindings.floor_ids)] = True
        self.allowed = torch.zeros((bindings.ngeom, bindings.ngeom), dtype=torch.bool, device=device)
        for a,b in bindings.allowed_pairs:
            self.allowed[a,b] = self.allowed[b,a] = True
        self.result = {k:torch.zeros((worlds,2),device=device,dtype=torch.bool) for k in ('support_contact','support_patch_contact')}
        self.result.update({k:torch.zeros(worlds,device=device,dtype=torch.bool) for k in ('forbidden_collision','human_g1_collision','contact_overflow','numerical_failure')})
        self.result['foot_normal_force_n'] = torch.zeros((worlds,2),device=device,dtype=dtype)

    def reduce(self, *, count, worldid, geom, pos, distance, force, overflow, anchors):
        c,n,b = self.capacity,self.n,self.bindings
        expected = {'worldid':(c,), 'geom':(c,2), 'pos':(c,3), 'distance':(c,),
                    'force':(c,6), 'overflow':(n,), 'anchors':(n,2,3), 'count':(1,)}
        values = dict(count=count,worldid=worldid,geom=geom,pos=pos,distance=distance,force=force,overflow=overflow,anchors=anchors)
        for name,value in values.items():
            if value.shape != expected[name] or value.device != self.device:
                raise ValueError(f'Malformed contact {name}')
            if name in ('pos','distance','force','anchors'):
                if value.dtype != self.dtype: raise ValueError(f'Contact float dtype mismatch: {name}')
            elif value.dtype not in (torch.int32,torch.int64): raise ValueError(f'Contact integer dtype required: {name}')
        present = self.indices < count[0]
        identities = (worldid>=0)&(worldid<n)&(geom>=0).all(-1)&(geom<b.ngeom).all(-1)
        finite = torch.isfinite(pos).all(-1)&torch.isfinite(distance)&torch.isfinite(force).all(-1)
        global_fault = ((count<0)|(count>c)).any() | (present & ~identities).any()
        valid = present & identities & finite
        wid = worldid.long().clamp(0,n-1)
        pair = geom.long().clamp(0,b.ngeom-1)
        a,z = pair[:,0],pair[:,1]
        human = self.human[a]|self.human[z]
        active = valid & human & ((distance<=0)|(force[:,0]>0))
        g1 = active & (self.g1[a]|self.g1[z])
        floor = self.floor[a]|self.floor[z]
        foot_ground = torch.zeros_like(active)
        finite_anchors = torch.isfinite(anchors).flatten(1).all(-1)
        def any_by_world(mask):
            total = torch.zeros(n,dtype=torch.int32,device=self.device)
            total.scatter_add_(0,wid,mask.to(torch.int32))
            return total > 0
        for side,foot in enumerate(b.foot_ids):
            matches = active & ~g1 & floor & ((a==foot)|(z==foot))
            foot_ground |= matches
            loaded = matches & (force[:,0]>=b.minimum_normal_force_n)
            patch = loaded & (torch.linalg.vector_norm(pos[:,:2]-anchors[wid,side,:2],dim=-1)<=b.patch_radius_m)
            self.result['support_contact'][:,side].copy_(any_by_world(loaded))
            self.result['support_patch_contact'][:,side].copy_(any_by_world(patch))
            load = torch.zeros(n,device=self.device,dtype=self.dtype)
            load.scatter_add_(0,wid,torch.where(matches,torch.nan_to_num(force[:,0]).clamp_min(0),0.))
            self.result['foot_normal_force_n'][:,side].copy_(load)
        forbidden = active & ~g1 & ~foot_ground & ~self.allowed[a,z]
        self.result['forbidden_collision'].copy_(any_by_world(forbidden))
        self.result['human_g1_collision'].copy_(any_by_world(g1))
        self.result['contact_overflow'].copy_((overflow!=0)|((count<0)|(count>c)).any())
        self.result['numerical_failure'].copy_(global_fault | ~finite_anchors | any_by_world(present&identities&~finite))
        return self.result


def decode_mask(ids,count,worldid,dimension,address,overflow,njmax,pyramidal):
    """Reject unsafe public-decoder reads before invoking its CUDA kernel.

    No-constraint address -1 is ordinary unloaded contact. Malformed indices and
    dimensions are numerical faults. Overflow worlds are skipped and remain
    explicitly invalid via contact_overflow, never treated as measured no-load.
    """
    worlds=overflow.numel();present=ids<count[0]
    owned=(worldid>=0)&(worldid<worlds);wid=worldid.long().clamp(0,worlds-1)
    dim_ok=(dimension==1)|(dimension==3)|(dimension==4)|(dimension==6)
    constrained=address[:,0]>=0
    if pyramidal:
        width=torch.where(dimension==1,1,2*(dimension-1))
        adr_ok=(address[:,0]>=0)&(address[:,0]+width<=njmax)
    else:
        columns=torch.arange(address.shape[1],device=address.device)
        used=columns[None,:]<dimension[:,None]
        adr_ok=(~used|((address>=0)&(address<njmax))).all(-1)
    bad=present & ((~owned)|(~dim_ok)|(address[:,0]<-1)|(constrained&~adr_ok))
    counts=torch.zeros(worlds,dtype=torch.int32,device=ids.device)
    counts.scatter_add_(0,wid,(bad&owned).to(torch.int32))
    global_fault=((count<0)|(count>ids.numel())).any()|(bad&~owned).any()
    fault=(counts>0)|global_fault
    safe=present&owned&dim_ok&constrained&adr_ok&(overflow[wid]==0)&~global_fault
    return safe,fault


class NativeContactTelemetry:
    """Preallocated public force decoding; no live data mutation or host reads."""
    def __init__(self, cpu_model, wp_model, wp_data, bindings, *, torch_device):
        import warp as wp
        self.model,self.data = wp_model,wp_data
        self.pyramidal = int(cpu_model.opt.cone) == 0
        self.dimension = wp.to_torch(wp_data.contact.dim)
        self.address = wp.to_torch(wp_data.contact.efc_address)
        self.capacity = wp_data.contact.geom.shape[0]
        self.reducer = ContactReducer(resolve_contacts(cpu_model,bindings),wp_data.nworld,
                                      self.capacity,torch_device)
        self.forces = wp.zeros(self.capacity,dtype=wp.spatial_vector,device=wp_data.qpos.device)
        self.force_tensor = wp.to_torch(self.forces)
        self.ids = torch.arange(self.capacity,dtype=torch.int32,device=torch_device)
        self.safe_ids = torch.empty_like(self.ids)
        self.wp_ids = wp.from_torch(self.safe_ids,dtype=wp.int32)
        self.arrays = {name:wp.to_torch(getattr(wp_data.contact,name)) for name in ('worldid','geom','pos','dist')}
        self.count,self.overflow = wp.to_torch(wp_data.nacon),wp.to_torch(wp_data.overflow)

    def sample(self, interval_start_anchors):
        import mujoco_warp as mjwarp
        # contact_force leaves inactive slots untouched. Clear our own output;
        # invalid world IDs skip native reads and are independently faulted.
        self.force_tensor.zero_()
        valid,decode_fault = decode_mask(self.ids,self.count,self.arrays['worldid'],
            self.dimension,self.address,self.overflow,self.data.njmax,self.pyramidal)
        self.safe_ids.copy_(torch.where(valid,self.ids,torch.iinfo(torch.int32).max))
        import warp as wp
        stream = current_native_stream(self.reducer.device)
        with wp.ScopedStream(stream):
            mjwarp.contact_force(self.model,self.data,self.wp_ids,False,self.forces)
        result = self.reducer.reduce(count=self.count,worldid=self.arrays['worldid'],
            geom=self.arrays['geom'],pos=self.arrays['pos'],distance=self.arrays['dist'],
            force=self.force_tensor,overflow=self.overflow,anchors=interval_start_anchors)
        result["numerical_failure"] |= decode_fault
        return result
