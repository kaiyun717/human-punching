"""Exact 5ms knot playback for new derivative-consistent reference recipes.

Future actor features use +20/+50 ticks. This samples stored analytic/numerical
knot derivatives without introducing a second floating-point interpolation
clock. It makes no claim about a continuously interpolated joint trajectory.
"""
from __future__ import annotations

import torch

VERSION = "task10b_integer_5ms_reference_knots_v1"
CONSTANT_KEYS = frozenset(("head_center","guard_glove_chest_pos"))


def future_offset_ticks(seconds):
    if isinstance(seconds,bool) or seconds not in (0.,.1,.25):
        raise ValueError("Original actor future offsets are exactly0/.1/.25seconds")
    return {0.:0,.1:20,.25:50}[seconds]


class IntegerReferenceSampler:
    """Borrow a bounded library's tensors; never tile or mutate source arrays."""
    def __init__(self,references):
        if not references:raise ValueError("At least one exact reference required")
        self.references=tuple(references)
        for row in self.references:
            t=row["t"]
            if t.ndim!=1 or not 3<=len(t)<=6001 or not bool(torch.isfinite(t).all()):
                raise ValueError("Finite bounded exact5ms reference grid required")
            expected=torch.arange(len(t),device=t.device,dtype=t.dtype)*.005
            # Disk grids have already passed1e-12 checks before float32 loading.
            # This tolerance only accounts for float32 conversion/multiplication.
            if not bool(torch.allclose(t,expected,atol=4.e-6,rtol=0)):
                raise ValueError("Reference grid differs from the accepted5ms master clock")

    def sample(self,key,reference_ids,ticks):
        if (reference_ids.ndim!=1 or ticks.shape!=reference_ids.shape
                or reference_ids.dtype!=torch.long or ticks.dtype!=torch.long
                or reference_ids.device!=ticks.device
                or bool(((reference_ids<0)|(reference_ids>=len(self.references))).any())
                or bool((ticks<0).any())):
            raise ValueError("One nonnegative integer clock and valid exact reference ID per world required")
        output=None
        for index,row in enumerate(self.references):
            values=row[key]
            if values.device!=ticks.device:raise ValueError("Reference and query devices differ")
            sequence=values.ndim and len(values)==len(row["t"]) and key not in CONSTANT_KEYS
            sampled=(values[ticks.clamp(max=len(row["t"])-1)] if sequence
                     else values.expand(len(ticks),*values.shape))
            if output is None:output=torch.empty_like(sampled)
            selected=reference_ids==index
            output[selected]=sampled[selected]
        if key=="qpos":
            output[:,3:7]/=output[:,3:7].norm(dim=-1,keepdim=True).clamp_min(1.e-8)
        return output


class IntegerReferencePlayback:
    """Composition mixin for NativeSpecialistEnv subclasses using new recipes."""
    def install_integer_reference_playback(self):
        self.integer_reference_sampler=IntegerReferenceSampler(self.library.arrays)

    def ref(self,key,offset=0.):
        # NativeSpecialistEnv construction creates the library before any reset.
        # Initialization users install explicitly before the first physical step.
        if not hasattr(self,"integer_reference_sampler"):
            return super().ref(key,offset)
        extra=future_offset_ticks(offset)
        query=self.ticks+torch.where(self.rejected,0,extra)
        return self.integer_reference_sampler.sample(key,self.ids_ref,query)
