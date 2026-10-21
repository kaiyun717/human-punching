"""Saved-audit checkpoint selection: exact declared file, then complete state digest."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]
VERSION='task10b_phase2_exact_evaluated_checkpoint_selection_v1'

def policy_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def matching_evaluated_checkpoint(paths,required_digest,binding):
    import torch
    path=Path(binding['path']).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve(strict=True)) or not path.is_file():
        raise ValueError('Owned explicitly evaluated checkpoint required')
    if path not in [Path(p).resolve(strict=True) for p in paths]:
        raise ValueError('Evaluated checkpoint absent from retained checkpoint inventory')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
        raise ValueError('Explicit evaluated checkpoint bytes changed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['policy']
    if policy_digest(state)!=required_digest:
        raise ValueError('Explicit evaluated checkpoint parameters differ from saved actor digest')
    return path,state
