"""Explicit, byte-bound verification of the two preserved runtime path roles.

The original launcher's verifier stays unchanged. No general input ownership or
path resolver uses these aliases; runtime manifest members keep their full checks.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

from .wbc.provenance import sha256_file,runtime_record

VERSION='task10b_phase2_explicit_runtime_namespace_v1'
PROOF_SHA256='0f086ac3dc4e123632a4ab7a959a01eb3de887d87fcd442c93d0f4fd8ce54b2f'
PARENT_SHA256='a6f4994cda915eae7f3cb5e485d65bfe8930a39e5ede4a4c5aef1825d3a2549f'


def owned(root,path):
    root=Path(root).resolve(strict=True);p=Path(path)
    p=(p if p.is_absolute() else root/p).resolve(strict=True)
    if not p.is_relative_to(root):raise ValueError('Runtime namespace input escaped owned root')
    return p


def evidence(root,proof_path):
    root=Path(root).resolve(strict=True);proof_path=owned(root,proof_path)
    if sha256_file(proof_path)!=PROOF_SHA256:raise ValueError('Exact reviewed runtime namespace proof required')
    proof=json.loads(proof_path.read_text());logical=Path(proof['local_root']);compute=Path(proof['compute_root'])
    if (proof.get('schema')!='task10b_phase2_host_mount_alias_probe_v1' or proof.get('passed') is not True
            or proof.get('fresh_owned_nonce_visible_identically_through_both_mounts') is not True
            or not logical.is_absolute() or not compute.is_absolute() or logical==compute
            or root not in (logical,compute)):
        raise ValueError('Runtime namespace proof is outside its two exact root spellings')
    relative=proof_path.relative_to(root);nonce_relative=relative.parent/'nonce.json'
    nonce=owned(root,nonce_relative);left=logical/nonce_relative;right=compute/nonce_relative
    if (sha256_file(nonce)!=proof['nonce_sha256'] or sha256_file(left)!=proof['nonce_sha256']
            or sha256_file(right)!=proof['nonce_sha256'] or left.stat().st_ino!=right.stat().st_ino):
        raise ValueError('Fresh owned runtime namespace nonce differs')
    payload=json.loads(nonce.read_text())
    if (payload.get('schema')!='task10b_phase2_host_mount_alias_nonce_v1'
            or payload.get('root')!=str(logical) or payload.get('compute_root')!=str(compute)
            or not payload.get('nonce')):raise ValueError('Runtime namespace nonce identity differs')
    bindings=[]
    for item in proof['bindings']:
        rel=Path(item['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Runtime alias binding escaped its root')
        a,b=logical/rel,compute/rel;p=owned(root,rel)
        if (not a.is_file() or not b.is_file() or a.stat().st_ino!=b.stat().st_ino
                or a.stat().st_ino!=item['local_inode'] or b.stat().st_ino!=item['compute_inode']
                or sha256_file(a)!=item['local_sha256'] or sha256_file(b)!=item['compute_mount_sha256']
                or item['local_sha256']!=item['compute_mount_sha256']):
            raise ValueError('Preserved runtime alias binding changed')
        bindings.append(dict(path=str(p.relative_to(root)),sha256=sha256_file(p)))
    bind=lambda p:dict(path=str(owned(root,p).relative_to(root)),sha256=sha256_file(owned(root,p)))
    parent=owned(root,'scripts/task10b_phase2.py')
    if sha256_file(parent)!=PARENT_SHA256:raise ValueError('Original runtime verifier source changed')
    return dict(version=VERSION,proof=bind(proof_path),nonce=bind(nonce),alias_bindings=bindings,
        source_verifier_ancestry=bind(parent),logical_root=str(logical),compute_root=str(compute),
        permitted_path_roles=['local_python_root','installed_distribution_metadata'])


def runtime_member(root,path,proof,*,role,declared_path):
    """Use normal ownership first, then only the proved, exact declared role."""
    root=Path(root).resolve(strict=True);raw=Path(path)
    if not raw.is_absolute() or '..' in raw.parts:raise ValueError('Absolute runtime member required')
    resolved=raw.resolve(strict=True)
    if role not in ('local_python_root','installed_distribution_metadata'):
        raise ValueError('Unreviewed runtime path role')
    declared=Path(declared_path);logical=Path(proof['logical_root'])
    if not declared.is_absolute() or '..' in declared.parts:raise ValueError('Absolute declared runtime member required')
    expected=declared.resolve(strict=True)
    if not expected.is_relative_to(root):
        if not declared.is_relative_to(logical):raise ValueError('Declared runtime member left its exact logical prefix')
        expected=owned(root,declared.relative_to(logical))
        if expected.stat().st_ino!=declared.stat().st_ino:raise ValueError('Declared runtime member inode differs')
    if resolved.is_relative_to(root):
        if expected!=resolved:raise ValueError('Selected runtime member differs from preserved receipt')
        return resolved
    if not raw.is_relative_to(logical) or raw!=declared:
        raise ValueError('Only the exact preserved logical runtime member may be mapped')
    mapped=owned(root,raw.relative_to(logical))
    if mapped!=expected or mapped.stat().st_ino!=raw.stat().st_ino:raise ValueError('Mapped runtime member inode differs')
    return mapped


def interpreter_leaf(python_root,runtime,python_path):
    """Resolve the exact executable leaf inside the proved interpreter tree."""
    python_root=Path(python_root).resolve(strict=True);actual=Path(python_path).resolve(strict=True)
    declared_root=Path(runtime['local_python_root']).resolve(strict=True)
    if actual!=Path(runtime['resolved_python']).resolve(strict=True):raise ValueError('Interpreter executable leaf differs from preserved receipt')
    if not actual.is_relative_to(declared_root):raise ValueError('Interpreter executable escaped its declared tree')
    member=(python_root/actual.relative_to(declared_root)).resolve(strict=True)
    if (not member.is_relative_to(python_root) or member.stat().st_ino!=actual.stat().st_ino
            or sha256_file(member)!=sha256_file(actual)):
        raise ValueError('Owned interpreter executable inode or bytes differ')
    return member


def verify_runtime(root,prep,proof_path,*,python_path=None):
    """Copy the original manifest/link/payload/metadata checks with role mapping."""
    root=Path(root).resolve(strict=True);prep=owned(root,prep);proof=evidence(root,proof_path)
    runtime=json.loads((prep/'runtime.json').read_text())
    if runtime_record().get('declared_pins_match') is not True:raise ValueError('Preserved runtime pins differ')
    python_root=runtime_member(root,runtime['local_python_root'],proof,role='local_python_root',declared_path=runtime['local_python_root'])
    manifest=prep/'runtime_interpreter_manifest.json'
    if sha256_file(manifest)!=runtime['interpreter_manifest_sha256']:
        raise ValueError('Interpreter manifest changed')
    entries=json.loads(manifest.read_text())
    for item in entries:
        member=python_root/item['path']
        if not member.resolve(strict=True).is_relative_to(python_root):raise ValueError('Interpreter path escaped')
        if 'symlink' in item:
            if not member.is_symlink() or os.readlink(member)!=item['symlink']:raise ValueError('Interpreter symlink changed')
        elif member.is_symlink() or member.stat().st_size!=item['bytes'] or sha256_file(member)!=item['sha256']:
            raise ValueError('Interpreter payload changed')
    for key,expected in runtime['distribution_metadata_hashes'].items():
        package,filename=key.rsplit('/',1)
        selected=Path(importlib.metadata.distribution(package)._path)/filename
        declared=runtime['distribution_metadata_verification'][key]['path']
        member=runtime_member(root,selected,proof,role='installed_distribution_metadata',declared_path=declared)
        if member.stat().st_size!=expected['bytes'] or sha256_file(member)!=expected['sha256']:
            raise ValueError('Installed dependency metadata changed')
    if python_path is not None:interpreter_leaf(python_root,runtime,python_path)
    return dict(proof=proof,interpreter_members_verified=len(entries),distribution_metadata_members_verified=len(runtime['distribution_metadata_hashes']),executable_leaf_verified=python_path is not None)
