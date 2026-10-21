"""Explicit runtime alias proofs preserve all original byte and link checks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from human_punching import phase2_runtime_namespace as runtime

ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def project(tmp_path,monkeypatch):
    logical=tmp_path/'logical';compute=tmp_path/'compute';logical.mkdir();compute.mkdir()
    def paired(relative,content):
        a,b=logical/relative,compute/relative;a.parent.mkdir(parents=True,exist_ok=True);b.parent.mkdir(parents=True,exist_ok=True)
        a.write_text(content);os.link(a,b);return a
    parent=paired('scripts/task10b_phase2.py','preserved parent verifier fixture')
    nonce=paired('outputs/task10b/proof/nonce.json',json.dumps(dict(schema='task10b_phase2_host_mount_alias_nonce_v1',root=str(logical),compute_root=str(compute),nonce='fresh_test_nonce')))
    rel=parent.relative_to(logical)
    proof=dict(schema='task10b_phase2_host_mount_alias_probe_v1',passed=True,local_root=str(logical),compute_root=str(compute),
        fresh_owned_nonce_visible_identically_through_both_mounts=True,nonce_sha256=digest(nonce),
        bindings=[dict(relative_path=str(rel),local_sha256=digest(parent),compute_mount_sha256=digest(parent),local_inode=parent.stat().st_ino,compute_inode=parent.stat().st_ino)])
    path=paired('outputs/task10b/proof/probe.json',json.dumps(proof))
    monkeypatch.setattr(runtime,'PROOF_SHA256',digest(path));monkeypatch.setattr(runtime,'PARENT_SHA256',digest(parent))
    python=logical/'python';python.mkdir();a=python/'payload';a.write_text('immutable interpreter payload');b=python/'other';b.write_text('other interpreter payload')
    (python/'relative_link').symlink_to('payload')
    entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in (a,b)]+[dict(path='relative_link',symlink='payload')]
    prep=logical/'prep';manifest=dump(prep/'runtime_interpreter_manifest.json',entries)
    metadata=paired('packages/dist-info/METADATA','immutable package metadata')
    config=dict(local_python_root=str(python),interpreter_manifest_sha256=digest(manifest),
        distribution_metadata_hashes={'fake/METADATA':dict(bytes=metadata.stat().st_size,sha256=digest(metadata))},
        distribution_metadata_verification={'fake/METADATA':dict(path=str(metadata))})
    dump(prep/'runtime.json',config)
    monkeypatch.setattr(runtime,'runtime_record',lambda:{'declared_pins_match':True})
    monkeypatch.setattr(runtime.importlib.metadata,'distribution',lambda name:SimpleNamespace(_path=metadata.parent))
    return SimpleNamespace(logical=logical,compute=compute,proof=path,nonce=nonce,parent=parent,python=python,prep=prep,metadata=metadata,config=config)


def test_proof_evidence_and_manifest_checks_are_portable_and_complete(project):
    p=project;left=runtime.evidence(p.logical,p.proof);right=runtime.evidence(p.compute,p.compute/p.proof.relative_to(p.logical))
    assert left==right and not Path(left['proof']['path']).is_absolute()
    verified=runtime.verify_runtime(p.logical,p.prep,p.proof)
    assert verified['proof']==left and verified['interpreter_members_verified']==3
    assert verified['distribution_metadata_members_verified']==1
    mapped=runtime.runtime_member(p.compute,p.metadata,left,role='installed_distribution_metadata',declared_path=p.metadata)
    assert mapped==p.compute/p.metadata.relative_to(p.logical)


@pytest.mark.parametrize('change',['proof_hash','nonce_hash','nonce_inode','binding_payload','parent_source','outside_proof'])
def test_proof_rejects_changed_nonce_bindings_parent_or_ownership(project,tmp_path,change):
    p=project;proof=p.proof
    if change=='proof_hash':proof.write_text(proof.read_text()+' ')
    elif change=='nonce_hash':p.nonce.write_text('changed nonce')
    elif change=='nonce_inode':
        remote=p.compute/p.nonce.relative_to(p.logical);content=remote.read_bytes();remote.unlink();remote.write_bytes(content)
    elif change in ('binding_payload','parent_source'):p.parent.write_text('changed parent source')
    else:proof=tmp_path/'outside.json';proof.write_bytes(p.proof.read_bytes())
    with pytest.raises((ValueError,FileNotFoundError)):runtime.evidence(p.logical,proof)


@pytest.mark.parametrize('change',['payload','link_target','link_escape','metadata_payload','metadata_selected_path','pins','manifest_hash','manifest_escape'])
def test_runtime_retains_each_manifest_metadata_and_pin_gate(project,monkeypatch,change):
    p=project
    if change=='payload':(p.python/'payload').write_text('changed interpreter bytes')
    elif change in ('link_target','link_escape'):
        link=p.python/'relative_link';link.unlink()
        if change=='link_escape':(p.logical/'outside').write_text('outside');link.symlink_to('../outside')
        else:link.symlink_to('other')
    elif change=='metadata_payload':p.metadata.write_text('changed dependency metadata')
    elif change=='metadata_selected_path':
        other=p.logical/'other_dist/METADATA';other.parent.mkdir();other.write_bytes(p.metadata.read_bytes())
        monkeypatch.setattr(runtime.importlib.metadata,'distribution',lambda name:SimpleNamespace(_path=other.parent))
    elif change=='pins':monkeypatch.setattr(runtime,'runtime_record',lambda:{'declared_pins_match':False})
    elif change=='manifest_hash':(p.prep/'runtime_interpreter_manifest.json').write_text('[]')
    else:
        (p.logical/'outside').write_text('outside')
        manifest=dump(p.prep/'runtime_interpreter_manifest.json',[dict(path='../outside',bytes=7,sha256=digest(p.logical/'outside'))])
        p.config['interpreter_manifest_sha256']=digest(manifest);dump(p.prep/'runtime.json',p.config)
    with pytest.raises(ValueError):runtime.verify_runtime(p.logical,p.prep,p.proof)


@pytest.mark.parametrize('change',['role','outside_prefix','changed_declared','parent_traversal','wrong_inode'])
def test_mapping_is_limited_to_the_exact_two_declared_runtime_roles(project,tmp_path,change):
    p=project;proof=runtime.evidence(p.logical,p.proof);raw=p.metadata;declared=raw;role='installed_distribution_metadata'
    if change=='role':role='catalog'
    elif change=='outside_prefix':raw=tmp_path/'outside';raw.write_text('outside');declared=raw
    elif change=='changed_declared':declared=p.parent
    elif change=='parent_traversal':raw=p.metadata.parent/'..'/p.metadata.parent.name/'METADATA';declared=raw
    else:
        target=p.compute/p.metadata.relative_to(p.logical);content=target.read_bytes();target.unlink();target.write_bytes(content)
    with pytest.raises(ValueError):runtime.runtime_member(p.compute,raw,proof,role=role,declared_path=declared)


def test_actual_reviewed_runtime_is_verified_in_both_container_namespaces():
    results=[]
    for root in (ROOT,Path('/orcd/home/002/kaisyun/Azizan_Lab/HumanBox/human-punching')):
        results.append(runtime.verify_runtime(root,root/'outputs/task10b/p2prep_20261021_01','outputs/task10b/p2hostalias01/probe.json',python_path=root/'.venv-task10b/bin/python'))
    assert results[0]==results[1]
    assert results[0]['interpreter_members_verified']==3931+1048
    assert results[0]['distribution_metadata_members_verified']==22


def test_optional_driver_proof_absence_uses_unchanged_base_verifier(monkeypatch):
    spec=importlib.util.spec_from_file_location('namespace_driver',ROOT/'scripts/task10b_phase2_expand.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    seen=[];monkeypatch.setattr(driver.base,'verify_runtime',lambda:seen.append('old verifier'))
    assert driver.verify_packet_runtime({}) is None and seen==['old verifier']
    with pytest.raises(ValueError):driver.verify_packet_runtime({'runtime_namespace_evidence':{}})


def test_metadata_selected_in_compute_namespace_matches_logical_preserved_receipt(project):
    p=project;proof=runtime.evidence(p.compute,p.compute/p.proof.relative_to(p.logical))
    selected=p.compute/p.metadata.relative_to(p.logical)
    assert runtime.runtime_member(p.compute,selected,proof,role='installed_distribution_metadata',declared_path=p.metadata)==selected


@pytest.mark.parametrize('change',['none','wrong_leaf','wrong_inode','changed_bytes','redirected_symlink'])
def test_executable_leaf_stays_in_exact_verified_tree_and_preserves_identity(project,change):
    p=project;owned_tree=p.compute/'python';owned_tree.mkdir()
    for name in ('payload','other'):os.link(p.python/name,owned_tree/name)
    entry=p.logical/'venv_python';entry.symlink_to(p.python/'payload')
    config={**p.config,'resolved_python':str(p.python/'payload')}
    if change=='wrong_leaf':entry.unlink();entry.symlink_to(p.python/'other')
    elif change in ('wrong_inode','changed_bytes'):
        counterpart=owned_tree/'payload';value=counterpart.read_bytes();counterpart.unlink();counterpart.write_bytes(value if change=='wrong_inode' else value+b'changed')
    elif change=='redirected_symlink':
        identical=p.logical/'identical_executable';identical.write_bytes((p.python/'payload').read_bytes());entry.unlink();entry.symlink_to(identical)
    if change=='none':assert runtime.interpreter_leaf(owned_tree,config,entry)==owned_tree/'payload'
    else:
        with pytest.raises(ValueError):runtime.interpreter_leaf(owned_tree,config,entry)
