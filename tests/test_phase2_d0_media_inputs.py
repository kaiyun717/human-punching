"""Read-only D0 media closure binds real traces without stale argument routes."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path('/orcd/home/002/kaisyun/Azizan_Lab/HumanBox/human-punching')
SPEC=importlib.util.spec_from_file_location('d0_media_input_driver',ROOT/'scripts/task10b_phase2_expand.py')
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)
from human_punching.phase2_d0_media_inputs import files
from human_punching.phase2_runtime_namespace import evidence

RUNS={'jab':'p2t01j','cross':'p2t02c','left_hook':'p2t02l','right_hook':'p2t02r'}


def record(family):
    run=ROOT/'outputs/task10b'/RUNS[family]/'run'
    return dict(mode='media',inputs=dict(run=str(run),panel=str(run/'validation_00000/summary.json')),
        release='outputs/four-punch-milestone',bundle='outputs/task10/release_baseline_02',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')


@pytest.mark.parametrize('family',list(RUNS))
def test_all_four_actual_d0_media_closures_bind_full_ledger_raw_states_and_opaque_accepted_arguments(family):
    r=record(family);before=driver.sha256_file(ROOT/r['inputs']['run']/'launch.json')
    closure=driver.required_inputs(r)
    run=Path(r['inputs']['run'])
    for p in run.rglob('*'):
        if p.is_file():assert closure[str(p.relative_to(ROOT))]==driver.sha256_file(p)
    args=f'outputs/four-punch-milestone/{family}/arguments.json'
    assert args in closure
    assert '/home/gridsan' in (ROOT/args).read_text()
    assert 'outputs/task10b/p2timingcatalog01/manifest.json' in closure
    assert driver.sha256_file(run/'launch.json')==before
    assert not any('/home/gridsan' in p for p in closure)


def test_optional_proof_and_nonce_are_explicit_portable_packet_inputs():
    r=record('jab');proof=evidence(ROOT,'outputs/task10b/p2hostalias01/probe.json')
    r.update(runtime_namespace_proof=proof['proof'],runtime_namespace_evidence=proof)
    closure=driver.required_inputs(r)
    for item in [proof['proof'],proof['nonce'],*proof['alias_bindings']]:assert closure[item['path']]==item['sha256']
    assert closure['src/human_punching/phase2_runtime_namespace.py']==driver.sha256_file(ROOT/'src/human_punching/phase2_runtime_namespace.py')
    r['runtime_namespace_evidence']={**proof,'permitted_path_roles':['catalog']}
    with pytest.raises(ValueError,match='evidence changed'):driver.required_inputs(r)


@pytest.mark.parametrize('change',['catalog_sha','accepted_input_path','accepted_input_sha','missing_trial','panel_escape','planned_cells'])
def test_d0_media_closure_rejects_tampered_real_bindings_and_ledgers(monkeypatch,change):
    r=record('jab');run=Path(r['inputs']['run']);launch=run/'launch.json';panel=Path(r['inputs']['panel'])
    original=Path.read_text
    def changed(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path==launch:
            data=json.loads(value)
            if change=='catalog_sha':data['catalog_sha256']='0'*64
            elif change=='accepted_input_path':data['release_inputs']['arguments']['path']=str(ROOT/'outputs/four-punch-milestone/jab/body_metadata.json')
            elif change=='accepted_input_sha':data['release_inputs']['arguments']['sha256']='0'*64
            elif change=='planned_cells':data['planned_cells'].pop()
            return json.dumps(data)
        if path==panel:
            data=json.loads(value)
            if change=='missing_trial':data['trials'].pop()
            elif change=='panel_escape':data['panels'][0]['path']=str(ROOT/'outputs/task10b/p2t02c/run')
            return json.dumps(data)
        return value
    monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(ValueError):driver.required_inputs(r)
