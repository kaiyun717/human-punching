"""Exact semantic edges with synthetic data; no model or native execution."""
import copy
import json
from pathlib import Path

import pytest

from human_punching import phase2_pipeline_inputs_v2 as inputs
from human_punching.wbc.provenance import sha256_file


def encoded(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': sha256_file(path)}


def test_domain_local_ledger_and_opaque_documentary_paths(tmp_path):
    folder = tmp_path/'domain'; folder.mkdir()
    ledger = folder/'all_slot_dispositions.npz'; ledger.write_bytes(b'bound complete dispositions')
    (tmp_path/ledger.name).write_bytes(b'wrong root member must never be selected')
    domain = encoded(folder/'domain.json', dict(schema='Task10BPhase2Domain/v1',
        all_slot_ledger={'path': ledger.name, 'sha256': sha256_file(ledger)},
        provenance={'path': '/never/read/paper.pdf', 'xml_path': '/home/gridsan/obsolete.xml'}))
    collector = inputs.Inputs(tmp_path)
    collector.domain(**{'path': domain['path'], 'digest': domain['sha256']})
    assert set(collector.finish()) == {'domain/domain.json', 'domain/all_slot_dispositions.npz'}
    ledger.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        inputs.Inputs(tmp_path).domain(domain['path'], domain['sha256'])


@pytest.mark.parametrize('value', ['../outside.npz', '/does/not/exist'])
def test_ledger_wrong_scope_or_missing_is_not_silently_ignored(tmp_path, value):
    domain = encoded(tmp_path/'domain.json', dict(schema='Task10BPhase2Domain/v1',
        all_slot_ledger={'path': value, 'sha256': 'a'*64}))
    with pytest.raises((ValueError, FileNotFoundError)):
        inputs.Inputs(tmp_path).domain(domain['path'], domain['sha256'])


@pytest.fixture
def pack_plan(tmp_path):
    from human_punching.phase2_runtime_pack import VERSION
    payload = tmp_path/'reference.npz'; payload.write_bytes(b'hash-only archive fixture')
    model = tmp_path/'model.xml'; model.write_text('<mujoco><worldbody/></mujoco>')
    recipe = encoded(tmp_path/'recipe.json', dict(schema='task10b_phase2_target_recipe_v1',
        start_mode='family_guard', xml_path='/home/gridsan/never/read.xml'))
    row = dict(candidate_id='declared', family='jab', split='train', kind='target_reference',
        start_mode='family_guard', recipe_sha256=recipe['sha256'], kinematic_admission=True,
        reference_path=str(payload), reference_sha256=sha256_file(payload),
        legacy_runtime_path=str(payload), legacy_runtime_sha256=sha256_file(payload),
        source_model_path=str(model), source_model_sha256=sha256_file(model))
    catalog = encoded(tmp_path/'catalog.json', dict(schema=inputs.CATALOG, complete=True, cells=[row]))
    plan = dict(schema='task10b_phase2_runtime_pack_plan_v1', storage_version=VERSION,
        byte_cap=1024**2, catalogs=[catalog], recipes=[recipe], previous_packed_catalogs=[])
    binding = encoded(tmp_path/'plan.json', plan)
    return tmp_path, binding, plan, row


def test_runtime_pack_exact_payload_closure_no_historical_path_walk(pack_plan):
    root, binding, _, _ = pack_plan
    result = inputs.files('runtime-pack', {'plan': binding['path']}, root=root)
    assert set(result) == {'plan.json', 'catalog.json', 'recipe.json', 'reference.npz', 'model.xml'}


@pytest.mark.parametrize('file', ['reference.npz', 'model.xml', 'catalog.json', 'recipe.json'])
def test_runtime_input_edits_rejected(pack_plan, file):
    root, binding, _, _ = pack_plan
    (root/file).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        inputs.files('runtime-pack', {'plan': binding['path']}, root=root)


@pytest.mark.parametrize('change', ['partial', 'test', 'original', 'other_recipe', 'shared', 'legacy'])
def test_exact_runtime_scope_refuses_substitution(pack_plan, change):
    root, binding, plan, row = pack_plan
    catalog = json.loads((root/'catalog.json').read_text())
    if change == 'partial': catalog['complete'] = False
    elif change == 'test': catalog['cells'][0]['split'] = 'test'
    elif change == 'original': catalog['cells'][0]['kind'] = 'original_control'
    elif change == 'shared': catalog['cells'][0]['start_mode'] = 'shared_jab_saved_state'
    elif change == 'other_recipe': catalog['cells'][0]['recipe_sha256'] = '0'*64
    else:
        other = root/'other.npz'; other.write_bytes(b'other')
        catalog['cells'][0].update(legacy_runtime_path=str(other), legacy_runtime_sha256=sha256_file(other))
    plan['catalogs'] = [encoded(root/'catalog.json', catalog)]
    encoded(root/'plan.json', plan)
    with pytest.raises(ValueError): inputs.files('runtime-pack', {'plan': binding['path']}, root=root)


def test_hash_bound_symlink_escape_refused(pack_plan):
    root, binding, plan, row = pack_plan
    outside = root.parent/(root.name+'_external'); outside.write_bytes(b'outside')
    link = root/'escape.npz'; link.symlink_to(outside)
    row.update(reference_path=str(link), legacy_runtime_path=str(link),
               reference_sha256=sha256_file(outside), legacy_runtime_sha256=sha256_file(outside))
    plan['catalogs'] = [encoded(root/'catalog.json', dict(schema=inputs.CATALOG, complete=True, cells=[row]))]
    encoded(root/'plan.json', plan)
    with pytest.raises(ValueError, match='owned'):
        inputs.files('runtime-pack', {'plan': binding['path']}, root=root)


def test_unclosed_model_assets_refused(pack_plan):
    root, binding, plan, row = pack_plan
    model = root/'model.xml'; model.write_text('<mujoco><include file="foreign.xml"/></mujoco>')
    row['source_model_sha256'] = sha256_file(model)
    plan['catalogs'] = [encoded(root/'catalog.json', dict(schema=inputs.CATALOG, complete=True, cells=[row]))]
    encoded(root/'plan.json', plan)
    with pytest.raises(ValueError, match='Self-contained'):
        inputs.files('runtime-pack', {'plan': binding['path']}, root=root)


def test_constructor_failure_is_preserved_without_invented_model_payload(pack_plan):
    root, binding, plan, row = pack_plan
    reasons = ['declared target cannot be reached within fixed refinement cap']
    failure = encoded(root/'failure.json', dict(recipe_sha256=row['recipe_sha256'], rejection_reasons=reasons))
    negative = dict(candidate_id='declared_failed', family='jab', split='train', start_mode='family_guard',
        recipe_sha256=row['recipe_sha256'], reference_identity_kind='generation_failure_receipt',
        kinematic_admission=False, generation_failure_path=failure['path'],
        generation_failure_sha256=failure['sha256'], reference_sha256=failure['sha256'], rejection_reasons=reasons)
    plan['catalogs'] = [encoded(root/'catalog.json', dict(schema=inputs.CATALOG, complete=True, cells=[row, negative]))]
    encoded(root/'plan.json', plan)
    result = inputs.files('runtime-pack', {'plan': binding['path']}, root=root)
    assert result['failure.json'] == failure['sha256']
    original = json.loads((root/'catalog.json').read_text())
    assert original['cells'][1] == negative and not original['cells'][1]['kinematic_admission']


def test_declared_negative_wrapper_parent_alias_is_stable(tmp_path, monkeypatch):
    from human_punching import phase2_runtime_pack as packing
    real = tmp_path/'physical'; real.mkdir()
    logical = tmp_path/'logical'; logical.symlink_to(real, target_is_directory=True)
    parent = encoded(real/'source_failure.json', dict(recipe_sha256='a'*64, rejection_reasons=['failed source']))
    ancestry = dict(runtime_storage_version=packing.VERSION, runtime_failure_wrapper_version=packing.FAILURE_WRAPPER,
        unpacked_recipe_sha256='a'*64, unpacked_generation_failure_path=str(logical/'source_failure.json'),
        unpacked_generation_failure_sha256=parent['sha256'])
    wrapper = encoded(real/'packed_failure.json', dict(recipe_sha256='b'*64, rejection_reasons=['failed source'], **ancestry))
    row = dict(recipe_sha256='b'*64, rejection_reasons=['failed source'], **ancestry,
        reference_identity_kind='generation_failure_receipt', kinematic_admission=False,
        generation_failure_path=str(logical/'packed_failure.json'), generation_failure_sha256=wrapper['sha256'],
        reference_sha256=wrapper['sha256'])
    def checked(binding):
        path = Path(binding['path']).resolve(strict=True)
        assert path.is_relative_to(real) and sha256_file(path) == binding['sha256']
        return path
    monkeypatch.setattr(packing, 'owned_input', checked)
    with pytest.raises(ValueError, match='changed'): packing.verify_rejection(row)
    with inputs.declared_packing_paths():
        path, evidence = packing.verify_rejection(row)
        assert path == logical/'packed_failure.json' and evidence['unpacked_generation_failure_path'] == str(logical/'source_failure.json')
    assert packing.owned_input is checked


@pytest.fixture
def grouped_plan(pack_plan, monkeypatch):
    from human_punching.phase2_runtime_pack import VERSION
    root, _, _, row = pack_plan
    ledger = root/'domain/dispositions.npz'; ledger.parent.mkdir(); ledger.write_bytes(b'ledger')
    domain = encoded(ledger.parent/'domain.json', dict(schema='Task10BPhase2Domain/v1',
        all_slot_ledger=dict(path=ledger.name, sha256=sha256_file(ledger))))
    geometry = encoded(root/'geometry.json', {'schema': 'opaque actual schema validated by scientific preflight'})
    recipe = encoded(root/'packed_recipe.json', {'xml_path': '/home/gridsan/never/read.xml'})
    def packed(split):
        original = encoded(root/(split+'_source.json'), dict(schema=inputs.CATALOG, complete=True, cells=[row]))
        value = dict(row, split=split, recipe_sha256=recipe['sha256'], runtime_storage_version=VERSION,
            unpacked_reference_path=row['reference_path'], unpacked_reference_sha256=row['reference_sha256'])
        return encoded(root/(split+'_packed.json'), dict(schema=inputs.CATALOG, complete=True,
            runtime_pack_complete=True, runtime_storage_version=VERSION, cells=[value],
            unpacked_catalog_path=original['path'], unpacked_catalog_sha256=original['sha256']))
    train, validation = packed('train'), packed('validation')
    starts = root/'starts.npz'; starts.write_bytes(b'four declared starts')
    manifest = encoded(root/'validation_manifest.json', dict(schema='task10b_phase2_expanded_manifest_v1',
        recipe_sha256=recipe['sha256'], domain_path=domain['path'], domain_sha256=domain['sha256'],
        target_geometry_path=geometry['path'], target_geometry_sha256=geometry['sha256'],
        catalog_path=validation['path'], catalog_sha256=validation['sha256'],
        trials=[dict(start_bank_path=str(starts), start_bank_sha256=sha256_file(starts))]))
    opaque = encoded(root/'opaque.json', {'xml_path': '/home/gridsan/never/read.xml'})
    callback = encoded(root/'callback.json', dict(schema='task10b_phase2_grouped_validation_configuration_v1',
        descriptor=opaque, declaration_template=opaque, manifest=manifest, mask=opaque, predicate=opaque, domain=domain))
    config = dict(schema='task10b_phase2_grouped_training_configuration_v1', declaration=opaque,
        catalog=train, domain=domain, target_geometry=geometry, predicate=opaque, mask=opaque,
        recipe=recipe, validation_configuration=callback)
    calls = []
    monkeypatch.setattr(inputs, 'grouped_preflight', lambda *args: calls.append(args))
    binding = encoded(root/'configuration.json', config)
    return root, binding, config, calls


def test_grouped_closes_domain_ledger_callback_starts_and_ancestry(grouped_plan):
    root, binding, _, calls = grouped_plan
    provenance = {}
    result = inputs.files('grouped-pilot', {'configuration': binding['path']}, root=root,
                          provenance_files=provenance)
    assert {'domain/dispositions.npz', 'starts.npz', 'callback.json', 'validation_manifest.json'} <= result.keys()
    assert set(provenance) == {'train_source.json', 'validation_source.json', 'reference.npz'}
    assert len(calls) == 1 and calls[0][1] == binding['sha256']


def test_scientific_preflight_failure_propagates(grouped_plan, monkeypatch):
    root, binding, _, _ = grouped_plan
    def rejected(*args): raise ValueError('unsupported exact 64/96 scientific requests')
    monkeypatch.setattr(inputs, 'grouped_preflight', rejected)
    with pytest.raises(ValueError, match='scientific'):
        inputs.files('grouped-pilot', {'configuration': binding['path']}, root=root)


def test_start_bank_tamper_is_caught_before_scientific_preflight(grouped_plan):
    root, binding, _, calls = grouped_plan
    (root/'starts.npz').write_bytes(b'changed start')
    with pytest.raises(ValueError, match='SHA'):
        inputs.files('grouped-pilot', {'configuration': binding['path']}, root=root)
    assert not calls


@pytest.mark.parametrize('mode,arguments', [('grouped-finetune', {}), ('runtime-pack', {'plan': 'x', 'command': 'arbitrary'}), ('shared', {})])
def test_unsupported_mode_or_extra_input_refused(tmp_path, mode, arguments):
    with pytest.raises(ValueError, match='supported'):
        inputs.files(mode, arguments, root=tmp_path)
