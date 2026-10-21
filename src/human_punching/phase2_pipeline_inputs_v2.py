"""Exact isolated grouped-PPO/runtime-pack inputs; documentary ancestry is opaque.

No native models, actor calls, reference construction or filesystem mutations.
The existing scientific validators still own admission and split semantics.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import xml.etree.ElementTree as ET

from .wbc.provenance import PROJECT_ROOT, sha256_file

VERSION = "task10b_phase2_isolated_pipeline_inputs_v2"
CATALOG = "task10b_phase2_reference_catalog_v1"
MODES = ("runtime-pack", "grouped-pilot")


@contextmanager
def declared_packing_paths():
    """Keep exact declared aliases after unchanged ownership/hash validation.

    The original negative-wrapper verifier compares a serialized parent name.
    Resolving that name must not rewrite its spelling on a different mount.
    This single-threaded adapter maps no foreign or missing path.
    """
    from . import phase2_runtime_pack as packing
    original = packing.owned_input
    def declared(binding):
        checked = original(binding)
        raw = Path(binding['path'])
        if raw.is_absolute() and raw.resolve(strict=True) == checked.resolve(strict=True):
            return raw
        return checked
    try:
        packing.owned_input = declared
        yield
    finally:
        packing.owned_input = original


class Inputs:
    """Only named semantic edges are followed; JSON path searches are forbidden."""
    def __init__(self, root, provenance=None):
        self.root = Path(root).resolve(strict=True)
        self.files = {}
        self.provenance = provenance if provenance is not None else {}

    def path(self, value, *, relative_to=None):
        raw = Path(value)
        if not raw.is_absolute():
            if ".." in raw.parts:
                raise ValueError("Relative semantic input may not traverse parent directories")
            raw = (relative_to or self.root)/raw
        path = raw.resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Pipeline input must be an owned regular file")
        return path

    def add(self, value, expected=None, *, relative_to=None, provenance_role=None):
        path = self.path(value, relative_to=relative_to)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("Pipeline input SHA changed")
        name = str(path.relative_to(self.root))
        if provenance_role is None:
            if self.files.setdefault(name, digest) != digest:
                raise ValueError("Input changed during exact closure")
        else:
            entry = self.provenance.setdefault(name, dict(sha256=digest,
                bytes=path.stat().st_size, roles=[]))
            if entry['sha256'] != digest:
                raise ValueError("Conflicting preserved ancestry identity")
            entry['roles'] = sorted(set(entry['roles']) | {provenance_role})
        return path

    def bound(self, value, **kwargs):
        if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
                or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in value['sha256'])):
            raise ValueError("Exact path/SHA256 binding required")
        return self.add(value['path'], value['sha256'], **kwargs)

    def read(self, path):
        return json.loads(self.path(path).read_text())

    def domain(self, path, digest):
        path = self.add(path, digest)
        domain = self.read(path)
        if domain.get('schema') != 'Task10BPhase2Domain/v1':
            raise ValueError("Canonical Phase2 domain required")
        ledger = domain.get('all_slot_ledger')
        if not isinstance(ledger, dict) or not {'path', 'sha256'} <= ledger.keys():
            raise ValueError("Domain must bind its complete all-slot disposition ledger")
        # This is a domain-local member, unlike project-relative packet fields.
        self.add(ledger['path'], ledger['sha256'], relative_to=path.parent)
        return domain

    def model(self, row):
        path = self.add(row['source_model_path'], row['source_model_sha256'])
        if any('file' in node.attrib or node.tag == 'include'
               for node in ET.fromstring(path.read_text()).iter()):
            raise ValueError("Self-contained accepted model required")

    def projection(self, path, catalog):
        if 'target_catalog_projection' not in catalog:
            return
        from .phase2_target_projection import validate
        validate(path)
        receipt = catalog['target_catalog_projection']
        self.add(path.parent/'projection.json')
        for key in ('native_catalog', 'generation_execution', 'generation_packet',
                    'terminal_accounting', 'alias_probe', 'nonce', 'archived_plan', 'archived_generator'):
            item = receipt[key]
            self.add(item['relative_name'], item['sha256'])
        packet_path = self.path(receipt['generation_packet']['relative_name'])
        packet = self.read(packet_path)
        for name, digest in packet['files'].items():
            # Explicit immutable packet inventory only, never generic ancestry.
            self.add(packet_path.parent/'source_snapshot'/name, digest,
                     provenance_role='verified_target_generation_archive')

    def catalog(self, binding, *, packed, recipes=None):
        from .phase2_runtime_pack import VERSION as STORAGE, verify_rejection
        path = self.bound(binding)
        catalog = self.read(path)
        rows = catalog.get('cells', [])
        if (catalog.get('schema') != CATALOG or catalog.get('complete') is not True
                or not 1 <= len(rows) <= 4096
                or len({r['candidate_id'] for r in rows}) != len(rows)
                or (packed and (catalog.get('runtime_storage_version') != STORAGE
                    or catalog.get('runtime_pack_complete') is not True))
                or (not packed and catalog.get('runtime_storage_version') is not None)):
            raise ValueError("Complete exact versioned candidate catalog required")
        if packed:
            ancestor = self.add(catalog['unpacked_catalog_path'], catalog['unpacked_catalog_sha256'],
                                provenance_role='unpacked_catalog_path')
            original = self.read(ancestor)
            if original.get('runtime_storage_version') is not None:
                raise ValueError("Recursive runtime packing is not supported")
            self.projection(ancestor, original)
        else:
            self.projection(path, catalog)
        for row in rows:
            if (row.get('kind') == 'original_control' or row.get('start_mode', 'family_guard') != 'family_guard'
                    or row.get('split') not in ('train', 'validation')
                    or (recipes is not None and row.get('recipe_sha256') not in recipes)):
                raise ValueError("Exact isolated train/validation candidate recipes required")
            if row.get('reference_path'):
                self.model(row)
                ref = self.add(row['reference_path'], row['reference_sha256'])
                legacy = self.add(row['legacy_runtime_path'], row['legacy_runtime_sha256'])
                if ref != legacy or row['reference_sha256'] != row['legacy_runtime_sha256']:
                    raise ValueError("State and feedforward must use one exact archive")
                if packed:
                    self.add(row['unpacked_reference_path'], row['unpacked_reference_sha256'],
                             provenance_role='unpacked_reference_path')
            else:
                with declared_packing_paths():
                    rejection, evidence = verify_rejection(row)
                self.add(rejection, row['generation_failure_sha256'])
                if packed:
                    self.add(row['unpacked_generation_failure_path'], row['unpacked_generation_failure_sha256'],
                             provenance_role='unpacked_generation_failure_path')
        return catalog

    def finish(self):
        for name, digest in self.files.items():
            if sha256_file(self.path(name)) != digest:
                raise ValueError("Input changed during closure")
        for name, item in self.provenance.items():
            if sha256_file(self.path(name)) != item['sha256']:
                raise ValueError("Preserved ancestry changed during closure")
        return dict(sorted(self.files.items()))


def runtime_pack_files(collector, path):
    from .phase2_runtime_pack import VERSION as STORAGE, CAP, pool_inventory
    plan = collector.read(collector.add(path))
    if (plan.get('schema') != 'task10b_phase2_runtime_pack_plan_v1'
            or plan.get('storage_version') != STORAGE
            or type(plan.get('byte_cap')) is not int or not 0 < plan['byte_cap'] <= CAP
            or not 1 <= len(plan.get('catalogs', [])) <= 64
            or not 1 <= len(plan.get('recipes', [])) <= 16
            or len(plan.get('previous_packed_catalogs', [])) > 64):
        raise ValueError("Finite exact runtime packing plan required")
    recipes = {}
    for item in plan['recipes']:
        recipe = collector.read(collector.bound(item))
        if recipe.get('schema') != 'task10b_phase2_target_recipe_v1' or recipe.get('start_mode') != 'family_guard':
            raise ValueError("Isolated target recipe required")
        recipes[item['sha256']] = recipe
    catalogs = [collector.catalog(item, packed=False, recipes=recipes) for item in plan['catalogs']]
    if sum(len(c['cells']) for c in catalogs) > 4096:
        raise ValueError("At most4096 explicitly planned packing cells")
    previous = plan.get('previous_packed_catalogs', [])
    for item in previous:
        collector.catalog(item, packed=True)
    with declared_packing_paths():
        pool_inventory(previous, cap=plan['byte_cap'])


def grouped_preflight(path, digest, root):
    """Existing scientific admission/preflight only; no native environment creation."""
    from .phase2_grouped_learning import (load_configuration, select_training_rows,
                                          preflight_training_references)
    from .phase2_validation_callback import GroupedValidationCallback
    declaration, items, budget, config, release = load_configuration(path, digest)
    if config.get('start_mode', 'family_guard') != 'family_guard':
        raise ValueError("This launcher covers isolated family guards only")
    read = lambda key: json.loads(Path(items[key]['path']).read_text())
    rows, _ = select_training_rows(declaration, read('catalog'), read('domain'), read('target_geometry'))
    preflight_training_references(rows, resident_limit=declaration.resident_reference_bytes)
    GroupedValidationCallback.from_configuration(items['validation_configuration']['path'],
        items['validation_configuration']['sha256'], declaration, release,
        Path(root)/'outputs/task10b/preflight_only_no_output_created')


def grouped_files(collector, path):
    path = collector.add(path)
    config = collector.read(path)
    if config.get('schema') != 'task10b_phase2_grouped_training_configuration_v1':
        raise ValueError("Isolated grouped training configuration required")
    names = ('declaration', 'catalog', 'domain', 'target_geometry', 'predicate', 'mask',
             'recipe', 'validation_configuration')
    bound = {name: collector.bound(config[name]) for name in names}
    collector.domain(bound['domain'], config['domain']['sha256'])
    recipe_sha = config['recipe']['sha256']
    collector.catalog(config['catalog'], packed=True, recipes={recipe_sha})
    validation = collector.read(bound['validation_configuration'])
    if validation.get('schema') != 'task10b_phase2_grouped_validation_configuration_v1':
        raise ValueError("Exact held-out callback configuration required")
    val = {key: collector.bound(validation[key]) for key in
           ('descriptor', 'declaration_template', 'manifest', 'mask', 'predicate', 'domain')}
    if val['domain'] != bound['domain'] or validation['domain']['sha256'] != config['domain']['sha256']:
        raise ValueError("Validation callback must bind the same canonical domain")
    manifest = collector.read(val['manifest'])
    if manifest.get('schema') != 'task10b_phase2_expanded_manifest_v1' or manifest.get('recipe_sha256') != recipe_sha:
        raise ValueError("One common recipe and exact expanded validation manifest required")
    for key in ('domain', 'target_geometry'):
        selected = collector.add(manifest[key+'_path'], manifest[key+'_sha256'])
        if selected != bound[key] or manifest[key+'_sha256'] != config[key]['sha256']:
            raise ValueError("Validation geometry/domain identity differs")
    collector.catalog(dict(path=manifest['catalog_path'], sha256=manifest['catalog_sha256']),
                      packed=True, recipes={recipe_sha})
    for entry in manifest['trials']:
        collector.add(entry['start_bank_path'], entry['start_bank_sha256'])
    grouped_preflight(path, sha256_file(path), collector.root)


def files(mode, inputs, *, root=PROJECT_ROOT, provenance_files=None):
    if mode not in MODES or set(inputs) != ({'plan'} if mode == 'runtime-pack' else {'configuration'}):
        raise ValueError("Exact supported pipeline mode/inputs required")
    collector = Inputs(root, provenance_files)
    if mode == 'runtime-pack':
        runtime_pack_files(collector, inputs['plan'])
    else:
        grouped_files(collector, inputs['configuration'])
    return collector.finish()
