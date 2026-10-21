"""Full native-state/parity-bound held-out validation for shared PPO."""
from __future__ import annotations

from .phase2_validation_callback import GroupedValidationCallback
from .phase2_expanded_evaluation import (EvaluationLimits,load_declaration,read_json,owned_file,
    declared_mask_decisions,validate_geometry_bindings)
from .phase2_grouped_learning import validate_validation_descriptor
from .phase2_shared_evaluation_v2 import SharedEvaluationProtocol
from .phase2_learning import family_arguments
from .wbc.provenance import writable_path,sha256_file


class SharedValidationCallback(GroupedValidationCallback):
    @classmethod
    def from_configuration(cls,path,expected_sha256,training_declaration,release,output):
        config=read_json(path,expected_sha256)
        if config.get('schema')!='task10b_phase2_shared_validation_configuration_v2':
            raise ValueError("Explicit shared native-state physical validation configuration required")
        inputs={key:owned_file(config[key]['path'],config[key]['sha256']) for key in
            ('descriptor','declaration_template','manifest','mask','predicate')}
        descriptor=read_json(inputs['descriptor']);declaration=load_declaration(inputs['declaration_template'])
        manifest=read_json(inputs['manifest'])
        if (declaration.start_mode!='shared_jab_saved_state' or declaration.split!='validation'
                or declaration.family!=training_declaration.family or declaration.seed!=training_declaration.seed
                or set(declaration.command_groups)!=set(descriptor['command_groups'])
                or declaration.trained_timings_s!=training_declaration.trained_timings_s
                or declaration.interpolation_timing_s!=training_declaration.interpolation_timing_s
                or declaration.predicate_sha256!=training_declaration.predicate_sha256
                or declaration.declared_mask_sha256!=training_declaration.declared_mask_sha256
                or {b.trial_id for b in declaration.bindings}!=set(descriptor['trial_ids'])
                or sha256_file(inputs['predicate'])!=declaration.predicate_sha256
                or sha256_file(inputs['mask'])!=declaration.declared_mask_sha256
                or manifest['domain_sha256']!=training_declaration.domain_sha256
                or manifest['target_geometry_sha256']!=training_declaration.target_geometry_sha256
                or manifest['recipe_sha256']!=training_declaration.recipe_sha256):
            raise ValueError("Shared validation physical template differs from the frozen training recipe/domain")
        domain=read_json(manifest['domain_path'],manifest['domain_sha256'])
        targets=read_json(manifest['target_geometry_path'],manifest['target_geometry_sha256'])
        catalog=read_json(manifest['catalog_path'],manifest['catalog_sha256'])
        if catalog.get('schema')!='task10b_phase2_reference_catalog_v1':raise ValueError("Canonical shared catalog required")
        validate_validation_descriptor(training_declaration,descriptor,domain)
        decisions=declared_mask_decisions(declaration,read_json(inputs['mask']))
        limits=EvaluationLimits(14400.,maximum_worlds=config.get('maximum_worlds',16),
            resident_reference_bytes=min(training_declaration.resident_reference_bytes,256*1024**2))
        protocol=SharedEvaluationProtocol()
        entries,rows,starts,scope=protocol.preflight(declaration,manifest,catalog,limits=limits,mask_decisions=decisions,
            release=release,route=dict(recipe_sha256=training_declaration.recipe_sha256))
        scope.update(validate_geometry_bindings(declaration,manifest,domain,targets,entries,rows))
        protocol.validate_starts(declaration,domain,starts)
        for entry in entries.values():
            if rows[entry['candidate_id']].get('recipe_sha256')!=training_declaration.recipe_sha256:
                raise ValueError("Shared validation reference belongs to a different recipe")
        self=cls();self.descriptor=descriptor;self.declaration=declaration
        self.args,self.paths=family_arguments(release,declaration.family)
        self.inputs=inputs;self.catalog=catalog;self.entries=entries;self.rows=rows;self.starts=starts
        self.manifest=manifest;self.decisions=decisions;self.limits=limits;self.scope=scope
        self.predicate=read_json(inputs['predicate']);self.targets={r['candidate_identity']:r for r in targets['geometry_candidates']}
        self.output=writable_path(output);self.training_declaration=training_declaration
        self.source_configuration=dict(path=str(owned_file(path)),sha256=expected_sha256)
        self.validation_protocol=protocol
        return self
