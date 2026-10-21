"""Exact sequential full-catalog ancestry for unchanged runtime packing/PPO.

The two existing scientific admission paths remain unchanged. Only the explicit
projection provenance edge differs from the older target-reference generator.
"""
from __future__ import annotations
from pathlib import Path
from . import phase2_pipeline_inputs_v2 as parent
from .phase2_sequential_projection_v1 import verify_projection,files as projection_files,FULL_SCHEMA
from .wbc.provenance import PROJECT_ROOT

VERSION='task10b_phase2_sequential_pipeline_inputs_v1'
MODES=parent.MODES


class Inputs(parent.Inputs):
    def projection(self,path,catalog):
        if 'sequential_catalog_projection' not in catalog:
            raise ValueError('Reviewed full sequential catalog projection required')
        if any(k in catalog for k in ('target_catalog_projection','timing_control_projection')):
            raise ValueError('Projection ancestry cannot masquerade as another schema')
        view,receipt=verify_projection(path)
        if view!=catalog or receipt['packet_schema']!=FULL_SCHEMA:
            raise ValueError('Exact full64/24 sequential source projection required')
        for name,digest in projection_files(path).items():
            # Immutable ancestor archives are verified documentary inputs. They
            # stay in their original packets rather than recursively repacking.
            role=('verified_sequential_generation_archive' if 'source_snapshot' in Path(name).parts else None)
            self.add(name,digest,provenance_role=role)


def files(mode,inputs,*,root=PROJECT_ROOT,provenance_files=None):
    original=parent.Inputs
    try:
        parent.Inputs=Inputs
        result=parent.files(mode,inputs,root=root,provenance_files=provenance_files)
    finally:parent.Inputs=original
    if mode=='runtime-pack':
        collector=Inputs(root,provenance_files)
        plan=collector.read(inputs['plan'])
        audits=plan.get('sequential_reference_audits',[])
        if len(plan['catalogs'])!=2 or len(audits)!=2 or len(plan['recipes'])!=1:
            raise ValueError('One common recipe and independently audited full64/24 catalogs required')
        for index,(binding,audit_binding) in enumerate(zip(plan['catalogs'],audits,strict=True)):
            view=collector.read(collector.bound(binding));audit=collector.read(collector.bound(audit_binding))
            split,count=(('train',64) if index==0 else ('validation',24))
            if (len(view['cells'])!=count or any(r['split']!=split for r in view['cells'])
                    or audit.get('schema')!='task10b_phase2_saved_target_reference_independent_audit_v1'
                    or audit.get('passed') is not True or audit.get('physics_steps')!=0
                    or audit.get('manifest_sha256')!=view['unpacked_catalog_sha256']
                    or len(audit.get('cells',[]))!=count
                    or any(r.get('all_checks_passed') is not True for r in audit['cells'])):
                raise ValueError('Exact successful saved-only full64/24 audit required')
        result.update(collector.finish())
    elif mode=='grouped-pilot':
        collector=Inputs(root,provenance_files)
        config=collector.read(inputs['configuration'])
        collector.bound(config['configuration_factory'])
        selection=collector.read(collector.bound(config['sequential_selection']))
        if selection.get('schema')!='task10b_phase2_sequential_full_grouped_selection_v1':
            raise ValueError('Exact sequential full grouped selection required')
        for item in (selection['original_selection'],selection['full_recipe'],
                *selection['full_plans'].values(),*selection['projected_catalogs'].values(),
                *selection['saved_audits'].values()):
            collector.bound(item)
        result.update(collector.finish())
    return dict(sorted(result.items()))
