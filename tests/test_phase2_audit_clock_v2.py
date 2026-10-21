"""Independent direct-clock arithmetic and unchanged static auditor ancestry."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'scripts/task10b_phase2_audit.py'
VERSIONED=ROOT/'scripts/task10b_phase2_audit_clock_v2.py'
PARENT_SHA='86de2f9e1260c98bbc5379c6156b4840b284b79342981dde3ba06c384aa824dd'
OLD="np.ceil(row['episode_duration_s']/.005)"
NEW="np.ceil(row['episode_duration_s']/.005-1e-10)"


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


audit=module(VERSIONED,'independent_audit_clock_v2')


def horizon_expression(source):
    matches=[node.value for node in ast.walk(ast.parse(source))
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.IfExp)
        and any(isinstance(target,ast.Name) and target.id=='steps' for target in node.targets)]
    assert len(matches)==1
    return compile(ast.Expression(matches[0]),'<independent-clock-expression>','eval')


OLD_EXPRESSION=horizon_expression(PARENT.read_text())
NEW_EXPRESSION=horizon_expression(VERSIONED.read_text())


def steps(duration,kind='original_control',expression=NEW_EXPRESSION):
    return eval(expression,{'np':np,'int':int},{'row':dict(episode_duration_s=duration,kind=kind)})


def without_declared_metadata(source):
    start=source.index('# BEGIN INDEPENDENT CLOCK V2 ANCESTRY\n')
    stop=source.index('# END INDEPENDENT CLOCK V2 ANCESTRY\n',start)+len('# END INDEPENDENT CLOCK V2 ANCESTRY\n\n\n')
    source=source[:start]+source[stop:]
    source=source.replace("        result['clock_auditor_version']=clock_version_receipt()\n",'')
    source=source.replace("    result['clock_auditor_version']=clock_version_receipt()\n",'')
    return source


def test_original_auditor_bytes_are_unchanged_and_versioned_source_is_static_copy():
    old=PARENT.read_bytes();new=VERSIONED.read_text()
    assert hashlib.sha256(old).hexdigest()==PARENT_SHA
    algorithm=without_declared_metadata(new)
    assert algorithm.count(NEW)==1
    assert algorithm.replace(NEW,OLD)==old.decode()
    assert ast.dump(ast.parse(algorithm.replace(NEW,OLD)))==ast.dump(ast.parse(old))


@pytest.mark.parametrize('ticks',[1,96,960,967,1800])
@pytest.mark.parametrize('direction',[-np.inf,None,np.inf])
def test_exact_five_ms_lattice_and_adjacent_float64_values_have_same_tick_count(ticks,direction):
    duration=ticks*.005
    if direction is not None:duration=np.nextafter(duration,direction)
    assert steps(duration)==ticks


@pytest.mark.parametrize('duration',[.006,.333,4.800000000005,4.800001,4.833333333333333,5.201])
def test_nonlattice_durations_keep_the_original_ceiling(duration):
    assert steps(duration)==steps(duration,expression=OLD_EXPRESSION)


@pytest.mark.parametrize('duration',[4.8,4.800000000000001,4.833333333333333])
def test_retimed_actor_boundary_padding_is_unchanged(duration):
    assert steps(duration,'uniform_timing')==steps(duration,'uniform_timing',OLD_EXPRESSION)


def test_cross_roundoff_bug_changes_only_the_spurious_extra_requested_frame():
    duration=4.800000000000001
    assert steps(duration,expression=OLD_EXPRESSION)==961
    assert steps(duration)==960
    saved_frames=961
    assert len(np.arange(steps(duration)+1)*.005)==saved_frames
    assert len(np.arange(steps(duration,expression=OLD_EXPRESSION)+1)*.005)!=saved_frames


def test_version_receipt_binds_preserved_parent_and_current_independent_source():
    receipt=audit.clock_version_receipt()
    assert receipt['version']=='task10b_phase2_independent_clock_v2'
    assert receipt['parent_source']['sha256']==PARENT_SHA
    assert receipt['source_sha256']==hashlib.sha256(VERSIONED.read_bytes()).hexdigest()
    assert receipt['new_physics_steps']==0 and not receipt['learning_helper_imported']
    assert receipt['nonoriginal_actor_padding_unchanged'] and receipt['event_clocks_unchanged']
    assert not receipt['old_failed_attempts_reclassified']


def test_version_receipt_fails_closed_on_parent_identity_mismatch(monkeypatch):
    monkeypatch.setattr(audit,'AUDIT_CLOCK_PARENT_SHA256','0'*64)
    with pytest.raises(ValueError,match='parent|v1 auditor'):audit.clock_version_receipt()


def test_direct_clock_expression_imports_no_learning_helper():
    tree=ast.parse(VERSIONED.read_text())
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    imports += [name.name for node in ast.walk(tree) if isinstance(node,ast.Import) for name in node.names]
    assert not any(name and ('phase2_learning' in name or 'phase2_d0_clock' in name) for name in imports)
    expression=ast.dump(next(node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and isinstance(node.value,ast.IfExp) and any(isinstance(t,ast.Name) and t.id=='steps' for t in node.targets)))
    assert "value=1e-10" in expression and "attr='ceil'" in expression
