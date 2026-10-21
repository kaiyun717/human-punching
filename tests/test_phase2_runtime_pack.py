"""Runtime storage identity/bounds only; no native execution is inferred."""
import json
from pathlib import Path

import numpy as np
import pytest

from human_punching import phase2_runtime_pack as packing
from human_punching import phase2_target_catalog
from human_punching.wbc.provenance import sha256_file


def encoded(path,value):
    path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=sha256_file(path))


def arrays():
    n=9;q=np.zeros((n,55));q[:,3]=1.;q[:,7]=np.linspace(.001,.019,n)
    return dict(t=np.arange(n)*.005,qpos=q,qvel=np.zeros((n,54)),torque=np.zeros((n,48)),
        head_center=np.array([1.234567890123,.1,1.2]),active_hand_index=np.array(0),
        guard_hold_start=np.array(.04),qacc=np.zeros((n,54)),glove_pos=np.zeros((n,2,3)),
        source_time_s=np.arange(n)*.005,phase_schedule_json=np.array('{"prep":0.02}'),
        guard_center_chest_rel=np.ones((2,3))*1.234567890123)


@pytest.fixture
def example(tmp_path,monkeypatch):
    monkeypatch.setattr(phase2_target_catalog,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(packing,"writable_path",lambda path:Path(path).resolve())
    source=tmp_path/"source.npz";np.savez_compressed(source,**arrays())
    recipe=encoded(tmp_path/"recipe.json",dict(schema="task10b_phase2_shared_recipe_v2",family="jab"))
    row=dict(candidate_id="jab_group_1_0600_start_0",kind="shared_jab_target_reference",
        family="jab",recipe_sha256=recipe["sha256"],reference_path=str(source),
        legacy_runtime_path=str(source),reference_sha256=sha256_file(source),legacy_runtime_sha256=sha256_file(source),kinematic_admission=True)
    catalog=encoded(tmp_path/"catalog.json",dict(schema="task10b_phase2_reference_catalog_v1",
        complete=True,recipe_sha256=recipe["sha256"],cells=[row]))
    plan=dict(schema="task10b_phase2_runtime_pack_plan_v1",storage_version=packing.VERSION,
        byte_cap=2**20,catalogs=[catalog],recipes=[recipe],previous_packed_catalogs=[])
    return tmp_path,plan,row


def test_native_precision_only_exact_target_clock_and_small_guard_metadata():
    data=arrays();packed,receipt=packing.runtime_arrays(data)
    for key in packing.EXECUTED:assert packed[key].dtype==np.float32
    for key in ("head_center","t","guard_center_chest_rel"):
        assert np.array_equal(packed[key],data[key]) and packed[key].dtype==np.float64
    assert set(receipt["omitted_arrays"])=={"qacc","glove_pos","source_time_s"}
    assert not receipt["physical_qualification"] and not receipt["native_execution_parity_assumed"]
    assert data["qpos"].dtype==np.float64


@pytest.mark.parametrize("change",["complex","nan","overflow","clock","shape","unknown_large","object"])
def test_unsupported_or_nonfinite_arrays_cannot_be_packed(change):
    data=arrays()
    if change=="complex":data["qpos"]=data["qpos"].astype(complex)
    elif change=="nan":data["qacc"][0,0]=np.nan
    elif change=="overflow":data["qvel"][0,0]=1.e100
    elif change=="clock":data["t"][2]+=.00001
    elif change=="shape":data["qpos"]=data["qpos"][:,:54]
    elif change=="object":data["extra"]=np.array([object()])
    else:data["unknown"]=np.zeros(1024)
    with np.errstate(over="ignore"),pytest.raises(ValueError):packing.runtime_arrays(data)


def run(example,**changes):
    folder,plan,_=example;plan={**plan,**changes}
    path=folder/"plan.json";encoded(path,plan)
    return packing.pack(path,folder/"run",wall_seconds=30.)


def test_packed_catalog_new_recipe_source_preserved_and_pool_inventory(example):
    folder,plan,row=example;original=Path(row["reference_path"]).read_bytes()
    result=run(example)
    assert result["status"]=="packed" and result["completed_cells"]==1
    assert Path(row["reference_path"]).read_bytes()==original
    catalog=json.loads(Path(result["catalogs"][0]["path"]).read_text());packed=catalog["cells"][0]
    assert packed["recipe_sha256"]!=row["recipe_sha256"]
    assert packed["unpacked_reference_sha256"]==row["reference_sha256"]
    assert packed["unpacked_recipe_sha256"]==row["recipe_sha256"]
    with np.load(packed["reference_path"],allow_pickle=False) as archive:
        assert archive["qpos"].dtype==np.float32
        assert np.array_equal(archive["head_center"],arrays()["head_center"])
    inventory=packing.pool_inventory(result["catalogs"],cap=plan["byte_cap"])
    assert inventory["uncompressed_array_bytes"]==result["active_pool_uncompressed_array_bytes"]
    assert inventory["unique_archives"]==1


def test_combined_previous_pool_counts_before_writing_new_arrays(example):
    folder,plan,row=example
    first=run(example)
    # A second physical archive with identical contents is still a separate
    # resident registration; the conservative pool cannot silently double it.
    plan["previous_packed_catalogs"]=first["catalogs"]
    plan["byte_cap"]=first["active_pool_uncompressed_array_bytes"]+1
    path=folder/"plan2.json";encoded(path,plan)
    result=packing.pack(path,folder/"run2",wall_seconds=30.)
    assert result["status"]=="failed" and result["completed_cells"]==0
    assert result["cell_ledger"]==[dict(catalog=0,candidate_id=row["candidate_id"],status="unattempted")]
    assert not json.loads(Path(result["catalogs"][0]["path"]).read_text())["complete"]


def test_negative_generation_evidence_preserved_and_tamper_rejected(example):
    folder,plan,row=example
    evidence=encoded(folder/"failure.json",dict(rejection_reasons=["unreachable"],recipe_sha256=row["recipe_sha256"]))
    negative={k:v for k,v in row.items() if k not in ("reference_path","legacy_runtime_path","legacy_runtime_sha256")}
    negative.update(kinematic_admission=False,reference_identity_kind="generation_failure_receipt",
        reference_sha256=evidence["sha256"],generation_failure_path=evidence["path"],
        generation_failure_sha256=evidence["sha256"],rejection_reasons=["unreachable"])
    plan["catalogs"]=[encoded(folder/"negative.json",dict(schema="task10b_phase2_reference_catalog_v1",
        complete=True,cells=[negative]))]
    result=run(example);assert result["status"]=="packed"
    assert packing.pool_inventory(result["catalogs"])["uncompressed_array_bytes"]==0
    packed=json.loads(Path(result["catalogs"][0]["path"]).read_text())["cells"][0]
    wrapper=json.loads(Path(packed["generation_failure_path"]).read_text())
    assert wrapper["recipe_sha256"]==packed["recipe_sha256"]!=row["recipe_sha256"]
    assert wrapper["unpacked_generation_failure_sha256"]==evidence["sha256"]
    assert json.loads(Path(evidence["path"]).read_text())["recipe_sha256"]==row["recipe_sha256"]
    Path(evidence["path"]).write_text('{"tampered":true}')
    with pytest.raises(ValueError,match="hash"):packing.pool_inventory(result["catalogs"])


def test_original_controls_and_partial_source_catalogs_cannot_gain_new_recipe(example):
    folder,plan,row=example
    row["kind"]="original_control"
    plan["catalogs"]=[encoded(folder/"control.json",dict(schema="task10b_phase2_reference_catalog_v1",complete=True,cells=[row]))]
    result=run(example);assert result["status"]=="failed" and not result["completed_cells"]


def test_source_identity_checked_before_archived_data_loading(example):
    _,_,row=example;Path(row["reference_path"]).write_bytes(b"changed")
    result=run(example)
    assert result["status"]=="failed" and "hash" in result["exception"]["message"]


@pytest.mark.parametrize("field,value",[("active_hand_index",np.array(2)),
    ("active_hand_index",np.array(True)),("guard_hold_start",np.array(.1))])
def test_runtime_hand_and_guard_metadata_are_explicit(field,value):
    data=arrays();data[field]=value
    with pytest.raises(ValueError):packing.runtime_arrays(data)


def test_partial_catalog_is_rejected_before_output_creation(example):
    folder,plan,row=example
    plan["catalogs"]=[encoded(folder/"partial.json",dict(schema="task10b_phase2_reference_catalog_v1",
        complete=False,cells=[row]))]
    with pytest.raises(ValueError,match="Complete"):run(example)
    assert not (folder/"run").exists()


def test_legacy_feedforward_identity_cannot_be_silently_changed(example):
    folder,plan,row=example
    legacy=folder/"different_feedforward.npz";data=arrays();data["torque"][:]=1.
    np.savez_compressed(legacy,**data)
    row.update(legacy_runtime_path=str(legacy),legacy_runtime_sha256=sha256_file(legacy))
    plan["catalogs"]=[encoded(folder/"different_catalog.json",dict(schema="task10b_phase2_reference_catalog_v1",complete=True,cells=[row]))]
    result=run(example)
    assert result["status"]=="failed" and result["completed_cells"]==0 and "feedforward" in result["exception"]["message"]


def test_same_source_bytes_at_distinct_owned_paths_dedupe_without_name_collision(example):
    folder,plan,row=example;duplicate=folder/"same_bytes.npz"
    duplicate.write_bytes(Path(row["reference_path"]).read_bytes())
    second={**row,"candidate_id":"jab_duplicate_bytes", "reference_path":str(duplicate),"legacy_runtime_path":str(duplicate)}
    plan["catalogs"]=[encoded(folder/"two_cells.json",dict(schema="task10b_phase2_reference_catalog_v1",complete=True,cells=[row,second]))]
    result=run(example)
    assert result["status"]=="packed" and result["completed_cells"]==2
    rows=json.loads(Path(result["catalogs"][0]["path"]).read_text())["cells"]
    assert rows[0]["reference_path"]==rows[1]["reference_path"]
    assert rows[0]["unpacked_reference_path"]!=rows[1]["unpacked_reference_path"]
    assert packing.pool_inventory(result["catalogs"])["unique_archives"]==1
