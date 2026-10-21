from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from human_punching import phase2_native_state_diagnostic as diagnostic


def vector():
    return np.arange(410, dtype=np.float64)[None]


def test_complete_layout_preserves_every_canonical_value():
    parts = diagnostic.split_integration(vector())
    assert list(parts) == [name for name, _ in diagnostic.LAYOUT]
    assert np.array_equal(np.concatenate(list(parts.values()), axis=1), vector())
    assert parts["qacc_warmstart"].shape == (1, 54)
    assert parts["xfrc_applied"].shape == (1, 144)


@pytest.mark.parametrize("bad", [np.zeros((410,), np.float64), np.zeros((2,410), np.float64),
                                  np.zeros((1,409), np.float64), np.zeros((1,410), np.float32),
                                  np.full((1,410), np.nan, np.float64)])
def test_incomplete_or_noncanonical_state_is_rejected(bad):
    with pytest.raises(ValueError): diagnostic.split_integration(bad)


def test_one_ulp_remains_an_exact_failure_and_is_localized():
    a = np.ones((1, 54), np.float32); b = a.copy()
    b[0, 17] = np.nextafter(b[0,17], np.float32(2))
    item = diagnostic.difference(a, b)
    assert item["exact_equal"] is False
    assert item["maximum_float32_ulp_distance"] == 1
    assert item["first_unequal_index"] == [0,17]
    assert item["unequal_values"] == 1


@pytest.mark.parametrize("value,direction", [(-1.,-2.),(-1.,0.),(0.,1.),(1.,2.)])
def test_ulp_diagnostic_has_consistent_sign_order(value, direction):
    a = np.array([value], np.float32)
    b = np.nextafter(a, np.float32(direction))
    assert diagnostic.difference(a,b)["maximum_float32_ulp_distance"] == 1


def test_zero_sign_bytes_are_visible_without_changing_parent_equality():
    item = diagnostic.difference(np.array([0.]), np.array([-0.]))
    assert item["exact_equal"] is True  # Same np.array_equal gate as preserved parent.
    assert item["bytes_equal"] is False


def test_missing_fields_and_different_storage_are_not_guessed():
    item = diagnostic.compare_snapshots({"native.integration": vector(), "x": np.zeros(1)},
                                        {"native.integration": vector(), "y": np.zeros(1)})
    assert item["exact_native_equal"] is True
    assert item["unequal_fields"] == ["x", "y"]
    assert diagnostic.difference(np.zeros(1,np.float32),np.zeros(1,np.float64))["same_schema"] is False


def test_warmstart_only_difference_is_distinct_from_physical_difference():
    a = {"native.integration": vector(), **{"native."+k:v for k,v in diagnostic.split_integration(vector()).items()}}
    b = {k:v.copy() for k,v in a.items()}
    b["native.qacc_warmstart"][0,0] += 1
    b["native.integration"][0,110] += 1
    result = diagnostic.compare_snapshots(a,b)
    assert result["exact_native_equal"] is False
    assert set(result["unequal_fields"]) == {"native.integration", "native.qacc_warmstart"}
    assert result["fields"]["native.qpos"]["exact_equal"] is True
    assert result["fields"]["native.qvel"]["exact_equal"] is True


class Backend:
    def __init__(self): self.counts = dict(reset=0, forward=0, step=0); self.fail=False
    def reset(self, *args, **kwargs): self.counts["reset"]+=1; return "reset"
    def forward(self, *args, **kwargs): self.counts["forward"]+=1; return "forward"
    def step(self, *args, **kwargs):
        self.counts["step"]+=1
        if self.fail: raise RuntimeError("original backend failed")
        return "step"


def recorder(tmp_path, capture=None):
    return diagnostic.RetainedRecorder(tmp_path, capture or (lambda env: {"native.integration": vector()}))


def test_composition_calls_original_backend_once_without_extra_intervals(tmp_path):
    backend = Backend(); env=object(); writer=recorder(tmp_path)
    proxy=diagnostic.ObservedSimulation(backend,env,writer,"source")
    assert proxy.reset()=="reset" and proxy.forward()=="forward" and proxy.step()=="step"
    assert backend.counts == dict(reset=1,forward=1,step=1)
    assert [row["label"] for row in writer.snapshots] == ["reset_before","reset_after","forward_before","forward_after","step_before","step_after"]
    assert len(writer.calls)==3 and all(item["call_returned"] for item in writer.calls)
    for item in writer.snapshots:
        path=Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
        with np.load(path,allow_pickle=False) as data: assert np.array_equal(data["native.integration"],vector())


def test_original_backend_failure_remains_failed_and_preserved(tmp_path):
    backend=Backend(); backend.fail=True; writer=recorder(tmp_path)
    proxy=diagnostic.ObservedSimulation(backend,object(),writer,"source")
    with pytest.raises(RuntimeError, match="original backend failed"): proxy.step()
    terminal=json.loads((tmp_path/"call_000_terminal.json").read_text())
    assert terminal["call_returned"] is False and terminal["exception"]["message"]=="original backend failed"
    assert backend.counts["step"]==1 and len(writer.snapshots)==1


def test_completed_backend_call_is_not_lost_when_after_capture_fails(tmp_path):
    count=0
    def capture(env):
        nonlocal count
        count+=1
        if count==2: raise ValueError("diagnostic read failed")
        return {"native.integration":vector()}
    backend=Backend(); writer=recorder(tmp_path,capture)
    proxy=diagnostic.ObservedSimulation(backend,object(),writer,"source")
    with pytest.raises(ValueError,match="diagnostic read failed"):proxy.step()
    assert backend.counts["step"]==1
    assert json.loads((tmp_path/"call_000_returned.json").read_text())["call_returned"] is True
    assert len(writer.snapshots)==1 and writer.calls[0]["call_returned"] is True


def test_object_payload_cannot_enter_npz(tmp_path):
    writer=recorder(tmp_path,lambda env:{"bad":np.array([object()],object)})
    with pytest.raises(ValueError,match="numeric"):writer.observe("source","bad",object())
    assert not list(tmp_path.iterdir())


def test_snapshot_and_total_memory_caps_precede_writes(tmp_path):
    writer=recorder(tmp_path,lambda env:{"huge":np.empty(diagnostic.MAX_SNAPSHOT_BYTES+1,np.uint8)})
    with pytest.raises(ValueError,match="byte cap"):writer.observe("source","huge",object())
    assert not list(tmp_path.iterdir())
    writer=recorder(tmp_path);writer.total_array_bytes=diagnostic.MAX_TOTAL_BYTES
    with pytest.raises(ValueError,match="byte cap"):writer.observe("source","full",object())
    assert not list(tmp_path.iterdir())


def test_snapshot_count_cap_and_existing_files_fail_closed(tmp_path):
    writer=recorder(tmp_path);writer.snapshots=[{}]*diagnostic.MAX_SNAPSHOTS
    with pytest.raises(ValueError,match="count exhausted"):writer.observe("source","full",object())
    writer=recorder(tmp_path);(tmp_path/"snapshot_000.npz").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):writer.observe("source","duplicate",object())
    assert (tmp_path/"snapshot_000.npz").read_bytes()==b"preserve"


def test_original_restore_source_is_preserved():
    parent=Path(diagnostic.__file__).with_name("phase2_native_state.py")
    assert diagnostic.digest(parent)==diagnostic.PARENT_SHA256


def test_pinned_warp_named_metadata_inventory_exists_without_native_execution():
    from mujoco_warp._src.types import Data, Contact, Constraint
    assert set(("solver_niter","ne","nf","nl","nefc","nacon","nisland","qacc","qacc_smooth",
                "qfrc_constraint","qfrc_smooth","actuator_force","qfrc_actuator","overflow")) <= set(Data.__dataclass_fields__)
    assert set(("geom","dim","efc_address","worldid","geomcollisionid","type","dist","pos","frame")) <= set(Contact.__dataclass_fields__)
    assert set(("type","id","state","pos","D","vel","aref","force","island")) <= set(Constraint.__dataclass_fields__)


def test_complete_planned_sequence_fits_predeclared_snapshot_and_byte_limits():
    # Two original resets+their forwards, one restore forward, two common
    # forwards and twelve explicit steps:19calls*2 snapshots, plus16 checkpoints.
    assert 2*(2+5+12)+16 == 54
    assert diagnostic.MAX_SNAPSHOTS >= 54
    assert diagnostic.MAX_TOTAL_BYTES >= 54*diagnostic.MAX_SNAPSHOT_BYTES


def test_simulation_composition_preserves_original_adapter_inventory(tmp_path):
    from test_phase2_native_state import fake_env
    import mujoco
    from human_punching import phase2_native_state as parent
    model=mujoco.MjModel.from_xml_path(str(Path(diagnostic.__file__).resolve().parents[2]/"outputs/four-punch-milestone/jab/model/human.xml"))
    env=fake_env(model,n=1,source=False)
    before=parent.inspect_adapter(env)
    env.sim=diagnostic.ObservedSimulation(env.sim,env,recorder(tmp_path),"restored")
    after=parent.inspect_adapter(env)
    assert after==before


def test_named_torch_views_capture_only_active_contact_and_constraint_rows():
    from types import SimpleNamespace
    from test_phase2_native_state import fake_env
    import mujoco
    import torch
    model=mujoco.MjModel.from_xml_path(str(Path(diagnostic.__file__).resolve().parents[2]/"outputs/four-punch-milestone/jab/model/human.xml"))
    env=fake_env(model,n=1,source=False)
    data=SimpleNamespace()
    for name in ("solver_niter","ne","nf","nl","nefc","nacon","nisland","qacc","qacc_smooth","qfrc_constraint","qfrc_smooth","actuator_force","qfrc_actuator","overflow"):
        setattr(data,name,torch.zeros(1,dtype=torch.int32) if name in ("solver_niter","ne","nf","nl","nefc","nacon","nisland","overflow") else torch.zeros(1,54))
    data.nacon[0]=1;data.nefc[0]=2
    data.contact=SimpleNamespace(**{name:torch.zeros((5,2),dtype=torch.int32) if name in ("geom","efc_address") else torch.zeros(5)
        for name in ("geom","dim","efc_address","worldid","geomcollisionid","type","dist","pos","frame")})
    data.efc=SimpleNamespace(**{name:torch.zeros(1,8) for name in ("type","id","state","pos","D","vel","aref","force","island")})
    env.sim.wp_data=data
    arrays=diagnostic.capture_arrays(env)
    assert arrays["native.integration"].shape==(1,410)
    assert arrays["contact.geom"].shape==(1,2)
    assert arrays["constraint.force"].shape==(1,2)
    assert arrays["controller.last_action"].shape==(1,48)
    assert arrays["static.ids"].shape==(48,)
    data.nacon[0]=diagnostic.MAX_CONTACTS+1
    with pytest.raises(ValueError,match="bound exceeded"):diagnostic.capture_arrays(env)
