"""Focused exact-product and frozen-target source-bracket checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching.task10d.replay import Placement
from human_punching.task10d.task08_diagnostic_inputs import first_entry_timing,validate_episode_matrix


def test_source_bracket_refines_same_active_outward_crossing_without_rate_change():
    # A known linear 1 m/s active path crosses .14m at t=.006s. Recovery
    # geometry is present but must not determine the active outward bracket.
    def sample(time,placement):
        gloves=np.array([[.146-time,0.,1.],[.4,0.,1.]])
        return SimpleNamespace(glove_pos=placement.transform_points(gloves))
    primitive=SimpleNamespace(active_hand="left",placement=Placement(),reference=SimpleNamespace(sample=sample))
    times=np.array([0.,.005,.010,.015]);gloves=np.stack([sample(t,Placement()).glove_pos for t in times])
    outward=np.array([True,True,True,False])
    placement=Placement((1.,2.),.5)
    head=placement.transform_points(np.array([[0.,0.,1.]]))[0]
    result=first_entry_timing(primitive,times,gloves,outward,head,placement)
    assert result["source_previous_sample_s"] == .005
    assert result["source_outward_entry_sample_s"] == .010
    assert result["source_outward_entry_interpolated_s"] == pytest.approx(.006,abs=1e-12)
    assert result["source_first_any_hand_index"] == 0
    assert not result["earlier_inactive_entry"]


def test_recovery_only_entry_is_not_accepted_as_launch_input():
    primitive=SimpleNamespace(active_hand="left",placement=Placement())
    times=np.array([0.,.005,.010])
    gloves=np.array([[[.3,0,1],[.5,0,1]],[[.2,0,1],[.5,0,1]],[[.1,0,1],[.5,0,1]]])
    with pytest.raises(ValueError,match="future active-outward"):
        first_entry_timing(primitive,times,gloves,np.array([True,True,False]),np.array([0.,0.,1.]),Placement(),interpolate=False)


def test_exact_cartesian_product_rejects_a_substituted_snapshot():
    ids=np.arange(1000)+10000
    arrays=dict(episode_id=np.arange(8000),snapshot_id=np.tile(ids,8),
        primitive_index=np.repeat(np.arange(8),1000),source_clock_rate=np.ones(8000),
        source_previous_sample_s=np.full(8000,.495),source_outward_entry_sample_s=np.full(8000,.5),
        source_outward_end_s=np.full(8000,.7))
    validate_episode_matrix(arrays,ids)
    arrays["snapshot_id"][2000]=99999
    with pytest.raises(AssertionError):
        validate_episode_matrix(arrays,ids)
