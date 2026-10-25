"""CPU production-FK parity for the batched source-clock placement transform."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from human_punching.task10d.phase2 import load_library
from human_punching.task10d.placement_phase3 import INDEX,PRIMITIVE_IDS
from human_punching.task10d.replay import Placement
from human_punching.task10d.task08_diagnostic import BatchedSourceMotion


@pytest.fixture(scope="module")
def primitives():
    return load_library(INDEX,list(PRIMITIVE_IDS))


@pytest.mark.parametrize("dtype,atol",[(torch.float64,2e-14),(torch.float32,5e-7)])
def test_all_eight_native_pose_parity_across_source_steps_and_placements(primitives,dtype,atol):
    xy=np.array([[0.,0.],[.17,-.12],[-.24,.51],[1.02,-.63]])
    yaw=np.array([0.,-.71,.43,2.81])
    maximum_position_error=0.
    maximum_rotation_error=0.
    for original in primitives:
        # Also exercise nonidentity accepted placement: its transform must be
        # removed before the requested native placement is composed once.
        for accepted in (original.placement,Placement((.23,-.19),.37)):
            primitive=replace(original,placement=accepted)
            motion=BatchedSourceMotion(primitive,xy,yaw,device="cpu",dtype=dtype)
            end=primitive.reference.end_time_s
            last=int(np.ceil(end/.005))
            outward=int(round(primitive.record["annotations"]["withdrawal_start_s"]/.005))
            steps=sorted(set((0,1,outward//2,outward,last,last+37)))
            for step in steps:
                positions,quaternions=motion.pose(step)
                actual_positions=positions.numpy()
                actual_quaternions=quaternions.numpy()
                for world in range(len(xy)):
                    expected=primitive.reference.sample(min(step*.005,end),Placement(tuple(xy[world]),yaw[world]))
                    maximum_position_error=max(maximum_position_error,float(np.max(abs(actual_positions[world]-expected.body_pos))))
                    actual_rotation=Rotation.from_quat(actual_quaternions[world],scalar_first=True).as_matrix()
                    expected_rotation=Rotation.from_quat(expected.body_quat,scalar_first=True).as_matrix()
                    maximum_rotation_error=max(maximum_rotation_error,float(np.max(abs(actual_rotation-expected_rotation))))
                    np.testing.assert_allclose(actual_positions[world],expected.body_pos,atol=atol,rtol=0)
                    np.testing.assert_allclose(actual_rotation,expected_rotation,atol=atol,rtol=0)
                    np.testing.assert_allclose(np.linalg.norm(actual_quaternions[world],axis=-1),1.,atol=atol,rtol=0)
    print(f"{dtype}: max position error={maximum_position_error:.3g}m; rotation matrix error={maximum_rotation_error:.3g}")


def test_write_respects_explicit_owned_mocap_ids_and_terminal_hold(primitives):
    primitive=primitives[4]
    motion=BatchedSourceMotion(primitive,np.array([[.2,-.3],[-.4,.1]]),np.array([.7,-.5]),
        device="cpu",dtype=torch.float64)
    count=motion.positions.shape[1]
    ids=np.arange(count)[::-1].copy()+2
    data=SimpleNamespace(mocap_pos=torch.full((2,count+4,3),-333.,dtype=torch.float64),
        mocap_quat=torch.full((2,count+4,4),-444.,dtype=torch.float64))
    binding=SimpleNamespace(mocap_ids=ids)
    motion.write(binding,data,10_000)
    expected_pos,expected_quat=motion.pose(len(motion.positions)-1)
    torch.testing.assert_close(data.mocap_pos[:,ids],expected_pos,atol=0,rtol=0)
    torch.testing.assert_close(data.mocap_quat[:,ids],expected_quat,atol=0,rtol=0)
    assert bool((data.mocap_pos[:,[0,1,count+2,count+3]]==-333).all())
    assert bool((data.mocap_quat[:,[0,1,count+2,count+3]]==-444).all())
