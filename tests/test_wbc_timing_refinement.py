import json
from pathlib import Path
import mujoco
import numpy as np
import pytest
from human_punching.wbc.references import PROJECT_ROOT,tangent_velocities
from human_punching.wbc.planning import _smoothstep
from human_punching.wbc.planning_refinement import _PoseCurve,_fk_contact
from human_punching.wbc.timing_refinement import PostContactTimingSettings,_expanded_time,refine_post_contact_timing


def test_expanded_clock_is_identity_before_window_monotone_and_c2():
    settings=PostContactTimingSettings(.5,1.,.3)
    times=np.linspace(0,1.5,10001);expanded=_expanded_time(times,settings)
    np.testing.assert_array_equal(expanded[times<=.5],times[times<=.5])
    assert np.min(np.diff(expanded))>0
    np.testing.assert_allclose(expanded[times>=1.]-times[times>=1.],.3,atol=1e-14)
    for endpoint in (.5,1.):
        h=1e-5;f=_expanded_time(np.array([endpoint-h,endpoint,endpoint+h]),settings)
        assert abs((f[2]-f[0])/(2*h)-1)<1e-7
        assert abs((f[2]-2*f[1]+f[0])/h**2)<.001


def test_postcontact_refinement_preserves_native_contact_path_and_quiet_endpoints(monkeypatch):
    monkeypatch.setattr(mujoco,'mj_step',lambda *args: (_ for _ in ()).throw(AssertionError('No integration')))
    path=PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    guard=np.array(json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())['guard_qpos'])
    times=np.arange(281)*.005;poses=np.tile(guard,(len(times),1))
    address=model.joint('LeftArm_z').qposadr[0]
    poses[:,address]+=.7*_smoothstep((times-.1)/.3)*_smoothstep((.7-times)/.3)
    data.qpos[:]=poses[80];mujoco.mj_forward(model,data)
    target=data.site_xpos[model.site('LeftHand_center').id].copy()
    contact=_fk_contact(model,_PoseCurve(times,poses),target,model.site('LeftHand_center').id,.6)
    settings=PostContactTimingSettings(contact+.01,.7,.25)
    final_time,final_pose,velocity,record=refine_post_contact_timing(path,times,poses,target,0,contact,.6,.9,settings=settings)
    assert abs(record['final_first_contact_s']-contact)<1e-7
    np.testing.assert_allclose(final_pose,_PoseCurve(times,poses)(record['source_clock_s']),atol=1e-12)
    np.testing.assert_array_equal(velocity,tangent_velocities(model,final_time,final_pose))
    np.testing.assert_allclose(velocity[[0,-1]],0,atol=1e-10)
    np.testing.assert_allclose(np.gradient(velocity,final_time,axis=0,edge_order=2)[[0,-1]],0,atol=1e-8)
    assert abs(record['guard_time_s']-1.15)<1e-12
    with pytest.raises(ValueError,match='after first contact'):
        refine_post_contact_timing(path,times,poses,target,0,contact,.6,.9,
            settings=PostContactTimingSettings(contact-.01,.7,.25))
