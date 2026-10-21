from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from human_punching import phase2_right_hook_grouped_media_audit_v1 as audit


def evidence():
    rows=[dict(trial_id=str(i),command_id=i,ttc_s=.6,start_index=0,request_identity=str(i),reasons=['fixed'],disposition='completed_native_attempt') for i in range(6)]
    expected=dict(selection=rows,predeclared_video_ids=list(map(str,range(6))),checkpoint={'sha256':'a'*64})
    plan=dict(selection=deepcopy(rows),predeclared_video_ids=expected['predeclared_video_ids'],integration=False,inference=False,checkpoint_sha256='a'*64,declaration_checkpoint_template_is_unbound=True)
    summary=dict(planned=6,results=[dict(**r,status='rendered') for r in rows],simulation_steps=0,actor_calls=0,rendered=6,failed=0,unavailable=0,unrenderable=0,unattempted=0)
    return expected,plan,summary


def test_exact_selected_union_and_failed_dispositions_are_auditable():
    e,p,s=evidence();assert all(audit.selection_checks(e,p,s).values())
    s['results'][0]['status']='failed';s.update(rendered=5,failed=1)
    assert all(audit.selection_checks(e,p,s).values())


@pytest.mark.parametrize('change',['drop','duplicate','reorder','resample','checkpoint','actor','counter','unbound','request','fpscope'])
def test_changed_or_incomplete_media_evidence_fails_integrity(change):
    e,p,s=evidence()
    if change=='drop':s['results'].pop()
    elif change=='duplicate':s['results'][0]=s['results'][1]
    elif change=='reorder':s['results'].reverse()
    elif change=='resample':p['predeclared_video_ids']=['other']
    elif change=='checkpoint':p['checkpoint_sha256']='b'*64
    elif change=='actor':s['actor_calls']=1
    elif change=='counter':s['rendered']=7
    elif change=='unbound':p['declaration_checkpoint_template_is_unbound']=False
    elif change=='request':s['results'][0]['ttc_s']=.75
    else:p['integration']=True
    assert not all(audit.selection_checks(e,p,s).values())


def test_full_decode_counts_actual_frames(monkeypatch):
    import imageio.v2 as imageio
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def get_meta_data(self):return {'fps':50}
        def __iter__(self):return iter([np.zeros((4,5,3),np.uint8) for _ in range(7)])
    monkeypatch.setattr(imageio,'get_reader',lambda p:Reader())
    assert audit.decode_video('mock')==dict(decoded_frames=7,fps=50.,shape=[4,5,3])


def test_empty_video_cannot_count_as_a_render(monkeypatch):
    import imageio.v2 as imageio
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def get_meta_data(self):return {'fps':50}
        def __iter__(self):return iter([])
    monkeypatch.setattr(imageio,'get_reader',lambda p:Reader())
    with pytest.raises(ValueError,match='Nonempty'):audit.decode_video('mock')
