"""CPU video selection/failure-prefix checks; no renderer or GPU is created."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC=importlib.util.spec_from_file_location('phase2_media',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_media.py')
MEDIA=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEDIA)


def panel(candidate='jab_original',failed=(2,),order=None):
    return dict(trials=[dict(candidate_id=candidate,trial_id=f'{candidate}_bank_{i:02d}',
        disposition='executed',numeric_complete_cycle=i not in failed) for i in (range(8) if order is None else order)])


def test_predeclared_selection_is_order_independent():
    original=MEDIA.selection(panel())
    shuffled=MEDIA.selection(panel(order=[7,3,0,4,6,1,5,2]))
    assert original==shuffled
    assert [(r['world'],r['reasons']) for r in original]==[(2,['flagged_failure']),(5,['predeclared_hash_sample'])]


def test_predeclared_bank_size_is_never_silently_resampled():
    with pytest.raises(ValueError,match='Incomplete'):
        MEDIA.selection(panel(order=range(7)))
    with pytest.raises(ValueError,match='Duplicate'):
        MEDIA.selection(panel(order=[0,1,2,3,4,5,6,6]))
    with pytest.raises(ValueError,match='identity'):
        MEDIA.selection(dict(trials=[dict(candidate_id='jab_original',trial_id='different',numeric_complete_cycle=True)]))


def test_all_rejected_or_unattempted_failures_stay_in_selection():
    data=panel(failed=range(8))
    for i,row in enumerate(data['trials']):
        row['disposition']='offline_rejection' if i<4 else 'unattempted_after_hard_failure'
    result=MEDIA.selection(data)
    assert len(result)==8
    assert all('flagged_failure' in row['reasons'] for row in result)
    assert next(row for row in result if row['world']==5)['reasons']==['predeclared_hash_sample','flagged_failure']


def test_finite_prefix_is_explicit_and_does_not_modify_failed_trace():
    q=np.zeros((5,55));q[:,3]=1.;v=np.zeros((5,54));q[3,10]=np.nan
    before=q.copy()
    count,receipt=MEDIA.finite_render_prefix(q,v)
    assert count==3
    assert receipt['first_invalid_sample']==3 and receipt['invalid_sample_indices']==[3]
    assert receipt['truncated_at_native_invalid_state']
    np.testing.assert_array_equal(q,before)


def test_invalid_quaternion_stops_rendering_prefix_even_when_finite():
    q=np.zeros((4,55));q[:,3]=1.;q[2,3]=.5
    count,receipt=MEDIA.finite_render_prefix(q,np.zeros((4,54)))
    assert count==2 and receipt['invalid_sample_indices']==[2]
    with pytest.raises(ValueError,match='coordinates'):
        MEDIA.finite_render_prefix(q[:,:-1],np.zeros((4,54)))


def test_media_requires_bounded_session_before_gpu_configuration():
    with pytest.raises(ValueError,match='wall limit'):
        MEDIA.render('unused','unused','unused',wall_seconds=20000.)


def test_accepted_predeclaration_hashes_match_implementation():
    import json
    path=Path(__file__).resolve().parents[1]/'outputs/task10b/p2prep_20261021_01/video_predeclaration_01.json'
    predeclared=json.loads(path.read_text())
    assert all(MEDIA.sample_bank_id(candidate)==world for candidate,world in predeclared['selected'].items())


def test_media_output_symlink_escape_rejected_before_native_import(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    link=tmp_path/'outside_project'
    link.symlink_to('/usr',target_is_directory=True)
    with pytest.raises(ValueError,match='output must be beneath'):
        MEDIA.render('unused','unused',link/'unwritten_media')
    assert not (link/'unwritten_media').exists()


def test_media_candidate_path_traversal_rejected():
    with pytest.raises(ValueError,match='path component'):
        MEDIA.selection(panel(candidate='../jab_original'))
