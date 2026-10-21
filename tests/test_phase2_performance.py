"""Offline performance receipts do not invent startup or real-time evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('phase2_performance',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_performance.py')
PERF=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PERF)


def test_update_rate_excludes_startup_and_validation_boundary():
    records=[dict(update=i,fresh_transitions=i*8192,elapsed_s=100.+i*2+(30. if i>8 else 0.)) for i in range(1,11)]
    result=PERF.update_throughput(records)
    assert result['fresh_transitions_per_s']==4096.
    assert result['included_fresh_transitions']==8*8192
    assert [(r['from_update'],r['to_update']) for r in result['excluded_intervals']]==[(8,9)]
    assert not result['real_time_operation_qualified']
    assert PERF.update_throughput(records[:1])['fresh_transitions_per_s'] is None


def test_array_payload_sizes_use_headers_and_keep_float_width(tmp_path):
    path=tmp_path/'trace.npz'
    np.savez_compressed(path,qpos=np.zeros((5,8,55),np.float32),clock=np.zeros((5,8),np.float64))
    sizes=PERF.array_shapes(path)
    assert sizes['qpos']['shape']==[5,8,55]
    assert sizes['qpos']['array_payload_bytes']==5*8*55*4
    assert sizes['clock']['array_payload_bytes']==5*8*8


def test_latency_statistics_keep_nonfinite_faults_explicit():
    result=PERF.scalar_statistics([.01,.02,.03,np.nan])
    assert result['samples']==4 and result['nonfinite']==1
    assert result['over_20ms']==1
    assert result['mean_s']==pytest.approx(.02)
    assert PERF.scalar_statistics([])['maximum_s'] is None


def test_warp_log_is_partial_observation_never_total_compile_time(tmp_path):
    path=tmp_path/'job.out'
    path.write_text("Module example 123 load on device 'cuda:0' took 250.0 ms (compiled)\n")
    result=PERF.compile_log_observations([path])
    assert result['reported_load_sum_s']==.25
    assert result['complete_compilation_wall_s'] is None
    assert PERF.compile_log_observations([])['reported_load_sum_s'] is None


def test_performance_output_symlink_escape_rejected_without_write(tmp_path):
    link=tmp_path/'outside_project'
    link.symlink_to('/usr',target_is_directory=True)
    with pytest.raises(ValueError,match='output must be beneath'):
        PERF.write_json(PERF.writable_path(link/'unwritten_performance.json'),{'native_execution_performed':False})
    assert not (link/'unwritten_performance.json').exists()
