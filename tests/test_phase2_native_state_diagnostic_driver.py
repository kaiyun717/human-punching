"""CPU checks for the output-local, fixed-scope diagnostic transport."""
import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'outputs/task10b/p2statediagprep01/driver.py'
spec = importlib.util.spec_from_file_location('isolated_diagnostic_driver', PATH)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_command_has_only_fixed_diagnostic_module_and_accepted_inputs():
    result = driver.native_command(dict(mode='native-state-diagnostic', output='outputs/task10b/p2sd01/run'), 269.)
    assert result[:4] == [str(driver.base.PYTHON), '-B', '-m', 'human_punching.phase2_native_state_diagnostic']
    assert result[4:] == ['--catalog', str(ROOT/'outputs/task10b/p2nativeparityprep01/manifest.json'),
        '--release', str(ROOT/'outputs/four-punch-milestone'), '--output', str(ROOT/'outputs/task10b/p2sd01/run'),
        '--wall-seconds', '269.0']


@pytest.mark.parametrize('remaining', [0, 120, 270.01, float('nan'), float('inf')])
def test_child_cannot_exceed_finite_declared_budget(remaining):
    with pytest.raises(ValueError):
        driver.native_command(dict(mode='native-state-diagnostic', output='x'), remaining)


def test_no_other_mode_can_use_diagnostic_transport():
    with pytest.raises(ValueError):
        driver.native_command(dict(mode='training', output='x'), 269)


def test_prepare_and_verify_have_no_submission_or_native_call():
    tree = ast.parse(PATH.read_text())
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in ('prepare', 'verify'):
            calls = [ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]
            assert not any(name in ('submit', 'execute', 'subprocess.Popen', 'os.killpg') for name in calls)
            assert 'native_command' not in calls


def test_submission_keeps_original_lock_ledger_budget_and_durable_intent():
    tree = ast.parse(PATH.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'submit')
    source = ast.unparse(node)
    assert 'fcntl.flock' in source and "base.PREP / 'allocation.lock'" in source
    assert 'expand.admit_reservation(base.ledger(), 360' in source
    assert source.index("'submission_intent.json',") < source.index('response = subprocess.run')
    assert 'successful_route_frozen=False' in source
    assert 'Existing submission intent' in source


def test_declared_watchdog_and_child_reserve_are_finite():
    source = PATH.read_text()
    assert 'remaining=300-(time.monotonic()-start)-30' in source
    assert 'time.monotonic()-start>=300' in source
    assert 'process.wait(timeout=15)' in source
    assert 'planned_explicit_integration_world_steps=12' in source
    assert 'child_wall_upper_bound_seconds=270' in source
    assert 'products_bytes_cap=3*1024**3' in source
