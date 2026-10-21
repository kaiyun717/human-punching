"""No scheduler/native calls: reservation and dispatch failure boundaries."""
import importlib.util
from pathlib import Path

import pytest

from human_punching.phase2_budget import Allocation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location('rh_interior_launcher_test',
        ROOT / 'scripts/task10b_phase2_right_hook_interior_v1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def record(driver):
    return dict(mode=driver.MODE, child_wall_seconds=90,
        allocation_wall_seconds=180, application_wall_seconds=150,
        interior_cells=1, interior_ttc_s=1.10, actor_calls=0,
        explicit_integration_steps=0, training=False,
        inputs={'plan': 'tests/test_phase2_right_hook_interior_launcher_v1.py'},
        output='outputs/task10b/future_interior/run')


def test_pending_allocations_do_not_release_budget(driver):
    # Reserving another180 exactly preserves4500; one more charged second fails.
    rows = [Allocation('pending', 9720)]
    result = driver.protected_reservation(rows, 180)
    assert result['remaining_after_reservation'] == 4500
    with pytest.raises(ValueError):
        driver.protected_reservation([Allocation('pending', 9721)], 180)


def test_only_measured_terminal_time_releases_reservation(driver):
    row = Allocation('terminal', 10000, elapsed_seconds=9720, terminal=True)
    assert driver.protected_reservation([row], 180)['remaining_after_reservation'] == 4500


def test_generation_cannot_drop_evaluation_reserve(driver):
    with pytest.raises(ValueError):
        driver.protected_reservation([Allocation('other', 10000)], 180,
            evaluation_reserve_seconds=0)


def test_generation_cannot_unlock_conditional_hours(driver):
    with pytest.raises(ValueError):
        driver.protected_reservation([], 180, successful_route_frozen=True)


@pytest.mark.parametrize('remaining', [None, 120., 100., 90.])
def test_dispatch_has_fixed_child_limit(driver, record, remaining):
    cmd = driver.native_command(record, remaining)
    assert cmd[2:4] == ['-m', 'human_punching.phase2_right_hook_interior_v1']
    assert cmd[-2:] == ['--wall-seconds', '90.0']


@pytest.mark.parametrize('remaining', [89.99, 121., float('nan'), float('inf'), True])
def test_late_or_nonfinite_setup_never_launches(driver, record, remaining):
    with pytest.raises(RuntimeError):
        driver.native_command(record, remaining)


@pytest.mark.parametrize(('key', 'value'), [('training', True), ('interior_cells', 2),
    ('interior_ttc_s', .90), ('actor_calls', 1), ('explicit_integration_steps', 1)])
def test_scope_expansion_never_dispatches(driver, record, key, value):
    record[key] = value
    with pytest.raises(ValueError):
        driver.native_command(record, 100)


def test_private_transport_retains_original_launcher_bytes(driver):
    # Loading the adapter cannot modify the sealed production transport.
    source = ROOT / 'scripts/task10b_phase2_expand.py'
    before = source.read_bytes()
    spec = importlib.util.spec_from_file_location('original_transport_independent', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    assert original.SCHEMA != driver.SCHEMA
    assert original.CAP != driver.PRODUCT_BYTES
    assert source.read_bytes() == before
