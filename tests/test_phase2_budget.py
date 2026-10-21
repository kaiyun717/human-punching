import pytest

from human_punching.phase2_budget import Allocation, budget_status, admit_reservation


def test_pending_reserves_full_wall_and_failure_counts_actual():
    rows = [Allocation('failed', 1800, elapsed_seconds=179, terminal=True), Allocation('queued', 1800)]
    result = budget_status(rows)
    assert result['remaining_gpu_seconds'] == 14400-179-1800
    assert result['actual_terminal_gpu_seconds'] == 179


def test_cannot_borrow_expected_early_exit_or_evaluation():
    rows = [Allocation('active', 14400)]
    with pytest.raises(ValueError, match='exceeds'):
        admit_reservation(rows, 1)
    with pytest.raises(ValueError, match='exceeds'):
        admit_reservation([], 12000, evaluation_reserve_seconds=3000)


@pytest.mark.parametrize('args', [dict(key='x', reserved_seconds=14401),
    dict(key='x', reserved_seconds=1, gpus=2),
    dict(key='x', reserved_seconds=2, elapsed_seconds=1),
    dict(key='x', reserved_seconds=2, terminal=True)])
def test_invalid_accounting_rejected(args):
    with pytest.raises(ValueError):
        Allocation(**args)


def test_conditional_ceiling_is_explicit():
    assert budget_status([])['ceiling_gpu_seconds'] == 14400
    assert budget_status([], successful_route_frozen=True)['ceiling_gpu_seconds'] == 100800


def test_overrun_preserved_not_capped():
    row = Allocation('overrun', 100, elapsed_seconds=102, terminal=True)
    assert budget_status([row])['actual_terminal_gpu_seconds'] == 102


def test_duplicate_job_receipts_fail():
    with pytest.raises(ValueError, match='Duplicate'):
        budget_status([Allocation('same', 100)]*2)
