"""Prospective Task10B Phase2 allocation accounting; no scheduler side effects."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class Allocation:
    key: str
    reserved_seconds: int
    gpus: int = 1
    elapsed_seconds: int | None = None
    terminal: bool = False

    def __post_init__(self):
        if (not self.key or type(self.gpus) is not int or self.gpus != 1 or type(self.reserved_seconds) is not int
                or self.reserved_seconds <= 0 or self.reserved_seconds > 14400):
            raise ValueError('One specialist/GPU and positive sessions at most four hours required')
        if self.terminal != (self.elapsed_seconds is not None):
            raise ValueError('Only terminal accounting releases a reservation')
        if self.elapsed_seconds is not None and (type(self.elapsed_seconds) is not int or self.elapsed_seconds < 0):
            raise ValueError('Negative allocated time')

    @property
    def charged_seconds(self) -> int:
        return self.elapsed_seconds if self.terminal else self.reserved_seconds


def budget_status(allocations: Iterable[Allocation], *, successful_route_frozen: bool = False) -> dict:
    rows = list(allocations)
    if len({row.key for row in rows}) != len(rows):
        raise ValueError('Duplicate allocations would corrupt accounting')
    ceiling = 14400 + (86400 if successful_route_frozen else 0)
    charged = sum(row.charged_seconds for row in rows)
    return {'ceiling_gpu_seconds': ceiling, 'charged_or_reserved_gpu_seconds': charged,
            'actual_terminal_gpu_seconds': sum(row.charged_seconds for row in rows if row.terminal),
            'outstanding_reserved_gpu_seconds': sum(row.charged_seconds for row in rows if not row.terminal),
            'remaining_gpu_seconds': ceiling - charged}


def admit_reservation(allocations: Iterable[Allocation], seconds: int, *,
                      evaluation_reserve_seconds: int = 0,
                      successful_route_frozen: bool = False) -> dict:
    Allocation('proposed', seconds)
    if (not isfinite(evaluation_reserve_seconds) or evaluation_reserve_seconds < 0):
        raise ValueError('Finite nonnegative evaluation reserve required')
    status = budget_status(allocations, successful_route_frozen=successful_route_frozen)
    if seconds + evaluation_reserve_seconds > status['remaining_gpu_seconds']:
        raise ValueError('Reservation plus protected evaluation reserve exceeds remaining authorization')
    return dict(status, proposed_gpu_seconds=seconds, protected_evaluation_seconds=evaluation_reserve_seconds,
                remaining_after_reservation=status['remaining_gpu_seconds']-seconds)
