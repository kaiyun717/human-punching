"""Focused smooth-entry clock geometry, derivative and schedule checks."""
import pytest
import torch

from human_punching.task10c.events import (
    guard_weight, interaction_weight, reference_clock, scheduled_normalizers,
)
from human_punching.task10c.imitation import IMITATION_WEIGHTS, imitation_rates


ANCHOR, END, RECOVERY, DT = .510, 1.420, .910, .020


def clock(elapsed, requested):
    return reference_clock(elapsed, requested, ANCHOR, END, RECOVERY, mode='smooth_entry')


def test_smooth_entry_anchor_terminal_hold_and_continuous_rate():
    requested = torch.linspace(.5, .9, 41, dtype=torch.float64)
    at_entry, entry_rate = clock(requested, requested)
    torch.testing.assert_close(at_entry, torch.full_like(requested, ANCHOR), atol=1e-15, rtol=0)
    torch.testing.assert_close(entry_rate, ANCHOR / requested, atol=1e-15, rtol=0)
    at_end, end_rate = clock(requested + RECOVERY, requested)
    torch.testing.assert_close(at_end, torch.full_like(requested, END), atol=1e-15, rtol=0)
    torch.testing.assert_close(end_rate, torch.zeros_like(requested), atol=0, rtol=0)
    after_end, hold_rate = clock(requested + RECOVERY + .4, requested)
    torch.testing.assert_close(after_end, at_end, atol=0, rtol=0)
    torch.testing.assert_close(hold_rate, end_rate, atol=0, rtol=0)
    eps = 1e-7
    for boundary, expected in ((requested, entry_rate), (requested + RECOVERY, end_rate)):
        before, rate_before = clock(boundary - eps, requested)
        after, rate_after = clock(boundary + eps, requested)
        torch.testing.assert_close(rate_before, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(rate_after, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close((after - before) / (2 * eps), expected, atol=1e-6, rtol=0)


def test_smooth_entry_monotone_clock_and_analytic_derivative_across_ttc_range():
    requested = torch.linspace(.5, .9, 41, dtype=torch.float64)[:, None]
    elapsed = torch.linspace(0, 2.3, 2301, dtype=torch.float64)[None, :]
    source, rate = clock(elapsed, requested)
    assert bool(torch.all(torch.diff(source, dim=-1) >= -1e-14))
    assert bool(torch.all(rate >= 0))
    assert bool(torch.all((source >= 0) & (source <= END)))
    # Differentiate independent elapsed entries: validates ds/dt, including
    # intermediate recovery points rather than only the Hermite endpoints.
    samples = requested * torch.tensor([.2, .7], dtype=torch.float64)
    recovery_samples = requested + RECOVERY * torch.tensor([.1, .37, .8], dtype=torch.float64)
    samples = torch.cat((samples, recovery_samples), dim=1).requires_grad_()
    source, rate = clock(samples, requested)
    derivative, = torch.autograd.grad(source.sum(), samples)
    torch.testing.assert_close(derivative, rate, atol=1e-14, rtol=0)


def test_smooth_entry_full_schedule_keeps_seven_plus_six_budget():
    # Reward schedules use float32 actor-grid times in the actual command.
    requested = torch.tensor([.5, .707811745538643, .9], dtype=torch.float32)
    horizon, interaction_z, guard_z = scheduled_normalizers(requested, RECOVERY, DT)
    torch.testing.assert_close(horizon, torch.ceil((requested + RECOVERY + .4) / DT) * DT)
    grid = torch.arange(1, round(float(horizon.max() / DT)) + 1, dtype=torch.float32) * DT
    valid = grid[None, :] <= horizon[:, None] + 1e-6
    guard = guard_weight(grid[None, :], requested[:, None], RECOVERY)
    interaction = interaction_weight(grid[None, :], requested[:, None])
    torch.testing.assert_close((interaction * valid).sum(-1) * DT, interaction_z, atol=1e-7, rtol=0)
    scores = {name: torch.ones_like(guard) for name in IMITATION_WEIGHTS}
    rates = imitation_rates(scores, horizon[:, None], guard, guard_z[:, None])
    for name, weight in IMITATION_WEIGHTS.items():
        torch.testing.assert_close((rates[name] * valid).sum(-1) * DT,
                                   torch.full_like(requested, 13 * weight / 5), atol=2e-6, rtol=0)
    terminal_source, terminal_rate = clock(horizon, requested)
    torch.testing.assert_close(terminal_source, torch.full_like(requested, END))
    torch.testing.assert_close(terminal_rate, torch.zeros_like(requested))


def test_legacy_default_unchanged_and_unknown_clock_rejected():
    elapsed = torch.tensor([.2, .6, .7, 1.4, 1.8])
    requested = torch.full_like(elapsed, .6)
    default = reference_clock(elapsed, requested, .9, 1.7, .8)
    explicit = reference_clock(elapsed, requested, .9, 1.7, .8, mode='legacy')
    for actual, expected in zip(default, explicit, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(default[0], torch.tensor([.3, .9, 1., 1.7, 1.7]))
    torch.testing.assert_close(default[1], torch.tensor([1.5, 1.5, 1., 0., 0.]))
    with pytest.raises(ValueError, match='Unknown reference clock mode'):
        reference_clock(elapsed, requested, .9, 1.7, .8, mode='unrecognized')
