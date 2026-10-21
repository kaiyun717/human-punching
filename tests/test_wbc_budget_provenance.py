from pathlib import Path

import pytest

from human_punching.wbc.budget import TrainingScale, plateau
from human_punching.wbc.provenance import writable_path


def test_global_batch_does_not_multiply_by_gpu_count():
    two, four = TrainingScale(world_size=2), TrainingScale(world_size=4)
    assert (two.environments_per_rank, four.environments_per_rank) == (4096, 2048)
    assert two.fresh_transitions == four.fresh_transitions == 196608
    assert two.optimizer_steps == four.optimizer_steps == 20
    assert two.report()["total_transitions"] == 1966080000
    with pytest.raises(ValueError):
        TrainingScale(world_size=3)


def test_output_rejects_symlink_escape_and_frozen_release(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        writable_path(root / "escape" / "outside.json", root)
    with pytest.raises(ValueError):
        writable_path(root / "outputs/four-punch-milestone/checkpoint.pt", root)
    assert writable_path(root / "outputs/task10/new.json", root).is_relative_to(root)


def test_plateau_requires_all_metrics_to_stall():
    panel = dict(worst_family_success=.8, targeting_error=.02,
                 timing_error=.04, recovery_time=1.)
    panels = [panel.copy() for _ in range(4)]
    assert plateau(panels)
    panels[-1]["timing_error"] = .02
    assert not plateau(panels)
    panels[-1]["timing_error"] = float("nan")
    with pytest.raises(ValueError):
        plateau(panels)
