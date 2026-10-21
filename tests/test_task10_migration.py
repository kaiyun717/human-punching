"""Migration command/resource guards; no native simulation or scheduler calls."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("task10_migration", ROOT / "scripts/task10_migration.py")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def arguments():
    return dict(timestep=.005, policy_dt=.02, speed=.83, hold=1.2, gain_scale=.9,
                action_scale=.25, no_feedforward=True, balance_feedback=False,
                controller="implicit_pd", refresh_kinematics=True,
                reward_profile="precision", glove_sharpness=600., joint_limit_penalty=.2,
                attack_tracking_weight=.15, attack_tracking_scale=.06, guard_relative_weight=.4,
                attack_window=[.25, 1.3], seed=42)


def argv(checkpoint=None):
    paths = dict(model="model.xml", reference="reference.npz", metadata="body.json", bank="bank.npz")
    return migration.build_argv(paths, arguments(), "output", checkpoint=checkpoint)


def test_paired_commands_only_differ_by_checkpoint():
    learned, baseline = argv("selected.pt"), argv()
    assert learned == baseline + ["--checkpoint", "selected.pt"]
    assert "--no-feedforward" in baseline
    assert "--balance-feedback" not in baseline
    for key in ("speed", "hold", "gain_scale", "action_scale", "reward_profile",
                "glove_sharpness", "joint_limit_penalty", "attack_tracking_weight",
                "attack_tracking_scale", "guard_relative_weight", "seed"):
        flag = "--" + key.replace("_", "-")
        assert baseline[baseline.index(flag)+1] == str(arguments()[key])
    assert "--initialize-policy" not in learned


@pytest.mark.parametrize("flag,value", [("--mode", "train"), ("--timestep", "0.002"),
    ("--policy-dt", "0.005"), ("--num-envs", "8192"), ("--recorded-worlds", "4"),
    ("--controller", "explicit_pd")])
def test_mode_clocks_and_counts_cannot_be_overridden(flag, value):
    command = argv()
    command[command.index(flag)+1] = value
    with pytest.raises(ValueError):
        migration.validate_argv(command)


def test_no_duplicate_output_or_transfer_or_missing_refresh():
    for command in (argv() + ["--output", "/tmp/unsafe"],
                    argv() + ["--initialize-policy", "old.pt"],
                    [arg for arg in argv() if arg != "--refresh-kinematics"]):
        with pytest.raises(ValueError):
            migration.validate_argv(command)


def test_pilot_pass_does_not_hide_missing_release_precision_or_wrong_hand():
    row = dict(pilot_pass=True, active_hand="Left", attack_minimum_glove_distance_m=[.12, .25],
               reference_tracking={"attack_active_glove_position_rms_m": .019},
               precision_targets={"active_glove_rms_target_met": True, "contact_timing_target_met": True})
    assert migration.strict_release_count({"trials": [row]}) == 1
    row["reference_tracking"]["attack_active_glove_position_rms_m"] = .021
    assert migration.strict_release_count({"trials": [row]}) == 0
    row["reference_tracking"]["attack_active_glove_position_rms_m"] = .019
    row["attack_minimum_glove_distance_m"] = [.25, .12]
    assert migration.strict_release_count({"trials": [row]}) == 0


def test_slurm_spooled_script_uses_submission_directory(tmp_path):
    root = tmp_path / "human-punching"
    for relative in ("src/human_punching/jab_learning.py", "scripts/task10_migration.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Test fixture; never executed\n")
    fake_python = root / ".venv-wbc/bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    fake_python.chmod(0o700)
    spooled = tmp_path / "var/spool/slurmd/job123/slurm_script"
    spooled.parent.mkdir(parents=True)
    spooled.write_bytes((ROOT / "scripts/task10_migration.sbatch").read_bytes())
    environment = {**os.environ, "SLURM_JOB_ID": "synthetic-shell-test",
                   "CUDA_VISIBLE_DEVICES": "0", "SLURM_SUBMIT_DIR": str(root)}
    result = subprocess.run(["bash", str(spooled), "prepared.json"], cwd=tmp_path,
                            env=environment, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ["-B", str(root / "scripts/task10_migration.py"),
                                         "execute", "--packet", "prepared.json"]
    environment["SLURM_SUBMIT_DIR"] = str(spooled.parent)
    result = subprocess.run(["bash", str(spooled), "prepared.json"], cwd=tmp_path,
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 2
    assert "human-punching" in result.stderr


def test_compute_mount_aliases_preserve_identity_but_distinct_paths_fail(tmp_path):
    real = tmp_path / "compute-mount"
    real.mkdir()
    alias = tmp_path / "login-mount"
    alias.symlink_to(real, target_is_directory=True)
    inputs = {key: str(alias / name) for key, name in (
        ("model", "model.xml"), ("reference", "reference.npz"),
        ("metadata", "body.json"), ("bank", "bank.npz"))}
    task = {"inputs": inputs,
            "argv": migration.build_argv(inputs, arguments(), alias / "output")}
    migration.verify_task_paths(task, real / "output")
    with pytest.raises(ValueError, match="--output"):
        migration.verify_task_paths(task, real / "different-output")
