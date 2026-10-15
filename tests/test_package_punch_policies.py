"""Release provenance/acceptance checks using synthetic records, not physics claims."""
import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_punch_policies.py"
SPEC = importlib.util.spec_from_file_location("package_punch_policies", SCRIPT)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def accepted_evidence(tmp_path):
    train = tmp_path / "outputs/train"
    evaluation = tmp_path / "outputs/eval"
    for root in (train, evaluation):
        (root / "model").mkdir(parents=True)
        (root / "model/human.xml").write_text('<mujoco><option timestep="0.005"/></mujoco>')
        (root / "model/human_runtime.xml").write_text('<mujoco model="runtime"><option timestep="0.005"/></mujoco>')
        for name in ("body_metadata.json", "arguments.json", "metrics.json"):
            write_json(root / name, {})
        for name in ("jab_learning_source.py", "scalar_control_source.py"):
            (root / name).write_text("# Synthetic unit-test source record\n")
        np.savez(root / "reference.npz", punch_family="jab", active_hand="Left", source_speed=1.,
                 attack_window_reference_s=[.5, 1.4])
    checkpoint = train / "checkpoint_00400.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    hashes = {"model_sha256": package.sha256(train / "model/human.xml"),
              "runtime_model_sha256": package.sha256(train / "model/human_runtime.xml"),
              "reference_sha256": package.sha256(train / "reference.npz")}
    schema = {"physics_dt": .005, "actor_dt": .02, "speed": 1., "controller_mode": "implicit_pd",
              "refresh_kinematics": True, "balance_feedback": False,
              **{key: value for key, value in hashes.items() if key != "runtime_model_sha256"}}
    saved = {"controller_schema": schema, "actions": 1, "update": 400,
             "policy": {"weight": [1.]}, "model_sha256": hashes["model_sha256"],
             "reference_sha256": hashes["reference_sha256"]}
    for root in (train, evaluation):
        write_json(root / "controller_schema.json", schema)
    qpos = np.zeros((32, 3))
    qvel = np.zeros_like(qpos)
    bank = evaluation / "evaluation_bank.npz"
    np.savez(bank, trial_ids=np.arange(32), qpos=qpos, qvel=qvel,
             model_sha256=hashes["model_sha256"], reference_sha256=hashes["reference_sha256"])
    bank_hash = package.sha256(bank)
    checks = dict.fromkeys(package.REQUIRED_GATES, True)
    rows = []
    for index in range(32):
        rollout = evaluation / f"rollout_{index:02d}.npz"
        np.savez(rollout, trial_id=index, checkpoint=str(checkpoint),
                 checkpoint_sha256=package.sha256(checkpoint), evaluation_kind="learned_policy",
                 physics_timestep_s=.005, policy_timestep_s=.02, reference_speed=1.,
                 time_s=np.arange(5)*.005, state_reset_count=1, physics_step_count=4,
                 policy_update_step_indices=[0], qpos=np.zeros((5, 3)), qvel=np.zeros((5, 3)),
                 evaluation_bank_sha256=bank_hash, **hashes)
        row = {"trial_id": index, "rollout": str(rollout), "pilot_pass": True,
               "failed_checks": [], "acceptance_checks": checks, "punch_family": "jab",
               "active_hand": "Left", "effective_source_speed": 1.,
               "evaluated_physics_timestep_s": .005, "model_xml_timestep_s": .005,
               "recorded_policy_timestep_s": .02, "simulator_clock_evidence": {"verified": True},
               "precision_targets": {"active_glove_rms_target_met": True, "contact_timing_target_met": True},
               "reference_tracking": {"active_glove_position_rms_m": .01, "attack_active_glove_position_rms_m": .015},
               "contact_timing": {"absolute_interpolated_contact_error_s": .01}}
        write_json(evaluation / f"independent/trial_{index:02d}.json", row)
        rows.append(row)
    summary = {"trial_count": 32, "pilot_pass_count": 32, "all_initial_states_match_bank": True,
               "required_physics_timestep_s": .005, "required_policy_timestep_s": .02,
               "gate_pass_counts": dict.fromkeys(checks, 32), "attack_window_reference_s": [.5, 1.4],
               "bank_sha256": bank_hash, "trials": rows, **hashes}
    write_json(evaluation / "independent/summary.json", summary)
    return {"project": tmp_path, "checkpoint": checkpoint, "evaluation": evaluation,
            "saved": saved, "summary": summary}


def validate(data, **kwargs):
    return package.validate_family("jab", data["checkpoint"], data["evaluation"],
                                   project=data["project"], checkpoint_loader=lambda _: data["saved"], **kwargs)


def test_validated_plan_requires_all_recorded_clocks_hashes_and_trials(accepted_evidence):
    result = validate(accepted_evidence, require_precision=True)
    assert result["record"]["accepted_physical_pilot_trials"] == 32
    assert result["record"]["checkpoint_hash_recorded_trials"] == 32
    assert result["record"]["all_precision_targets_met"]
    assert "evaluation/rollout_31.npz" in result["files"]
    assert not (accepted_evidence["project"] / "outputs/four-punch-milestone").exists()


@pytest.mark.parametrize("failure", ["gate", "clock", "runtime", "checkpoint", "trial"])
def test_rejects_failed_or_mismatched_evidence(accepted_evidence, failure):
    data = accepted_evidence
    summary_path = data["evaluation"] / "independent/summary.json"
    if failure == "gate":
        data["summary"]["gate_pass_counts"]["no_fall"] = 31
        write_json(summary_path, data["summary"])
    elif failure == "clock":
        data["saved"]["controller_schema"]["physics_dt"] = .002
    elif failure == "runtime":
        (data["evaluation"] / "model/human_runtime.xml").write_text('<mujoco><option timestep="0.002"/></mujoco>')
    elif failure == "checkpoint":
        data["checkpoint"].write_bytes(b"different-checkpoint")
    else:
        data["summary"]["trials"][-1]["trial_id"] = 0
        write_json(summary_path, data["summary"])
    with pytest.raises(ValueError):
        validate(data)


def test_precision_is_reported_separately_and_can_be_required(accepted_evidence):
    data = accepted_evidence
    data["summary"]["trials"][0]["reference_tracking"]["attack_active_glove_position_rms_m"] = .03
    write_json(data["evaluation"] / "independent/summary.json", data["summary"])
    write_json(data["evaluation"] / "independent/trial_00.json", data["summary"]["trials"][0])
    result = validate(data)
    assert result["record"]["precision_target_pass_counts"]["attack_active_glove_rms_2cm"] == 31
    assert result["record"]["all_precision_targets_met"] is False
    with pytest.raises(ValueError, match="precision target"):
        validate(data, require_precision=True)


def test_atomic_release_rejects_missing_family_or_post_validation_file_change(accepted_evidence):
    data = accepted_evidence
    plan = validate(data)
    output = data["project"] / "outputs/release"
    with pytest.raises(ValueError, match="all four"):
        package.package_release({"jab": plan}, output, project=data["project"])
    assert not output.exists()
    # A changed frozen source must abort publication and remove staging files.
    data["checkpoint"].write_bytes(b"changed-after-validation")
    with pytest.raises(ValueError, match="changed since validation"):
        package.package_release(dict.fromkeys(package.FAMILIES, plan), output, project=data["project"])
    assert not output.exists()
    assert not list(output.parent.glob(".release-*"))


def test_atomic_release_includes_global_evidence_and_bundled_evaluation_command(accepted_evidence):
    data = accepted_evidence
    plan = validate(data, require_precision=True)
    plans = {family: deepcopy(plan) for family in package.FAMILIES}
    for family in package.FAMILIES:
        plans[family]["record"]["family"] = family
    licenses = data["project"] / "assets/vendor/protomotions"
    licenses.mkdir(parents=True)
    for name in ("LICENSE.md", "NOTICE.md", "THIRD_PARTY_NOTICES.txt", "manifest.json"):
        (licenses / name).write_text("Synthetic license fixture\n")
    support = data["project"] / "outputs/validation"
    write_json(support / "manifest.json", {"synthetic": True})
    supporting = package.supporting_directory_files([support], project=data["project"])
    output = package.package_release(plans, data["project"] / "outputs/release", project=data["project"],
                                     supporting_files=supporting)
    manifest = package.read_json(output / "manifest.json")
    assert manifest["precision_targets_required_for_all_families"]
    assert "supporting/validation/manifest.json" in manifest["artifacts"]
    for relative, metadata in manifest["artifacts"].items():
        assert package.sha256(output / relative) == metadata["sha256"]
    readme = (output / "README.md").read_text()
    assert '--checkpoint "$family_bundle/checkpoint.pt"' in readme
    assert '--reference "$family_bundle/reference.npz"' in readme
    assert '--body-metadata "$family_bundle/body_metadata.json"' in readme
    assert "bundle_root=outputs/release" in readme


def test_nonuniform_timing_requires_explicit_admission_and_has_no_scalar_source_speed():
    reference = {"timing_kind": "nonuniform_pchip_phase_warp", "local_source_speed_min": .5,
                 "local_source_speed_max": 1.3, "timing_description": "Declared local timing change"}
    with pytest.raises(ValueError, match="allow-conditioned-timing"):
        package.reference_timing_metadata(reference)
    result = package.reference_timing_metadata(reference, allow_conditioned_timing=True)
    assert result["effective_source_speed"] is None
    assert result["source_timing_modified"] is True


@pytest.mark.parametrize("mislabel", [None, "uniform", "rate"])
def test_retimed_hook_uses_real_nested_evaluator_timing_schema(accepted_evidence, mislabel):
    data = accepted_evidence
    training = data["checkpoint"].parent
    evaluation = data["evaluation"]
    timing = {"timing_kind": "nonuniform_pchip_phase_warp", "local_source_speed_min": .49,
              "local_source_speed_max": 1.37, "timing_description": "Declared nonuniform timing",
              "source_speed_kind": "nonuniform_local_rates", "retiming_duration_ratio": 1.}
    for root in (training, evaluation):
        np.savez(root / "reference.npz", punch_family="right_hook", active_hand="Right",
                 attack_window_reference_s=[.5, 1.4], source_timing_modified=True, **timing)
    reference_hash = package.sha256(training / "reference.npz")
    data["saved"]["reference_sha256"] = reference_hash
    data["saved"]["controller_schema"]["reference_sha256"] = reference_hash
    for root in (training, evaluation):
        write_json(root / "controller_schema.json", data["saved"]["controller_schema"])
    bank_path = evaluation / "evaluation_bank.npz"
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = dict(archive)
    bank["reference_sha256"] = reference_hash
    np.savez(bank_path, **bank)
    bank_hash = package.sha256(bank_path)
    summary = data["summary"]
    summary.update(reference_sha256=reference_hash, bank_sha256=bank_hash)
    for row in summary["trials"]:
        row.update(punch_family="right_hook", active_hand="Right", source_speed=None,
                   effective_source_speed=None, reference_timing=timing.copy())
        if mislabel == "uniform":
            row["effective_source_speed"] = 1.
        elif mislabel == "rate":
            row["reference_timing"]["local_source_speed_min"] = .9
        write_json(evaluation / f"independent/trial_{row['trial_id']:02d}.json", row)
        rollout = Path(row["rollout"])
        with np.load(rollout, allow_pickle=False) as archive:
            values = dict(archive)
        values.update(reference_sha256=reference_hash, evaluation_bank_sha256=bank_hash)
        np.savez(rollout, **values)
    write_json(evaluation / "independent/summary.json", summary)
    kwargs = dict(project=data["project"], checkpoint_loader=lambda _: data["saved"])
    with pytest.raises(ValueError, match="allow-conditioned-timing"):
        package.validate_family("right_hook", data["checkpoint"], evaluation, **kwargs)
    if mislabel:
        with pytest.raises(ValueError, match="mislabels nonuniform timing"):
            package.validate_family("right_hook", data["checkpoint"], evaluation,
                                    allow_conditioned_timing=True, **kwargs)
    else:
        result = package.validate_family("right_hook", data["checkpoint"], evaluation,
                                         allow_conditioned_timing=True, require_precision=True, **kwargs)
        assert result["record"]["source_speed"] is None
        assert result["record"]["reference_timing"]["local_source_speed_min"] == .49
        assert result["record"]["accepted_physical_pilot_trials"] == 32


def test_selected_family_preflight_and_multiple_media_mappings_are_explicit():
    assert package.mapping(["jab=checkpoint.pt"], require_all=False) == {"jab": Path("checkpoint.pt")}
    with pytest.raises(ValueError, match="all four"):
        package.mapping(["jab=checkpoint.pt"])
    result = package.mapping(["jab=video", "jab=plots"], require_all=False, allow_multiple=True)
    assert result == {"jab": [Path("video"), Path("plots")]}


def test_media_requires_matching_checkpoint_and_copies_hash_provenance(accepted_evidence):
    data = accepted_evidence
    media = data["project"] / "outputs/media"
    write_json(media / "render.json", {"label": "Learned jab"})
    evidence = {"evaluation_kind": "learned_policy", "punch_family": "jab",
                "recorded_checkpoint_sha256": package.sha256(data["checkpoint"])}
    write_json(media / "evaluation.json", evidence)
    (media / "closest.png").write_bytes(b"synthetic-image")
    video = media / "jab-rollout.mp4"
    video.write_bytes(b"synthetic-video")
    decoding = {"video": str(video), "full_decode_exit_code": 0, "bytes": video.stat().st_size,
                "recorded_rollout": str(data["evaluation"] / "rollout_00.npz")}
    write_json(media / "video-validation.json", decoding)
    result = package.attach_optional_evidence(validate(data), media_dir=media, project=data["project"])
    assert result["files"]["media/media/closest.png"]["sha256"] == package.sha256(media / "closest.png")
    assert result["record"]["optional_evidence"]["media"][0]["render_checkpoint_verified"]
    decoding["bytes"] += 1
    write_json(media / "video-validation.json", decoding)
    with pytest.raises(ValueError, match="video size changed"):
        package.attach_optional_evidence(validate(data), media_dir=media, project=data["project"])
    decoding["bytes"] -= 1
    decoding["full_decode_exit_code"] = 1
    write_json(media / "video-validation.json", decoding)
    with pytest.raises(ValueError, match="video decoding failed"):
        package.attach_optional_evidence(validate(data), media_dir=media, project=data["project"])
    evidence["recorded_checkpoint_sha256"] = "different-checkpoint"
    write_json(media / "evaluation.json", evidence)
    with pytest.raises(ValueError, match="selected learned checkpoint"):
        package.attach_optional_evidence(validate(data), media_dir=media, project=data["project"])


def make_baseline(data):
    baseline = data["project"] / "outputs/baseline"
    shutil.copytree(data["evaluation"], baseline)
    write_json(baseline / "arguments.json", {"checkpoint": None})
    write_json(baseline / "metrics.json", {"kind": "reference_pd"})
    summary = package.read_json(baseline / "independent/summary.json")
    summary["pilot_pass_count"] = 0
    summary["gate_pass_counts"]["no_fall"] = 0
    for row in summary["trials"]:
        trial_id = row["trial_id"]
        rollout = baseline / f"rollout_{trial_id:02d}.npz"
        with np.load(rollout, allow_pickle=False) as archive:
            values = dict(archive)
        values.update(evaluation_kind="reference_pd", checkpoint="", checkpoint_sha256="")
        np.savez(rollout, **values)
        row.update(rollout=str(rollout), pilot_pass=False, failed_checks=["no_fall"])
        row["acceptance_checks"]["no_fall"] = False
        write_json(baseline / f"independent/trial_{trial_id:02d}.json", row)
    write_json(baseline / "independent/summary.json", summary)
    return baseline


@pytest.mark.parametrize("mismatch", [None, "schema", "bank"])
def test_zero_residual_baseline_is_paired_but_need_not_succeed(accepted_evidence, mismatch):
    data = accepted_evidence
    baseline = make_baseline(data)
    if mismatch == "schema":
        write_json(baseline / "controller_schema.json", {})
    elif mismatch == "bank":
        (baseline / "evaluation_bank.npz").write_bytes(b"different-bank")
    if mismatch:
        with pytest.raises(ValueError, match="differs|differ"):
            package.attach_optional_evidence(validate(data), baseline_dir=baseline, project=data["project"])
    else:
        result = package.attach_optional_evidence(validate(data), baseline_dir=baseline, project=data["project"])
        paired = result["record"]["optional_evidence"]["zero_residual_baseline"]
        assert paired["learned_pilot_pass_count"] == 32
        assert paired["baseline_pilot_pass_count"] == 0
        assert "baseline/rollout_31.npz" in result["files"]
        assert "baseline/independent/summary.json" in result["files"]
        media = data["project"] / "outputs/baseline-preview"
        write_json(media / "render.json", {"label": "Zero residual baseline"})
        write_json(media / "evaluation.json", {"evaluation_kind": "reference_pd", "punch_family": "jab",
                    "model_xml": str(baseline / "model/human_runtime.xml"), "recorded_checkpoint_sha256": ""})
        result = package.attach_optional_evidence(result, media_dir=media, project=data["project"])
        assert result["record"]["optional_evidence"]["media"][0]["render_baseline_verified"]
