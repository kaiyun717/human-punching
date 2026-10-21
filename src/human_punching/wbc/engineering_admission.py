"""Explicit admission for scratch engineering learning, never task qualification.

Fixed-controller success and exact-path inverse-dynamics feasibility are useful
diagnostics, not prerequisites for learning a corrective residual. This adapter
instead requires native controller/clock health, complete reference geometry and
initial-velocity screening, immutable commands, and recorded dynamic diagnostics.
It leaves the production reference and runner gates unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from .contracts import LaunchSnapshot, PunchRequest
from .measurement_contract import digest
from .planning import launch_features
from .provenance import PROJECT_ROOT, configure_local_caches, runtime_record, sha256_file
from .reference_screening import screen_reference_geometry
from .runtime.bundle import load_runtime_bundle

SCHEMA = "human_wbc_engineering_learning_admission_v1"
SCOPE = "canonical_four_family_scratch_engineering"
LAUNCH_FEATURE_ABSOLUTE_TOLERANCE = 64*np.finfo(np.float64).eps


@dataclass(frozen=True)
class EngineeringAdmission:
    """Digest-bound evidence, distinct in type and scope from production approval."""
    payload: dict
    sha256: str

    @classmethod
    def from_dict(cls, value: dict) -> "EngineeringAdmission":
        payload = {key: val for key, val in value.items() if key != "sha256"}
        if (payload.get("schema") != SCHEMA or payload.get("scope") != SCOPE
                or value.get("sha256") != digest(payload)
                or payload.get("qualified_for_training") is not False
                or payload.get("target_ttc_domain_qualified") is not False):
            raise ValueError("Engineering admission schema, scope or digest differs")
        return cls(deepcopy(payload), value["sha256"])

    def as_dict(self) -> dict:
        value = {**deepcopy(self.payload), "sha256": self.sha256}
        type(self).from_dict(value)
        return value


def read_artifact(record: dict) -> Path:
    """Resolve relative identity or a content-identical NFS mount alias for reads.

    Login and compute nodes may expose the same project through different paths
    and inode identities. Both readable copies must match the recorded digest;
    writes still target only the local resolved project root.
    """
    root = PROJECT_ROOT.resolve()
    path = Path(record["path"])
    source = (root/path).resolve(strict=True)
    if not source.is_file() or sha256_file(source) != record.get("sha256"):
        raise ValueError("Engineering artifact content digest differs")
    if source.is_relative_to(root):
        return source
    matches = []
    for index, name in enumerate(source.parts):
        if name == root.name:
            candidate = root.joinpath(*source.parts[index+1:]).resolve()
            if (candidate.is_relative_to(root) and candidate.is_file()
                    and sha256_file(candidate) == record["sha256"]):
                matches.append(candidate)
    if len(set(matches)) != 1:
        raise ValueError("Engineering artifact is not a hash-identical local project file")
    return matches[0]


def _evidence(path: Path) -> dict:
    source = (PROJECT_ROOT/Path(path)).resolve(strict=True)
    local = read_artifact({"path": str(source), "sha256": sha256_file(source)})
    return {"path": str(local.relative_to(PROJECT_ROOT.resolve())), "sha256": sha256_file(local)}


def audit_controller_trace(arrays: dict, body: dict) -> dict:
    """Independently check affine drive and force at the preceding interval state."""
    kp, kd, caps = (np.asarray(body[name]) for name in ("kp", "kd", "effort"))
    target, velocity, ff, ctrl, force = (np.asarray(arrays[name], float) for name in
        ("joint_position_target", "joint_velocity_target", "joint_effort_target", "ctrl", "actuator_force"))
    qpos, qvel = np.asarray(arrays["qpos"], float), np.asarray(arrays["qvel"], float)
    if (target.ndim != 3 or target.shape[-1] != 48 or target.shape[0] < 2
            or any(a.shape != target.shape for a in (velocity, ff, ctrl, force))
            or qpos.shape != target.shape[:-1]+(55,) or qvel.shape != target.shape[:-1]+(54,)
            or any(a.shape != (48,) for a in (kp, kd, caps))
            or not all(np.isfinite(a).all() for a in (target, velocity, ff, ctrl, force, qpos, qvel))):
        raise ValueError("Malformed native controller trace")
    drive_error = float(np.max(abs(ctrl-(kp*target+kd*velocity+ff))))
    expected_force = np.clip(ctrl[1:]-kp*qpos[:-1, :, 7:]-kd*qvel[:-1, :, 6:], -caps, caps)
    force_error = float(np.max(abs(force[1:]-expected_force)))
    checks = {"native_affine_drive": drive_error <= 2e-4,
              "native_force_from_preintegration_state": force_error <= 2e-4,
              "declared_zero_feedforward": bool(np.all(ff == 0))}
    return {"passed": all(checks.values()), "checks": checks,
            "maximum_drive_error_nm": drive_error, "maximum_force_error_nm": force_error}


def native_controller_evidence(bundle, report_path: Path) -> dict:
    """Reaudit an actual zero-FF native guard trace; retain falls as failures."""
    import mujoco
    from .native_probe import audit_probe_arrays
    from .trainedactor_evaluation import verify_compiled_controller
    checked = load_runtime_bundle(bundle)
    evidence = _evidence(report_path)
    path = read_artifact(evidence)
    report = json.loads(path.read_text())
    identity = report.get("guard_identity", {})
    if (report.get("schema") != "human_wbc_static_guard_probe_v1" or not report.get("slurm_job_id")
            or report.get("includes_ppo") is not False or report.get("recorded_samples") != 1201
            or identity.get("model_sha256") != sha256_file(checked.model_path)
            or identity.get("source_controller_sha256") != checked.controller_sha256()
            or report.get("runtime", {}).get("packages") != runtime_record()["packages"]
            or report.get("runtime", {}).get("declared_pins_match") is not True):
        raise ValueError("Native evidence does not bind this body/controller/runtime")
    files = {}
    for filename, expected in (("native_controller_trace.npz", report.get("native_controller_trace_sha256")),
                               ("compiled_scene.mjb", report.get("compiled_scene", {}).get("sha256"))):
        item = _evidence(path.parent/filename)
        if item["sha256"] != expected:
            raise ValueError("Native evidence payload hash differs")
        files[filename] = item
    model = mujoco.MjModel.from_binary_path(str(read_artifact(files["compiled_scene.mjb"])))
    verify_compiled_controller(model, checked)
    if (model.opt.timestep != .005 or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            or model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL
            or model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON or model.opt.iterations != 10
            or model.opt.ls_iterations != 20 or model.opt.tolerance != 1e-8):
        raise ValueError("Native solver differs from the accepted engineering contract")
    with np.load(read_artifact(files["native_controller_trace.npz"]), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
        clock = audit_probe_arrays(arrays, policy_steps=300, physics_dt_s=model.opt.timestep,
                                   effort_limits=checked.body["effort"])
        drive = audit_controller_trace(arrays, checked.body)
    if clock["status"] != "passed" or not drive["passed"]:
        raise ValueError("Native controller/clock evidence failed")
    return {"report": evidence, "payloads": files, "clock_force": clock, "drive": drive,
        "guard_admission_passed": report.get("guard_admission_passed", False),
        "source_reference_manifest_sha256": report["reference_manifest_sha256"],
        "scope": "Controller/clock evidence only; this trace does not qualify the canonical punch references."}


def launch_feature_roundoff(request, stored_features) -> dict:
    """Check derived frame transforms; original snapshot identities stay exact."""
    actual, saved = np.asarray(launch_features(request), dtype=np.float64), np.asarray(stored_features, dtype=np.float64)
    if (actual.shape != (77,) or saved.shape != actual.shape
            or not np.isfinite(actual).all() or not np.isfinite(saved).all()):
        raise ValueError("Frozen derived launch features are malformed or nonfinite")
    maximum = float(np.max(np.abs(actual-saved)))
    if maximum > LAUNCH_FEATURE_ABSOLUTE_TOLERANCE:
        raise ValueError("Frozen derived launch features differ beyond float64 roundoff")
    return {"maximum_abs_error": maximum, "absolute_tolerance": LAUNCH_FEATURE_ABSOLUTE_TOLERANCE,
            "relative_tolerance": 0., "scope": "Derived float64 frame transforms only; immutable input identities remain exact"}


def reference_engineering_evidence(bundle) -> list[dict]:
    """Fresh complete-cycle geometry and frozen-command screening; no simulation."""
    checked = load_runtime_bundle(load_runtime_bundle(bundle).root)
    assembly = checked.manifest.get("assembly_report", {})
    if not assembly or sha256_file(checked.path(assembly["path"])) != assembly.get("sha256"):
        raise ValueError("Canonical engineering bundle requires intact assembly provenance")
    report = json.loads(checked.path(assembly["path"]).read_text())
    if report.get("schema") not in ("human_wbc_canonical_engineering_assembly_v1",
                                    "human_wbc_canonical_presampled_assembly_v1"):
        raise ValueError("Expected canonical presampled assembly evidence")
    rows = []
    for clip in checked.clips:
        master = checked.path(clip["master_path"])
        screen = screen_reference_geometry(checked.model_path, master, guard_time_s=clip["guard_time_s"])
        snap = LaunchSnapshot(**clip["launch_snapshot"])
        request = PunchRequest(clip["family"], clip["requested_ttc_s"], snap, clip["guard_id"])
        if (clip.get("g1_snapshot_present") is not True or snap.fingerprint != clip["snapshot_fingerprint"]
                or request.fingerprint != clip["request_fingerprint"]
                or not np.array_equal(snap.head_position_w, clip["target_w"])):
            raise ValueError("Frozen snapshot/target/command conditioning differs")
        feature_evidence = launch_feature_roundoff(request, clip["launch_features"])
        with np.load(master, allow_pickle=False) as source:
            time, glove = source["time_s"], source["glove_pos_w"]
            active = 0 if clip["family"] in ("jab", "left_hook") else 1
            distance = np.linalg.norm(glove-np.asarray(clip["target_w"]), axis=-1)
            hits = np.flatnonzero((time > 0) & (time <= clip["outward_end_s"]) & (distance[:, active] <= .14))
            contact = None if not len(hits) else float(time[hits[0]])
            checks = {"complete_geometry_and_readiness": screen["passed"],
                "launch_clock_zero": clip["launch_time_s"] == 0.,
                "initial_glove_clearance": bool(np.all(distance[0] >= .16)),
                "first_contact_matches_requested_ttc": contact is not None and abs(contact-request.ttc_s) <= .005+1e-10,
                "original_guard_return": bool(np.max(abs(source["qpos"][-1]-source["qpos"][0])) <= 1e-10),
                "zero_joint_feedforward": bool(np.all(source["joint_feedforward"] == 0))}
            # New converter uses the reference id, the original canonical packet
            # used family directories. Require exactly one physical diagnostic.
            candidates = [checked.root/"provenance"/name/"exact_path_effort.npz"
                          for name in dict.fromkeys((clip["reference_id"], clip["family"]))]
            paths = [path for path in candidates if path.exists()]
            if len(paths) != 1:
                raise ValueError("Each canonical reference needs explicit exact-path effort diagnostics")
            with np.load(paths[0], allow_pickle=False) as diagnostic:
                feasible = diagnostic["support_feasible"]
                ratio = diagnostic["minimum_peak_effort_ratio"]
                if (not np.array_equal(diagnostic["time_s"], time) or feasible.dtype != np.bool_
                        or feasible.shape != time.shape or ratio.shape != time.shape
                        or not np.isfinite(ratio[feasible]).all()):
                    raise ValueError("Exact-path effort diagnostic arrays differ")
                dynamics = {"support_infeasible_frames": int(np.sum(~feasible)),
                    "over_cap_frames": int(np.sum(feasible & (ratio > 1+1e-6))),
                    "peak_feasible_effort_ratio": float(np.max(ratio[feasible])) if feasible.any() else None}
        if not all(checks.values()):
            raise ValueError(f"Canonical hard reference screens failed for {clip['family']}: {checks}")
        rows.append({"family": clip["family"], "reference_id": clip["reference_id"],
            "master_sha256": clip["master_sha256"], "checks": checks, "geometry": screen,
            "launch_feature_roundoff": feature_evidence,
            "first_sampled_contact_s": contact, "exact_path_diagnostics": dynamics,
            "effort_evidence": _evidence(paths[0]), "physical_success_qualified": False})
    return rows


def prepare_engineering_admission(bundle, native_guard_report: Path) -> EngineeringAdmission:
    # Verification also runs inside allocated workers; preserve their confined
    # rank-specific native caches rather than silently redirecting them.
    if any(not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(PROJECT_ROOT)
           for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TMPDIR", "WARP_CACHE_PATH", "CUDA_CACHE_PATH")):
        configure_local_caches()
    checked = load_runtime_bundle(load_runtime_bundle(bundle).root)
    if checked.manifest.get("feedforward_mode") != "disabled_zero":
        raise ValueError("Canonical engineering adapter currently requires explicit zero feedforward")
    payload = {"schema": SCHEMA, "scope": SCOPE,
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "body_sha256": sha256_file(checked.model_path), "controller_sha256": checked.controller_sha256(),
        "adapter_sha256": sha256_file(Path(__file__)),
        "artifact_path_semantics": "human_project_relative_content_sha256",
        "native_controller": native_controller_evidence(checked, native_guard_report),
        "references": reference_engineering_evidence(checked),
        "qualified_for_training": False, "target_ttc_domain_qualified": False,
        "physical_guard_stability_qualified": False, "initialization": "scratch",
        "capacity_weights_may_initialize_training": False,
        "zero_residual_success_required": False, "exact_path_feasibility_required": False,
        "external_assistance": False, "midcycle_reference_reset": False,
        "limitations": ["Engineering learning admission is not production or task acceptance.",
            "Reference geometry does not establish loaded support, stability or successful physical execution.",
            "Exact-path LP failures and effort excess remain diagnostics; learning may depart from the reference.",
            "Four canonical requests do not establish a reachable target/TTC domain."]}
    return EngineeringAdmission.from_dict({**payload, "sha256": digest(payload)})


def verify_engineering_admission(bundle, admission: EngineeringAdmission) -> None:
    if not isinstance(admission, EngineeringAdmission):
        raise TypeError("An explicit typed engineering admission is required")
    saved = admission.as_dict()
    report = read_artifact(saved["native_controller"]["report"])
    rebuilt = prepare_engineering_admission(bundle, report).as_dict()
    # Derived transforms can differ at float64 roundoff across CPU/BLAS paths.
    # Keep both measured errors as evidence, while comparing all input identities,
    # screening results and all other diagnostic fields exactly. The admission
    # digest above is still verified before this narrowly scoped normalization.
    for value in (saved, rebuilt):
        value.pop("sha256")
        for row in value.get("references", []):
            feature = row.get("launch_feature_roundoff")
            if (not isinstance(feature, dict)
                    or feature.get("absolute_tolerance") != LAUNCH_FEATURE_ABSOLUTE_TOLERANCE
                    or feature.get("relative_tolerance") != 0.
                    or type(feature.get("maximum_abs_error")) is not float
                    or not np.isfinite(feature["maximum_abs_error"])
                    or not 0 <= feature["maximum_abs_error"] <= LAUNCH_FEATURE_ABSOLUTE_TOLERANCE):
                raise ValueError("Derived launch-feature roundoff evidence is missing or outside its bound")
            feature.pop("maximum_abs_error")
    if rebuilt != saved:
        raise ValueError("Engineering reference/body/controller/evidence changed")


def build_engineering_env_cfg(bundle, admission: EngineeringAdmission, *, num_envs: int,
                              auto_reset: bool):
    """Allow only explicit episode-boundary resets; clip completion still holds guard."""
    from .runtime.task import build_human_env_cfg
    if type(auto_reset) is not bool:
        raise TypeError("The engineering learning auto_reset override must be explicit boolean")
    verify_engineering_admission(bundle, admission)
    cfg = build_human_env_cfg(bundle, num_envs=num_envs, evaluation=True)
    cfg.auto_reset = auto_reset
    cfg.sim.nan_guard.enabled = True
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = cfg.scene.num_envs
        cfg.scene.terrain.env_spacing = cfg.scene.env_spacing
    return cfg


def engineering_runner(admission: EngineeringAdmission):
    """Keep inherited PPO and human exports; scoped scratch gate replaces only admission."""
    from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
    from .runtime.runner import HumanMotionTrackingRunner

    class EngineeringMotionTrackingRunner(HumanMotionTrackingRunner):
        def load(self, *args, **kwargs):
            raise RuntimeError("Engineering capacity starts from scratch; checkpoint initialization is forbidden")

        def learn(self, *args, **kwargs):
            command = self.env.unwrapped.command_manager.get_term("motion")
            verify_engineering_admission(command.cfg.bundle_path, admission)
            return PolicyOnlyMotionTrackingRunner.learn(self, *args, **kwargs)

        def save(self, path, infos=None):
            return super().save(path, {**(infos or {}), "engineering_capacity": {
                "schema": "human_wbc_engineering_capacity_checkpoint_v1",
                "admission_sha256": admission.sha256, "scope": SCOPE,
                "scratch_initialized": True, "production_checkpoint": False,
                "capacity_weights_may_initialize_training": False}})

        def build_policy_export_metadata(self, run_path):
            return {**super().build_policy_export_metadata(run_path),
                    "engineering_admission_sha256": admission.sha256,
                    "learning_scope": SCOPE, "production_checkpoint": False}

    return EngineeringMotionTrackingRunner
