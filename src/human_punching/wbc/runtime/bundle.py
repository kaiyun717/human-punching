"""Validated, read-only runtime view of converted reference bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SOURCE_PINS = {"wbc_mjlab": "eb6c024b9586bc1de41b3dd22b9f18ad51cb1da9",
               "mjlab": "8ee51fbcf806a7419189f706d9e394cbeb7790fa"}
SIMULATION_CONTRACT = {"physics_dt_s": .005, "policy_dt_s": .020, "reference_dt_s": .005,
                       "decimation": 4, "integrator": "implicitfast", "solver": "newton",
                       "cone": "pyramidal", "iterations": 10, "ls_iterations": 20,
                       "tolerance": 1.e-8}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RuntimeBundle:
    root: Path
    manifest: dict
    body: dict
    clips: tuple[dict, ...]

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError(f"Bundle member escapes its root: {relative}")
        return path

    @property
    def model_path(self) -> Path:
        return self.path(self.manifest["model_path"])

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.body["joint_names"])

    @property
    def body_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.body["body_map"], key=self.body["body_map"].get))

    def controller_contract(self) -> dict:
        return {"law": "native_affine_pd_force_limited", "joint_names": list(self.joint_names),
                "kp": self.body["kp"], "kd": self.body["kd"], "effort": self.body["effort"],
                "joint_limits_rad": self.body["joint_limits_rad"], "residual_scale_rad": .25,
                "raw_clip": 1., "feedforward_mode": self.manifest.get("feedforward_mode"),
                "root_actuated": False, "external_assistance": False}

    def controller_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.controller_contract(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def require_training_qualification(self) -> None:
        # This evidence is necessary, not sufficient: the CLI also checks runtime
        # qualification and the requested compute authorization.
        if self.manifest.get("qualified_for_training") is not True:
            raise ValueError("Reference bundle is unqualified for training; use evaluation fixtures")
        for clip in self.clips:
            ttc = clip.get("requested_ttc_s")
            context = clip.get("launch_features")
            if not isinstance(ttc, (int, float)) or not np.isfinite(ttc) or ttc <= 0:
                raise ValueError("Training requires a validated requested TTC for every clip")
            if (not isinstance(context, list) or len(context) != 77
                    or not np.isfinite(context).all()):
                raise ValueError("Training requires frozen G1 launch context for every clip")
            if clip.get("g1_snapshot_present") is not True or not clip.get("snapshot_fingerprint"):
                raise ValueError("Training requires identified G1 launch snapshot payloads")
            if clip.get("launch_time_s") != 0.:
                raise ValueError("Training reference clock must start at the accepted punch launch")
            if (clip.get("guard_source") != "explicit_common_guard"
                    or clip.get("common_guard_validated") is not True):
                raise ValueError("Training requires an admitted common guard for every reference")
        evidence = self.manifest.get("qualification_evidence")
        if not isinstance(evidence, dict) or not evidence.get("path") or not evidence.get("sha256"):
            raise ValueError("Training qualification must reference hashed physical evidence")
        if sha256(self.path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("Training qualification evidence hash mismatch")
        report = json.loads(self.path(evidence["path"]).read_text())
        if report.get("schema") != "human_wbc_reference_qualification_v1" or report.get("status") != "passed":
            raise ValueError("Expected a passed human reference qualification report")
        if report.get("source_pins") != SOURCE_PINS:
            raise ValueError("Qualification source revisions differ from the runtime pins")
        if report.get("simulation_contract") != SIMULATION_CONTRACT:
            raise ValueError("Qualification physics/control/reference contract mismatch")
        if report.get("model_sha256") != sha256(self.model_path):
            raise ValueError("Qualification model hash mismatch")
        if report.get("controller_sha256") != self.controller_sha256():
            raise ValueError("Qualification human controller hash mismatch")
        required_checks = ("native_force_law", "force_caps", "five_ms_clock", "free_root",
                           "zero_external_assistance", "reference_frames_and_order",
                           "common_guard_admission", "initial_glove_clearance",
                           "target_ttc_domain_admission")
        if any(report.get("checks", {}).get(name) is not True for name in required_checks):
            raise ValueError("Qualification lacks required physical/API screening checks")
        rows = report.get("references", [])
        expected = {(c.get("reference_id", Path(c["motion_path"]).stem), c["motion_sha256"],
                     c["master_sha256"]) for c in self.clips}
        actual = {(row.get("reference_id"), row.get("motion_sha256"), row.get("master_sha256"))
                  for row in rows if row.get("kinematic_screen_passed") is True}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("Qualification does not cover the exact reference variants")


def load_runtime_bundle(bundle: str | Path | RuntimeBundle) -> RuntimeBundle:
    if isinstance(bundle, RuntimeBundle):
        return bundle
    root = Path(bundle).resolve(strict=True)
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "human_wbc_references_v1":
        raise ValueError("Unsupported human reference bundle schema")
    if any(manifest.get(key) != value for key, value in (
            ("physics_dt_s", .005), ("policy_dt_s", .020), ("decimation", 4),
            ("quaternion_order", "wxyz"))):
        raise ValueError("Reference bundle clock or quaternion contract mismatch")
    provisional = RuntimeBundle(root, manifest, {}, ())
    for field in ("model_path", "body_metadata_path", "controller_schema_path"):
        asset = provisional.path(manifest[field])
        expected_hash = manifest.get("asset_sha256", {}).get(asset.name)
        if expected_hash != sha256(asset):
            raise ValueError(f"Reference bundle asset hash mismatch: {field}")
    body = json.loads(provisional.path(manifest["body_metadata_path"]).read_text())
    clips = tuple(sorted(manifest["clips"], key=lambda x: x["motion_path"]))
    result = RuntimeBundle(root, manifest, body, clips)
    if not clips or set(c["family"] for c in clips) != set(FAMILIES):
        raise ValueError("A shared human library must include all four punch families")
    if len(result.joint_names) != 48 or body.get("root_actuated") is not False:
        raise ValueError("Expected the constrained 48-joint human with a free unactuated root")
    if len(set(result.joint_names)) != 48:
        raise ValueError("Human joint names must be unique")
    if (not result.body_names or result.body_names[0] != "Hips"
            or sorted(body["body_map"].values()) != list(range(1, len(result.body_names) + 1))):
        raise ValueError("Reference body ordering must exclude world and start with Hips")
    if body.get("xml_sha256") != sha256(result.model_path):
        raise ValueError("Human body hash does not match body metadata")
    for key in ("kp", "kd", "effort"):
        values = np.asarray(body[key], dtype=float)
        if values.shape != (48,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid human actuator {key}")
        if np.any(values < 0) or (key != "kd" and np.any(values == 0)):
            raise ValueError(f"Invalid human actuator {key}")
    limits = np.asarray(body["joint_limits_rad"], dtype=float)
    if (limits.shape != (48, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError("Invalid physical joint limits")
    names = [Path(c["motion_path"]).stem for c in clips]
    if len(set(names)) != len(names):
        raise ValueError("Reference variants require unique motion filenames")
    context_sizes = set()
    for clip in clips:
        for field in ("motion", "master"):
            path = result.path(clip[f"{field}_path"])
            if sha256(path) != clip[f"{field}_sha256"]:
                raise ValueError(f"{field} hash mismatch for {clip['family']}")
        with np.load(result.path(clip["motion_path"]), allow_pickle=False) as arrays:
            fps = np.asarray(arrays["fps"]).reshape(-1)
            if fps.shape != (1,) or fps[0] != 50.0:
                raise ValueError("WBC observation references must be sampled at 50 Hz")
            if tuple(arrays["joint_names"].tolist()) != result.joint_names:
                raise ValueError("Reference joint order mismatch")
            if tuple(arrays["body_names"].tolist()) != result.body_names:
                raise ValueError("Reference body order must exclude world and start with Hips")
            motion = {key: arrays[key].copy() for key in (
                "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w")}
        with np.load(result.path(clip["master_path"]), allow_pickle=False) as arrays:
            times = arrays["time_s"]
            if (times.ndim != 1 or len(times) < 5 or not np.isfinite(times).all()
                    or abs(times[0]) > 1e-12
                    or not np.allclose(np.diff(times), .005, rtol=0., atol=1e-12)):
                raise ValueError("Master references require uniform 5 ms samples starting at zero")
            if (len(times) - 1) % 4:
                raise ValueError("Master endpoint must also lie on the 50 Hz view clock")
            if tuple(arrays["joint_names"].tolist()) != result.joint_names:
                raise ValueError("Master reference joint order mismatch")
            if tuple(arrays["body_names"].tolist()) != result.body_names:
                raise ValueError("Master reference body order mismatch")
            body_count = len(result.body_names)
            expected = {"qpos": (len(times), 55), "qvel": (len(times), 54),
                        "joint_feedforward": (len(times), 48),
                        "body_pos_w": (len(times), body_count, 3),
                        "body_quat_w": (len(times), body_count, 4),
                        "body_lin_vel_w": (len(times), body_count, 3),
                        "body_ang_vel_w": (len(times), body_count, 3)}
            for key, shape in expected.items():
                if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"Invalid master reference {key}")
            for values in (arrays["qpos"][:, 3:7], arrays["body_quat_w"]):
                if not np.allclose(np.linalg.norm(values, axis=-1), 1., atol=1e-5):
                    raise ValueError("Master reference quaternions must be normalized wxyz")
            for key, view in motion.items():
                master = (arrays["qpos"][:, 7:] if key == "joint_pos" else
                          arrays["qvel"][:, 6:] if key == "joint_vel" else arrays[key])
                if view.shape != master[::4].shape or not np.array_equal(view, master[::4]):
                    raise ValueError(f"WBC 50 Hz {key} differs from its 200 Hz master view")
        target = np.asarray(clip["target_w"], dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Invalid frozen world target")
        context_sizes.add(len(clip.get("launch_features") or []))
    if len(context_sizes) != 1:
        raise ValueError("All reference variants require the same G1 feature layout")
    return result
