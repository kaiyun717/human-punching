"""Frozen Task08 actor, delivered-image history and unchanged first-event rules.

The selected H1-final archive is immutable. This diagnostic explicitly replaces
its historical camera pitch with the accepted +25-degree mount; it does not
claim that the new H2 motions/camera reproduce Task08's original evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch

from .g1 import ROBUST, _imports

_imports()
from boxing.training.residual_observations import ResidualObservationHistory

H1_FINAL_PATH = ROBUST / "constants/boxing/residual_policies/warm948/v1_20261008/s_h1_normal/policy_latest_000030/policy_latest_000030.zip"
H1_FINAL_SHA256 = "116f37b6f99c9cd889def4959a5f93c96a521d83e041ad3b415bd9e53275b1cb"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(path=H1_FINAL_PATH, sha256=H1_FINAL_SHA256):
    """Bounded pinned loading with an explicit, pitch-only runtime difference.

    The shared full learner loader requires exact historical camera identity.
    This inference-only loader instead checks that exact historical identity
    against the current contract with its camera explicitly set to old15down.
    No global constant is changed and no optimizer/replay is constructed.
    """
    from boxing.perception.rectified_camera import RectifiedCameraConfig
    from boxing.perception.runtime_contract import contract_manifest, CHECKPOINT_SCHEMA_ID
    from boxing.perception.dependencies import validate_execution_dependencies
    path = Path(path).resolve(strict=True)
    limit = 32 * 1024 ** 2
    if not 0 < path.stat().st_size <= limit:
        raise ValueError("Expected a bounded Task08 archive")
    archive = path.read_bytes()
    if hashlib.sha256(archive).hexdigest() != sha256:
        raise ValueError("Selected Task08 archive SHA256 differs")
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        info = source.infolist()
        if (len(info) != 2 or {v.filename for v in info} != {"manifest.json", "weights.pt"}
                or sum(v.file_size for v in info) > limit
                or any(v.compress_type != zipfile.ZIP_STORED or v.flag_bits & 1 for v in info)
                or source.getinfo("manifest.json").file_size > 512 * 1024):
            raise ValueError("Unexpected Task08 archive envelope")
        manifest = json.loads(source.read("manifest.json"))
        payload = source.read("weights.pt")
    if (manifest["schema_id"] != CHECKPOINT_SCHEMA_ID
            or manifest["checkpoint_mode"] != "weights_only_inference"
            or hashlib.sha256(payload).hexdigest() != manifest["weights_sha256"]):
        raise ValueError("Task08 checkpoint schema or payload identity differs")
    current = contract_manifest()
    if current["camera"]["mount_pitch_down_deg"] != -25.:
        raise ValueError("Diagnostic requires canonical +25-degree upward camera")
    historical = deepcopy(current)
    historical["camera"] = RectifiedCameraConfig(mount_pitch_down_deg=15.).record()
    historical["depth"]["camera"] = deepcopy(historical["camera"])
    historical = json.loads(json.dumps(historical))
    saved = manifest["config"]["runtime_contract"]
    perception = manifest["identity"]["perception"]
    if (saved != historical or perception["runtime_contract"] != saved
            or manifest["config"]["runtime_contract_sha256"] != _json_hash(saved)
            or perception["runtime_contract_sha256"] != _json_hash(saved)):
        raise ValueError("Checkpoint differs beyond the explicitly authorized camera pitch")
    validate_execution_dependencies(manifest["identity"]["nominal_identity"])
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        infos = inner.infolist()
        if (not infos or len(infos) > 128 or sum(v.file_size for v in infos) > limit
                or any(v.compress_type != zipfile.ZIP_STORED or v.flag_bits & 1 for v in infos)):
            raise ValueError("Invalid tensor archive")
    weights = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or set(weights) != set(manifest["network_tensors"]):
        raise ValueError("Checkpoint tensor names differ")
    for name, value in weights.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or not torch.isfinite(value).all()
                or manifest["network_tensors"][name] != {"shape": list(value.shape), "dtype": str(value.dtype)}):
            raise ValueError(f"Invalid saved tensor: {name}")
    return manifest, weights, dict(path=str(path), sha256=sha256,
        weights_sha256=manifest["weights_sha256"], historical_runtime_sha256=_json_hash(saved),
        diagnostic_runtime_sha256=_json_hash(current), archived_pitch_up_deg=-15.,
        diagnostic_pitch_up_deg=25., runtime_change="mount pitch only; original archive unchanged",
        selection="user-selected warm H1 final" if sha256 == H1_FINAL_SHA256 else "explicit pinned override",
        promotion=False)


class FrozenTask08Policy:
    """Native residual + exactly one frozen Fresh96 normalization/composition."""

    def __init__(self, g1, *, checkpoint=H1_FINAL_PATH, sha256=H1_FINAL_SHA256):
        from boxing.training.residual_learning import ResidualCompositeActor, ResidualCritic
        from boxing.training.control_models import frozen_nominal_digest
        self.device = torch.device(g1.device)
        manifest, weights, self.identity = checkpoint_payload(checkpoint, sha256)
        self.manifest = manifest
        self.actor = ResidualCompositeActor(deepcopy(g1.nominal).cpu(),
            g1.scale, g1.default, g1.ctrl_lower, g1.ctrl_upper,
            profile_id=manifest["config"]["profile_id"], seed=manifest["config"]["seed"])
        if self.actor.action_manifest() != manifest["identity"]["action"]:
            raise ValueError("Executing action composition/limits differ from checkpoint")
        if frozen_nominal_digest(self.actor) != manifest["identity"]["frozen_nominal_state_sha256"]:
            raise ValueError("Fresh96 actor/embedded normalizer differs from checkpoint")
        self.q1, self.q2 = ResidualCritic(), ResidualCritic()
        modules = {"residual": self.actor.residual, "q1": self.q1, "q2": self.q2}
        expected = {f"{group}.{name}" for group, module in modules.items() for name in module.state_dict()}
        if set(weights) != expected:
            raise ValueError("Checkpoint network architecture differs")
        for group, module in modules.items():
            module.load_state_dict({name[len(group)+1:]: value for name, value in weights.items()
                                    if name.startswith(group + ".")}, strict=True)
        for module in (self.actor, self.q1, self.q2):
            module.to(self.device).eval().requires_grad_(False)
        self._initial_hashes = self._hashes()
        self.identity.update(observation_schema="task08_actor948_critic2011_delay_v1",
            actor_features=948, nominal_features=96, profile_id=manifest["config"]["profile_id"],
            normalization="embedded Fresh96 once; residual uses original-unit observation",
            physical_residual_limit_rad=.30, action_history="executed limited physical target inverted to raw manager action",
            learning=False, action_exploration=False, original_network_hashes=self._initial_hashes)

    def _hashes(self):
        return {name: _module_hash(getattr(self, name)) for name in ("actor", "q1", "q2")}

    @torch.no_grad()
    def action(self, actor948):
        if actor948.device != self.device or actor948.dtype != torch.float32:
            raise ValueError("Policy observation must use the native float32 device")
        # The original forward proposes unclipped physical targets; evaluation
        # always applies this same native hard-joint limiter before execution.
        return self.actor.deterministic_actions(actor948).collection.u_applied

    @torch.no_grad()
    def inference_health(self, actor948=None):
        generator = torch.Generator(device="cpu").manual_seed(270507)
        if actor948 is None:
            actor948 = (torch.randn((32, 948), generator=generator) * .05).to(self.device)
        critic = (torch.randn((len(actor948), 2011), generator=generator) * .05).to(self.device)
        actions = self.action(actor948)
        repeated = self.action(actor948)
        bounded = self.actor.bounded_residual(actor948)
        q = [module(critic, actions, torch.zeros_like(actions)) for module in (self.q1, self.q2)]
        finite = all(bool(torch.isfinite(value).all()) for value in (actor948, actions, bounded, *q))
        error = float((actions - repeated).abs().max())
        return dict(passed=finite and error == 0., finite=finite, repeated_action_max_abs_error=error,
            residual_max_abs=float(bounded.abs().max()), physical_target_max_abs=float(actions.abs().max()),
            q1_range=[float(q[0].min()), float(q[0].max())], q2_range=[float(q[1].min()), float(q[1].max())],
            probe_rows=len(actor948), critic_probe="synthetic only, never used for control/outcomes", learning=False)

    def assert_unchanged(self):
        actual = self._hashes()
        if actual != self._initial_hashes:
            raise AssertionError("Frozen policy, normalization or critic changed")
        return dict(passed=True, original=self._initial_hashes, final=actual,
                    checkpoint_sha256=self.identity["sha256"])


class Task08ActorHistory:
    """Accepted term ordering plus the19-slot history of DELIVERED packets.

    Never redraw measurements or change action-manager history. Only O_rsa and
    O_stand are emitted; unused punch/critic slots are zero-filled and discarded.
    The actor receives no clean depth, timestamps, mask, completion or family.
    """

    def __init__(self, count, device, dtype=torch.float32):
        self.history = ResidualObservationHistory(count, device=device, dtype=dtype)
        self.device, self.dtype, self.count = self.history.device, dtype, count
        self._all = torch.ones(count, dtype=torch.bool, device=self.device)
        self._zero_punch = torch.zeros(count, 13, device=self.device, dtype=dtype)
        self._zero_height_fault = torch.zeros(count, device=self.device, dtype=dtype)
        shape = (count, self.history.capacity)
        self.metadata = {
            "frame_ids": torch.zeros(shape, dtype=torch.long, device=self.device),
            "capture_timestamps": torch.zeros(shape, dtype=torch.float64, device=self.device),
            "padding": torch.zeros(shape, dtype=torch.bool, device=self.device),
        }

    def _tensor(self, value, dtype):
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    @torch.no_grad()
    def acquire(self, g1, exposure, time_s, *, reset=False, mask=None):
        mask = self._all if mask is None else mask
        times = torch.full((self.count,), float(time_s), dtype=torch.float64, device=self.device)
        noisy96 = g1.actor99[:, :96]
        clean108 = torch.cat((g1.clean111[:, :96], g1.clean111[:, 99:]), -1)
        qpos = getattr(g1.data.qpos, "_tensor", g1.data.qpos)
        height = qpos[:, 2]
        source = self._tensor(exposure["capture_timestamps"], torch.float64)
        torch._assert_async(torch.all(~mask | (source == times)), "History must use this actual control-tick acquisition")
        if not reset:
            latest = self.history._timestamps.gather(1, self.history._cursor[:, None])[:, 0]
            torch._assert_async(torch.all(~mask | ((times-latest-.02).abs() < 1e-8)), "Acquire history only at20ms ticks")
        operation = self.history.reset if reset else self.history.append
        operation(mask, noisy96, clean108, self._zero_punch,
            self._tensor(exposure["delivered_normalized"], self.dtype), height, times,
            clean_normalized_depth=self._tensor(exposure["delivered_clean_normalized"], self.dtype),
            height_epsilon=self._zero_height_fault)
        write = (mask[:, None].expand(-1, self.history.capacity) if reset else
                 mask[:, None] & (self.history._slots[None] == self.history._cursor[:, None]))
        for name, exposure_name in (("frame_ids", "delivered_frame_ids"),
            ("capture_timestamps", "delivered_capture_timestamps"), ("padding", "padding")):
            target = self.metadata[name]
            value = self._tensor(exposure[exposure_name], target.dtype)
            target.copy_(torch.where(write, value[:, None], target))
        emitted = self.history.observe(height, self._all & False)
        slots = (self.history._cursor[:, None] - self.history._visual_lags[None]) % self.history.capacity
        packet = {name: value[self.history._rows, slots].clone() for name, value in self.metadata.items()}
        prefill = self.history._capture_counts[:, None] - 1 < self.history._visual_lags[None]
        return dict(O_rsa=emitted["O_rsa"], O_stand=emitted["O_stand"],
            proprioception_timestamps=emitted["proprioception_timestamps"],
            visual_delivery_timestamps=emitted["visual_timestamps"],
            visual_timestamps=packet["capture_timestamps"], visual_frame_ids=packet["frame_ids"],
            visual_padding=packet["padding"] | prefill, visual_age_s=times[:, None]-packet["capture_timestamps"],
            delay_ticks=self._tensor(exposure["delay_ticks"], torch.long))


class Task08EpisodeOutcomes:
    """Shared safety/arrival precedence at every5ms sample, including reset.

    Recovery-complete is the immutable reference's end under source-speed play.
    It merely opens the arrival gate; success additionally requires safety and
    stance energy<=50. The original return+3s cutoff and30s watchdog remain.
    """

    def __init__(self, recovery_end_s, device):
        self.recovery_end_s = torch.as_tensor(recovery_end_s, device=device, dtype=torch.float64).reshape(-1)
        count = len(self.recovery_end_s)
        if not bool(torch.isfinite(self.recovery_end_s).all() & (self.recovery_end_s > 0).all()):
            raise ValueError("Finite positive source recovery-completion times required")
        self.active = torch.ones(count, dtype=torch.bool, device=device)
        self.reason = torch.zeros(count, dtype=torch.long, device=device)
        self.event_time = torch.full((count,), float("nan"), dtype=torch.float64, device=device)
        self.first_return_time = torch.full_like(self.event_time, float("nan"))
        self.hit = torch.zeros_like(self.active)
        self.min_clearance = torch.full_like(self.event_time, float("inf"))
        self.min_height = torch.full_like(self.event_time, float("inf"))

    @torch.no_grad()
    def update(self, geometry, gloves, time_s):
        from boxing.punches.glove_geometry import labels
        from boxing.training.collector import first_event
        times = torch.full_like(self.event_time, float(time_s))
        done = times >= self.recovery_end_s - 1e-10
        safe, arrival, continuation, reason, clearance, hit = labels(geometry, gloves, done, 50.)
        eligible = self.active.clone()
        returned = eligible & done & torch.isnan(self.first_return_time)
        self.first_return_time.copy_(torch.where(returned, times, self.first_return_time))
        self.hit |= eligible & hit
        self.min_clearance.copy_(torch.where(eligible, torch.minimum(self.min_clearance, clearance), self.min_clearance))
        self.min_height.copy_(torch.where(eligible, torch.minimum(self.min_height, geometry["height"]), self.min_height))
        # Shared scheduler starts the three-second recovery window at the first
        # physics sample that actually reports completion, not between samples.
        at_cutoff = torch.isfinite(self.first_return_time) & (times >= self.first_return_time + 3. - 1e-10)
        watchdog = times >= 30.
        admin_reason = torch.where(at_cutoff, 6, torch.full_like(reason, 9))
        stopped, final_reason, administrative = first_event(eligible, safe, arrival, reason,
                                                           at_cutoff | watchdog, admin_reason)
        self.reason.copy_(torch.where(stopped, final_reason, self.reason))
        self.event_time.copy_(torch.where(stopped, times, self.event_time))
        self.active &= ~stopped
        return dict(stopped=stopped, safe=safe, arrival=arrival, continuation=continuation,
                    physical_hit=hit, clearance_m=clearance, recovery_complete=done,
                    administrative=administrative)

    def records(self):
        from boxing.training.collector import OUTCOMES
        def optional(value):
            return float(value) if torch.isfinite(value) else None
        return [dict(disposition=OUTCOMES[int(self.reason[i])], event_time_s=optional(self.event_time[i]),
            observed_hit=bool(self.hit[i]), min_safe_clearance_m=optional(self.min_clearance[i]),
            min_true_height_m=optional(self.min_height[i]), punch_return_time_s=optional(self.first_return_time[i]),
            source_recovery_end_s=float(self.recovery_end_s[i]),
            safe_arrival=bool(self.reason[i] == 1), completed=not bool(self.active[i]))
            for i in range(len(self.reason))]
