"""Bounded aggregate engineering receipts; never held-out acceptance evidence."""
import hashlib
import json
from pathlib import Path

import torch

from .readiness_clock import Failure, Phase

START_TYPES = ("strict_guard_settling", "training_reference_frame_zero", "unclassified_legacy")


class AggregateAttemptSink:
    """Count every terminal attempt with constant memory and bounded disk use.

    Individual trajectories/context are intentionally not retained by this
    capacity/engineering sink. Acceptance evaluation requires its independent
    complete trace recorder. Call flush() at each phase boundary and close.
    Crash loss is at most flush_every_batches received batches since the last
    successful fsync/replace; the durable report labels that limitation.
    """
    def __init__(self, output, *, shared_identity, flush_every_batches=128):
        from human_punching.wbc.provenance import writable_path
        if type(flush_every_batches) is not int or flush_every_batches < 1:
            raise ValueError("Positive bounded flush batch count required")
        if not isinstance(shared_identity, dict) or not shared_identity:
            raise ValueError("Shared source/reference/controller/schema identity is required")
        self.output = writable_path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        identity = json.dumps(shared_identity, sort_keys=True, allow_nan=False).encode()
        identity_file = self.output/"identity.json"
        with identity_file.open("xb") as stream:
            stream.write(identity)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.identity_sha256 = hashlib.sha256(identity).hexdigest()
        self.flush_every_batches = flush_every_batches
        self.batches = self.attempts = self.launches = self.contacts = self.recoveries = 0
        self.by_family = [0]*4
        self.by_failure = [0]*len(Failure)
        self.last_sequence = -1
        self.by_start_type = [[0]*(5+len(Failure)) for _ in START_TYPES]
        self.support_grace_ticks = self.support_grace_attempts = 0

    def __call__(self, receipt):
        if receipt["sequence"] != self.last_sequence+1:
            raise ValueError("Terminal batches must be consumed exactly once in order")
        r, e = receipt["readiness"], receipt["events"]
        count = len(receipt["environment_ids"])
        family = receipt["requested_family"]
        if family.shape != (count,) or r["phase"].shape != (count,):
            raise ValueError("Aggregate sink requires selected-row receipts")
        if receipt.get("acceptance_eligible", False):
            raise ValueError("Aggregate engineering receipts cannot claim acceptance")
        mode = torch.full_like(family, 2)
        coherent = (((r["phase"] == Phase.COMPLETE) | (r["phase"] == Phase.FAILED)).all()
            & (~e["has_contact"] | r["launched"]).all()
            & ((r["phase"] != Phase.COMPLETE) | r["launched"]).all())
        grace = torch.zeros_like(family)
        initialization = receipt.get("initialization")
        if initialization is not None:
            from human_punching.wbc.runtime.training_initialization import TrainingInitializationCfg
            if (not isinstance(initialization, dict)
                    or initialization.get("schema") != "human_wbc_training_initialization_v1"):
                raise ValueError("Explicit coherent initialization metadata required")
            try:
                config = TrainingInitializationCfg(**initialization["config"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Malformed aggregate initialization configuration") from error
            for key, dtype in (("start_type_codes", torch.int64), ("strict_launch_eligible", torch.bool),
                    ("strict_ready_launch", torch.bool), ("training_reference_start", torch.bool),
                    ("initial_support_grace_ticks_used", torch.int64)):
                value = initialization.get(key)
                if (not isinstance(value, torch.Tensor) or value.shape != (count,)
                        or value.dtype != dtype or value.device != family.device):
                    raise ValueError(f"Malformed aggregate initialization field {key}")
            mode = initialization["start_type_codes"]
            grace = initialization["initial_support_grace_ticks_used"]
            coherent &= (((mode == 0) | (mode == 1)).all()
                & (initialization["strict_launch_eligible"] == (mode == 0)).all()
                & (initialization["strict_ready_launch"] == (r["launched"] & (mode == 0))).all()
                & (initialization["training_reference_start"] == (r["launched"] & (mode == 1))).all()
                & ((mode != 1) | config.enabled).all()
                & (grace >= 0).all() & (grace <= config.initial_support_grace_ticks).all()
                & ((mode == 1) | (grace == 0)).all())
            if (initialization.get("initial_support_grace_ticks_limit") != config.initial_support_grace_ticks
                    or initialization.get("physical_reset_only") is not True
                    or initialization.get("interior_rsi") is not False):
                raise ValueError("Initialization scope/grace metadata is incoherent")
        by_mode = []
        for index in range(len(START_TYPES)):
            selected = mode == index
            by_mode.extend((torch.stack((selected.sum(), (selected & r["launched"]).sum(),
                (selected & e["has_contact"]).sum(), (selected & (r["phase"] == Phase.COMPLETE)).sum(),
                (selected & (r["phase"] == Phase.FAILED)).sum())),
                torch.bincount(r["failure_category"][selected], minlength=len(Failure))))
        # Only a fixed-size integer summary leaves the GPU; never all states,
        # G1 contexts, nominal postures or per-attempt JSON.
        summary = torch.cat((torch.stack((
            r["launched"].sum(), e["has_contact"].sum(), (r["phase"] == Phase.COMPLETE).sum())),
            torch.bincount(family, minlength=4),
            torch.bincount(r["failure_category"], minlength=len(Failure)),
            torch.stack((coherent.long(), grace.sum(), (grace > 0).sum())), *by_mode)).cpu().tolist()
        base = 7+len(Failure)
        if (len(summary) != base+3+len(START_TYPES)*(5+len(Failure))
                or sum(summary[3:7]) != count or sum(summary[7:base]) != count):
            raise ValueError("Bounded known family/failure labels required")
        if summary[base] != 1:
            raise ValueError("Initialization labels cannot misclassify training starts as strict-ready launches")
        self.batches += 1
        self.attempts += count
        self.launches += summary[0]
        self.contacts += summary[1]
        self.recoveries += summary[2]
        self.by_family = [a+b for a,b in zip(self.by_family, summary[3:7], strict=True)]
        self.by_failure = [a+b for a,b in zip(self.by_failure, summary[7:base], strict=True)]
        self.support_grace_ticks += summary[base+1]
        self.support_grace_attempts += summary[base+2]
        width = 5+len(Failure)
        for index in range(len(START_TYPES)):
            offset = base+3+index*width
            self.by_start_type[index] = [a+b for a,b in zip(
                self.by_start_type[index], summary[offset:offset+width], strict=True)]
        self.last_sequence = receipt["sequence"]
        if self.batches % self.flush_every_batches == 0:
            self.flush()

    def report(self):
        return {"schema": "human_wbc_aggregate_attempts_v1", "acceptance_eligible": False,
            "mode": "aggregate_engineering_only", "individual_traces_retained": False,
            "identity_sha256": self.identity_sha256, "batches": self.batches,
            "terminal_attempts": self.attempts, "launched_attempts": self.launches,
            "unlaunched_attempts": self.attempts-self.launches, "contact_attempts": self.contacts,
            "recovered_attempts": self.recoveries, "terminal_attempts_by_family": self.by_family,
            "launch_count_semantics": "all_reference_starts_including_training_initialization",
            "strict_ready_launch_attempts": self.by_start_type[0][1],
            "training_reference_start_attempts": self.by_start_type[1][1],
            "unclassified_launch_attempts": self.by_start_type[2][1],
            "initialization_metadata_missing_attempts": self.by_start_type[2][0],
            "terminal_attempts_by_start_type": {name: {
                "attempts": values[0], "reference_starts": values[1], "contacts": values[2],
                "recoveries": values[3], "failures": values[4],
                "failure_categories": {entry.name.lower(): values[5+int(entry)] for entry in Failure}}
                for name, values in zip(START_TYPES, self.by_start_type, strict=True)},
            "initial_support_grace_ticks_used": self.support_grace_ticks,
            "initial_support_grace_attempts": self.support_grace_attempts,
            "terminal_attempts_by_failure": {entry.name.lower(): self.by_failure[int(entry)] for entry in Failure},
            "last_sequence": self.last_sequence, "flush_every_batches": self.flush_every_batches,
            "maximum_unflushed_batches_on_crash": self.flush_every_batches,
            "incomplete_inflight_attempts_not_in_terminal_counts": True}

    def flush(self):
        """Atomically replace this sink's own bounded summary, never run evidence."""
        import os
        path = self.output/"aggregate.json"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            stream.write(json.dumps(self.report(), sort_keys=True, allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        directory = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
