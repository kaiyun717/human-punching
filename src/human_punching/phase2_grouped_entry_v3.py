"""Strict saved-checkpoint integrity gate around unchanged grouped original PPO.

The original trainer and its receipts remain untouched. This entry writes one
additional receipt; exact reload parity is necessary, never sufficient, for any
later learned-route credit. Importing this module performs no native work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "task10b_phase2_grouped_entry_integrity_v3"


def _owned(path):
    candidate = Path(path).resolve()
    candidate.relative_to(ROOT.resolve())
    if candidate == ROOT.resolve():
        raise ValueError("A project leaf path is required")
    return candidate


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _binding(path):
    path = _owned(path)
    return {"path": str(path), "sha256": _sha(path)}


def _json_evidence(value):
    """Keep malformed/nonfinite parity evidence JSON-safe without crediting it."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: _json_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_evidence(item) for item in value]
    return value


def checkpoint_integrity(result):
    """Pure decision over reported parity; file identity is checked separately."""
    parity = result.get("reload_parity")
    values = parity if isinstance(parity, dict) else {}
    error = values.get("maximum_action_error")
    checks = {
        "parity_present": isinstance(parity, dict),
        "parameters_exact": values.get("parameters_equal") is True,
        "actions_exact": values.get("actions_equal") is True,
        "finite_zero_action_error": (
            type(error) in (int, float) and error == 0 and math.isfinite(error)
        ),
        "original_bounded_stop": result.get("status") == "bounded_stop",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reported_reload_parity": _json_evidence(deepcopy(parity)),
        "original_status": result.get("status"),
    }


def canonical_receipt(value):
    """Exact finite JSON value identity, permitting JSON's tuple-to-array conversion.

    Reject non-string mapping keys and foreign scalar types; do not normalize
    changed values, booleans into numbers, NaN, or infinities into evidence.
    """
    def validate(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Receipt contains nonfinite float")
            return
        if type(item) in (list, tuple):
            for child in item:
                validate(child)
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("Receipt mapping keys must be strings")
            for child in item.values():
                validate(child)
            return
        raise TypeError("Receipt contains unsupported value type")
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def receipt_matches(saved, returned):
    return (type(saved) is dict and type(returned) is dict
            and saved.get("grouped_schema") == "task10b_phase2_grouped_training_receipt_v1"
            and canonical_receipt(saved) == canonical_receipt(returned))


def _original_execute_configuration(path, expected_sha256, output):
    from .phase2_grouped_learning import execute_configuration as original
    return original(path, expected_sha256, output)


def execute_configuration(path, expected_sha256, output):
    """Delegate training once, then seal a separate fail-closed integrity result."""
    folder = _owned(output)
    folder.relative_to((ROOT / "outputs/task10b").resolve())
    if folder.exists():
        raise FileExistsError("Grouped v3 output must be fresh; prior evidence is preserved")
    configuration = _owned(path)
    if _sha(configuration) != expected_sha256:
        raise ValueError("Grouped configuration identity changed before execution")
    original_exception = None
    try:
        result = _original_execute_configuration(configuration, expected_sha256, folder)
        if not isinstance(result, dict):
            raise TypeError("Original grouped execution returned no result dictionary")
    except Exception as error:
        result = {"status": "failed"}
        original_exception = {"type": type(error).__name__, "message": str(error)}

    integrity = checkpoint_integrity(result)
    receipt_path = folder / "grouped_training_receipt.json"
    checkpoint = result.get("checkpoint")
    original_receipt = None
    checkpoint_binding = None
    file_errors = []
    checks = integrity["checks"]
    checks["unchanged_configuration"] = _sha(configuration) == expected_sha256
    checks["original_receipt_matches_returned_result"] = False
    checks["final_checkpoint_identity"] = False
    try:
        original_receipt = _binding(receipt_path)
        saved = json.loads(receipt_path.read_text())
        checks["original_receipt_matches_returned_result"] = receipt_matches(saved, result)
    except (OSError, ValueError, TypeError) as error:
        file_errors.append({"input": "original_receipt", "type": type(error).__name__, "message": str(error)})
    try:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
            raise ValueError("Exact final checkpoint path/SHA binding is required")
        checkpoint_path = _owned(checkpoint["path"])
        if checkpoint_path != _owned(folder / "training/checkpoint_final.pt"):
            raise ValueError("Checkpoint must be this execution's final saved checkpoint")
        checkpoint_binding = _binding(checkpoint_path)
        checks["final_checkpoint_identity"] = checkpoint_binding["sha256"] == checkpoint["sha256"]
    except (OSError, ValueError, TypeError) as error:
        file_errors.append({"input": "final_checkpoint", "type": type(error).__name__, "message": str(error)})

    passed = all(checks.values()) and original_exception is None
    receipt = {
        "schema": SCHEMA,
        **integrity,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "configuration": {"path": str(configuration), "sha256": expected_sha256},
        "original_receipt": original_receipt,
        "checkpoint": checkpoint_binding,
        "original_exception": original_exception,
        "file_errors": file_errors,
        "source": _binding(__file__),
        "original_receipt_overwritten": False,
        "original_ppo_modified": False,
        "receipt_comparison": "exact_finite_JSON_value_v1",
        "integrity_only": True,
        "physical_qualification_credit": False,
        "phase2_complete": False,
        "formal_acceptance": False,
    }
    folder.mkdir(parents=True, exist_ok=True)
    destination = _owned(folder / "grouped_integrity_v3.json")
    with destination.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {
        **result,
        "status": result["status"] if passed else "failed",
        "grouped_integrity_pass": passed,
        "grouped_integrity_v3": _binding(destination),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    result = execute_configuration(options.configuration, options.configuration_sha256, options.output)
    print(json.dumps(_json_evidence(result), sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "bounded_stop" and result["grouped_integrity_pass"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
