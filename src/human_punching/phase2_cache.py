"""Bounded, exact-key offline reference cache; no planning or physics execution.

Every unique archive is hashed and its NPY headers are inspected before any
payload is loaded. Only files resolving inside human-punching are admitted.
Cache eviction changes RAM only. A lookup proves identity, not feasibility.

``lookup`` returns metadata; ``borrow_arrays`` pins immutable resident arrays
for a bounded caller scope. Callers must release references when that scope
ends. The resident budget covers managed payload bytes, not Python/ZIP overhead
or NumPy's temporary decode/copy allocations. Neither measured lookup latency
nor its 20 ms flag establishes end-to-end or delayed-execution real time.

There is intentionally no import of phase2_phases: a caller can pass the exact
key, payload SHA and measured *whole planning* latency to LaunchLatch. A cache
lookup alone never satisfies readiness, continuity or launch commitment.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Mapping
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOTAL_BYTES = 2 * 1024**3
DEFAULT_RESIDENT_BYTES = 256 * 1024**2
DEFAULT_RESIDENT_REFERENCES = 16
MAX_HEADER_BYTES = 16384
MAX_ARRAYS = 256
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


class CacheRejected(ValueError):
    """Archive, request or ownership fails the declared cache contract."""


class UnsupportedReference(KeyError):
    """There is no exact registered key; substitution is forbidden."""


class ResidentCapacityError(CacheRejected):
    """Pinned references prevent a load within the resident limit."""


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise CacheRejected("A lower-case SHA256 identity is required")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise CacheRejected(f"Explicit {name} required")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheRejected(f"{name} must be a positive integer")
    return value


def _freeze_metadata(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CacheRejected("Metadata keys must be strings")
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CacheRejected("Metadata must contain only finite JSON-compatible values")


@dataclass(frozen=True)
class ReferenceKey:
    family: str
    requested_ttc_s: float
    style: str
    target_identity: str
    target_w: tuple[float, float, float]
    start_version: str
    support_version: str
    recipe_sha256: str
    command_group_id: int
    group_split: str

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise CacheRejected("Unknown specialist family")
        if isinstance(self.requested_ttc_s, bool):
            raise CacheRejected("Finite TTC >= 0.50 s required")
        ttc = float(self.requested_ttc_s)
        if not math.isfinite(ttc) or ttc < .5:
            raise CacheRejected("Finite TTC >= 0.50 s required")
        object.__setattr__(self, "requested_ttc_s", ttc)
        for name in ("style", "target_identity", "start_version", "support_version"):
            _text(getattr(self, name), name)
        target = tuple(float(value) for value in self.target_w)
        if len(target) != 3 or not all(math.isfinite(value) for value in target):
            raise CacheRejected("An exact finite three-coordinate target is required")
        object.__setattr__(self, "target_w", target)
        _sha(self.recipe_sha256)
        if (isinstance(self.command_group_id, bool) or not isinstance(self.command_group_id, int)
                or self.command_group_id < 0):
            raise CacheRejected("Explicit nonnegative command group ID required")
        if self.group_split not in ("train", "validation", "test", "calibration"):
            raise CacheRejected("Explicit train/validation/test/calibration group split required")


@dataclass(frozen=True)
class ReferenceRegistration:
    key: ReferenceKey
    path: Path
    sha256: str
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.key, ReferenceKey):
            raise CacheRejected("An exact ReferenceKey is required")
        object.__setattr__(self, "path", Path(self.path))
        _sha(self.sha256)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class ArraySchema:
    name: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool
    nbytes: int


@dataclass(frozen=True)
class PayloadPreflight:
    path: Path
    sha256: str
    arrays: tuple[ArraySchema, ...]
    uncompressed_array_bytes: int
    numeric_array_bytes: int
    archive_bytes: int
    hash_wall_s: float
    header_wall_s: float
    source_stat: tuple[int, ...]


@dataclass(frozen=True)
class CacheLookup:
    key: ReferenceKey
    path: Path
    sha256: str
    arrays: tuple[ArraySchema, ...]
    metadata: Mapping
    payload_bytes: int
    resident_bytes: int
    resident_references: int
    cache_hit: bool
    lookup_wall_s: float
    hash_wall_s: float
    load_wall_s: float
    total_wall_s: float
    within_20ms: bool
    real_time_qualified: bool = False
    physical_qualification: bool = False


def _owned_path(value):
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CacheRejected(f"Reference path cannot be resolved: {value}") from exc
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix != ".npz":
        raise CacheRejected("All reference paths must resolve to regular .npz files inside human-punching")
    return path


def _stat_tuple(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _hash_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _preflight(path, expected_sha, max_payload_bytes):
    before = _stat_tuple(path.stat())
    with path.open("rb") as stream:
        start = time.perf_counter()
        if _hash_stream(stream) != expected_sha:
            raise CacheRejected(f"Payload hash mismatch: {path}")
        hash_wall_s = time.perf_counter() - start
        start = time.perf_counter()
        try:
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                if not 1 <= len(members) <= MAX_ARRAYS:
                    raise CacheRejected("NPZ array count exceeds the bounded schema")
                schemas = []
                names = set()
                array_bytes = numeric_bytes = 0
                for member in members:
                    name = member.filename
                    if (not name.endswith(".npy") or "/" in name or "\\" in name or name == ".npy"
                            or member.is_dir() or member.flag_bits & 1 or name in names):
                        raise CacheRejected("NPZ members must be unique flat unencrypted .npy arrays")
                    names.add(name)
                    if member.file_size > max_payload_bytes + MAX_HEADER_BYTES + 12:
                        raise CacheRejected("Uncompressed NPZ member exceeds the payload budget")
                    with archive.open(member) as array_stream:
                        version = np.lib.format.read_magic(array_stream)
                        if version == (1, 0):
                            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                                array_stream, max_header_size=MAX_HEADER_BYTES)
                        elif version == (2, 0):
                            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                                array_stream, max_header_size=MAX_HEADER_BYTES)
                        else:
                            raise CacheRejected("Only reviewed NPY header versions 1.0 and 2.0 are supported")
                        header_bytes = array_stream.tell()
                    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufcSU":
                        raise CacheRejected("NPZ object/structured/non-numeric non-string dtypes are forbidden")
                    if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
                        raise CacheRejected("Malformed array shape")
                    nbytes = math.prod(shape) * dtype.itemsize
                    if header_bytes + nbytes != member.file_size:
                        raise CacheRejected("NPY header shape/dtype and uncompressed member length differ")
                    array_bytes += nbytes
                    numeric_bytes += nbytes if dtype.kind in "biufc" else 0
                    if array_bytes > max_payload_bytes:
                        raise CacheRejected("Uncompressed payload arrays exceed the resident/total budget")
                    schemas.append(ArraySchema(name[:-4], tuple(shape), dtype.str, fortran, nbytes))
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, NotImplementedError) as exc:
            if isinstance(exc, CacheRejected):
                raise
            raise CacheRejected(f"Malformed NPZ archive: {path}") from exc
        header_wall_s = time.perf_counter() - start
    if before != _stat_tuple(path.stat()):
        raise CacheRejected("Source archive changed during preflight")
    return PayloadPreflight(path, expected_sha, tuple(schemas), array_bytes,
                            numeric_bytes, before[2], hash_wall_s, header_wall_s, before)


class ExactReferenceCache:
    """Thread-safe exact lookup with immutable arrays and RAM-only LRU eviction."""

    def __init__(self, registrations, *, total_bytes=DEFAULT_TOTAL_BYTES,
                 resident_bytes=DEFAULT_RESIDENT_BYTES,
                 resident_references=DEFAULT_RESIDENT_REFERENCES):
        self._total_limit = _positive(total_bytes, "total_bytes")
        self._resident_limit = _positive(resident_bytes, "resident_bytes")
        self._reference_limit = _positive(resident_references, "resident_references")
        # User-authorized ceilings cannot be enlarged through helper options.
        if self._total_limit > DEFAULT_TOTAL_BYTES:
            raise CacheRejected("Initial total reference data ceiling is 2 GiB")
        self._registrations = {}
        unique = {}
        group_splits = {}
        for item in registrations:
            if not isinstance(item, ReferenceRegistration):
                raise CacheRejected("Use explicit ReferenceRegistration entries")
            if item.key in self._registrations:
                raise CacheRejected("Duplicate exact command key")
            group = item.key.command_group_id
            if group in group_splits and group_splits[group] != item.key.group_split:
                raise CacheRejected("All variants of a command group must stay in one split")
            if group == 3307 and item.key.group_split != "calibration":
                raise CacheRejected("Repeated standing command 3307 is calibration only")
            group_splits[group] = item.key.group_split
            path = _owned_path(item.path)
            if path in unique and unique[path] != item.sha256:
                raise CacheRejected("One reference path has conflicting hash identities")
            unique[path] = item.sha256
            self._registrations[item.key] = (path, item)
        if not unique:
            raise CacheRejected("An explicit nonempty reference catalog is required")
        preflights = {}
        total = 0
        for path, expected_sha in unique.items():
            receipt = _preflight(path, expected_sha, min(self._total_limit, self._resident_limit))
            total += receipt.uncompressed_array_bytes
            if total > self._total_limit:
                raise CacheRejected("All unique uncompressed NPZ payloads exceed the total reference budget")
            preflights[path] = receipt
        # No array load occurs until every unique archive has passed preflight.
        self._preflights = MappingProxyType(preflights)
        self._total_bytes = total
        self._resident = OrderedDict()
        self._pins = {}
        self._resident_bytes = 0
        self._evictions = 0
        self._lock = threading.RLock()

    @property
    def preflights(self):
        return self._preflights

    @property
    def total_uncompressed_bytes(self):
        return self._total_bytes

    @property
    def resident_bytes(self):
        with self._lock:
            return self._resident_bytes

    @property
    def resident_references(self):
        with self._lock:
            return len(self._resident)

    @property
    def evictions(self):
        with self._lock:
            return self._evictions

    def _reserve(self, requested):
        victims = []
        remaining_bytes = self._resident_bytes
        remaining_count = len(self._resident)
        for path in self._resident:
            if remaining_bytes + requested <= self._resident_limit and remaining_count < self._reference_limit:
                break
            if not self._pins.get(path, 0):
                victims.append(path)
                remaining_bytes -= self._preflights[path].uncompressed_array_bytes
                remaining_count -= 1
        if remaining_bytes + requested > self._resident_limit or remaining_count >= self._reference_limit:
            raise ResidentCapacityError("Pinned payloads prevent a load within the resident budget")
        for path in victims:
            del self._resident[path]
            self._resident_bytes -= self._preflights[path].uncompressed_array_bytes
            self._evictions += 1

    def lookup(self, key):
        """Return exact-key metadata; never choose another target, TTC or split.

        Every lookup rehashes disk source, including RAM hits. ``lookup_wall_s``
        is bookkeeping/wait overhead; hash/load times are disjoint subsets of
        total measured lookup latency. A failed lookup returns no commitment.
        """
        start = time.perf_counter()
        with self._lock:
            if not isinstance(key, ReferenceKey) or key not in self._registrations:
                raise UnsupportedReference("Unsupported exact request; no nearest-reference fallback")
            path, registration = self._registrations[key]
            receipt = self._preflights[path]
            if _owned_path(path) != path or _stat_tuple(path.stat()) != receipt.source_stat:
                raise CacheRejected("Source identity changed after preflight")
            hit = path in self._resident
            hash_wall_s = load_wall_s = 0.
            with path.open("rb") as stream:
                hash_start = time.perf_counter()
                if _hash_stream(stream) != receipt.sha256:
                    raise CacheRejected("Payload hash mismatch after preflight")
                hash_wall_s += time.perf_counter() - hash_start
                if not hit:
                    self._reserve(receipt.uncompressed_array_bytes)
                    load_start = time.perf_counter()
                    try:
                        arrays = {}
                        with np.load(stream, allow_pickle=False, max_header_size=MAX_HEADER_BYTES) as archive:
                            if set(archive.files) != {schema.name for schema in receipt.arrays}:
                                raise CacheRejected("Loaded archive schema differs from preflight")
                            for schema in receipt.arrays:
                                array = archive[schema.name]
                                if (array.shape != schema.shape or array.dtype.str != schema.dtype
                                        or array.nbytes != schema.nbytes):
                                    raise CacheRejected("Loaded array differs from preflight header")
                                # Bytes backing cannot be made writable through setflags.
                                order = "F" if schema.fortran_order else "C"
                                arrays[schema.name] = np.frombuffer(array.tobytes(order=order),
                                    dtype=array.dtype).reshape(array.shape, order=order)
                                del array
                    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                        if isinstance(exc, CacheRejected):
                            raise
                        raise CacheRejected("Archive decoding failed") from exc
                    load_wall_s = time.perf_counter() - load_start
                    hash_start = time.perf_counter()
                    if _hash_stream(stream) != receipt.sha256:
                        raise CacheRejected("Payload changed while loading")
                    hash_wall_s += time.perf_counter() - hash_start
                    if _stat_tuple(path.stat()) != receipt.source_stat:
                        raise CacheRejected("Source identity changed while loading")
                    self._resident[path] = MappingProxyType(arrays)
                    self._resident_bytes += receipt.uncompressed_array_bytes
            if _stat_tuple(path.stat()) != receipt.source_stat:
                raise CacheRejected("Source identity changed during lookup")
            self._resident.move_to_end(path)
            total = time.perf_counter() - start
            return CacheLookup(key, path, receipt.sha256, receipt.arrays, registration.metadata,
                receipt.uncompressed_array_bytes, self._resident_bytes, len(self._resident), hit,
                max(0., total - hash_wall_s - load_wall_s), hash_wall_s, load_wall_s,
                total, total <= .020)

    @contextmanager
    def borrow_arrays(self, key):
        """Pin arrays until context exit; do not retain views after releasing.

        The returned pair is (immutable array mapping, lookup metadata). Copies
        retained by consumers belong to their own explicitly bounded memory.
        """
        with self._lock:
            result = self.lookup(key)
            self._pins[result.path] = self._pins.get(result.path, 0) + 1
            arrays = self._resident[result.path]
        try:
            yield arrays, result
        finally:
            with self._lock:
                self._pins[result.path] -= 1
                if not self._pins[result.path]:
                    del self._pins[result.path]
