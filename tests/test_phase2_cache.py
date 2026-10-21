"""Synthetic cache identity/bounds tests; no generated physical references."""
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from human_punching import phase2_cache as module
from human_punching.phase2_cache import (CacheRejected, ExactReferenceCache,
    ReferenceKey, ReferenceRegistration, ResidentCapacityError, UnsupportedReference)


def key(**changes):
    return replace(ReferenceKey("jab", .6, "original", "snapshot:test", (1., 0., 1.4),
        "jab_guard_v1", "jab_support_v1", "a"*64, 17, "train"), **changes)


def registration(path, request=None):
    return ReferenceRegistration(request or key(), path,
        hashlib.sha256(path.read_bytes()).hexdigest(), {"phase": {"launch_tick": 80}})


def archive(tmp_path, filename="ref.npz", **arrays):
    # The runner supplies a fresh human-punching-owned pytest basetemp.
    assert tmp_path.resolve().is_relative_to(module.PROJECT_ROOT)
    path = tmp_path/filename
    np.savez_compressed(path, **(arrays or {"qpos": np.arange(8, dtype=np.float64)}))
    return path


def malformed_archive(tmp_path, member, data):
    path = tmp_path/"malformed.npz"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr(member, data)
    return path


def test_exact_identity_never_substitutes_and_returns_unqualified_metadata(tmp_path):
    path = archive(tmp_path)
    cache = ExactReferenceCache([registration(path)])
    first = cache.lookup(key())
    assert not first.cache_hit and first.payload_bytes == 64
    assert not first.physical_qualification and not first.real_time_qualified
    assert first.within_20ms == (first.total_wall_s <= .020)
    assert first.lookup_wall_s + first.hash_wall_s + first.load_wall_s == pytest.approx(first.total_wall_s)
    assert cache.lookup(key()).cache_hit
    for request in (key(requested_ttc_s=.600000001), key(target_w=(1.000000001, 0., 1.4)),
                    key(style="other"), key(group_split="test"), key(command_group_id=18),
                    key(support_version="other"), key(start_version="other"),
                    key(target_identity="other"), key(recipe_sha256="b"*64)):
        with pytest.raises(UnsupportedReference, match="no nearest"):
            cache.lookup(request)
    with pytest.raises(TypeError):
        first.metadata["phase"]["launch_tick"] = 0


def test_preflight_reads_all_headers_without_loading_arrays(tmp_path, monkeypatch):
    first = archive(tmp_path, "first.npz")
    second = archive(tmp_path, "second.npz", t=np.arange(4), label=np.array("retimed"))
    calls = []
    original_load = np.load
    monkeypatch.setattr(np, "load", lambda *a, **kw: calls.append(True))
    cache = ExactReferenceCache([registration(first), registration(second, key(command_group_id=18))])
    assert calls == [] and len(cache.preflights) == 2 and cache.resident_bytes == 0
    assert cache.total_uncompressed_bytes == 64 + 32 + 28
    assert cache.preflights[second].numeric_array_bytes == 32
    monkeypatch.setattr(np, "load", original_load)
    assert cache.lookup(key()).payload_bytes == 64


def test_total_cap_is_uncompressed_and_unique_paths_are_counted_once(tmp_path):
    first = archive(tmp_path, "first.npz", a=np.zeros(1024))
    assert first.stat().st_size < 1024
    with pytest.raises(CacheRejected, match="Uncompressed"):
        ExactReferenceCache([registration(first)], total_bytes=1024)
    cache = ExactReferenceCache([registration(first), registration(first, key(command_group_id=18))])
    assert cache.total_uncompressed_bytes == 8192 and len(cache.preflights) == 1
    cache.lookup(key()); cache.lookup(key(command_group_id=18))
    assert cache.resident_references == 1
    second = archive(tmp_path, "second.npz", a=np.zeros(1024))
    with pytest.raises(CacheRejected, match="All unique"):
        ExactReferenceCache([registration(first), registration(second, key(command_group_id=18))], total_bytes=12000)
    with pytest.raises(CacheRejected, match="2 GiB"):
        ExactReferenceCache([registration(first)], total_bytes=2*1024**3+1)


@pytest.mark.parametrize("array", [np.array([object()], dtype=object),
    np.zeros(1, dtype=[("x", "f8")]), np.array(["2026-10-21"], dtype="datetime64[D]")])
def test_rejects_object_structured_and_unreviewed_dtypes_before_load(tmp_path, monkeypatch, array):
    path = archive(tmp_path, a=array)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Preflight must not load arrays"))
    with pytest.raises(CacheRejected, match="dtypes"):
        ExactReferenceCache([registration(path)])


def test_malformed_lengths_and_member_names_are_rejected(tmp_path):
    stream = io.BytesIO(); np.save(stream, np.arange(2))
    path = malformed_archive(tmp_path, "a.npy", stream.getvalue()[:-1])
    with pytest.raises(CacheRejected, match="length differ"):
        ExactReferenceCache([registration(path)])
    path = malformed_archive(tmp_path, "../a.npy", stream.getvalue())
    with pytest.raises(CacheRejected, match="unique flat"):
        ExactReferenceCache([registration(path)])


def test_hash_and_source_change_rejected(tmp_path):
    path = archive(tmp_path)
    with pytest.raises(CacheRejected, match="hash mismatch"):
        ExactReferenceCache([replace(registration(path), sha256="b"*64)])
    cache = ExactReferenceCache([registration(path)])
    cache.lookup(key())
    np.savez_compressed(path, a=np.zeros(8))  # Only this test's disposable fixture.
    with pytest.raises(CacheRejected, match="Source identity changed"):
        cache.lookup(key())


def test_symlink_outside_owned_root_is_rejected(tmp_path):
    path = tmp_path/"escape.npz"
    path.symlink_to("/etc/hosts")
    with pytest.raises(CacheRejected, match="inside human-punching"):
        ExactReferenceCache([ReferenceRegistration(key(), path, "a"*64)])


def test_arrays_and_metadata_are_immutable_and_fortran_values_preserved(tmp_path):
    values = np.asfortranarray(np.arange(12).reshape(3, 4))
    path = archive(tmp_path, q=values, name=np.array("control"))
    cache = ExactReferenceCache([registration(path)])
    with cache.borrow_arrays(key()) as (arrays, result):
        np.testing.assert_array_equal(arrays["q"], values)
        assert arrays["q"].flags.f_contiguous and not arrays["q"].flags.writeable
        with pytest.raises(ValueError):
            arrays["q"].setflags(write=True)
        with pytest.raises(TypeError):
            arrays["q"] = values
        assert result.resident_bytes == values.nbytes + 28


def test_lru_and_pinning_bound_residency_without_disk_mutation(tmp_path):
    paths = [archive(tmp_path, f"ref{i}.npz", q=np.arange(8)+i) for i in range(3)]
    registrations = [registration(path, key(command_group_id=17+i)) for i, path in enumerate(paths)]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    cache = ExactReferenceCache(registrations, resident_references=2, resident_bytes=128)
    cache.lookup(key()); cache.lookup(key(command_group_id=18)); cache.lookup(key())
    cache.lookup(key(command_group_id=19))
    assert cache.resident_bytes == 128 and cache.resident_references == 2 and cache.evictions == 1
    assert cache.lookup(key()).cache_hit
    assert not cache.lookup(key(command_group_id=18)).cache_hit
    with cache.borrow_arrays(key()) as (a, _):
        with cache.borrow_arrays(key(command_group_id=18)):
            with pytest.raises(ResidentCapacityError, match="Pinned"):
                cache.lookup(key(command_group_id=19))
            np.testing.assert_array_equal(a["q"], np.arange(8))
    cache.lookup(key(command_group_id=19))
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == hashes


def test_duplicate_keys_conflicting_hashes_and_group_splits_rejected(tmp_path):
    path = archive(tmp_path)
    item = registration(path)
    with pytest.raises(CacheRejected, match="Duplicate exact"):
        ExactReferenceCache([item, item])
    with pytest.raises(CacheRejected, match="conflicting hash"):
        ExactReferenceCache([item, replace(item, key=key(command_group_id=18), sha256="b"*64)])
    with pytest.raises(CacheRejected, match="one split"):
        ExactReferenceCache([item, replace(item, key=key(group_split="test"))])
    with pytest.raises(CacheRejected, match="calibration only"):
        ExactReferenceCache([replace(item, key=key(command_group_id=3307))])


def test_slow_lookup_flag_does_not_assert_real_time(tmp_path, monkeypatch):
    path = archive(tmp_path)
    cache = ExactReferenceCache([registration(path)])
    ticks = iter(np.arange(0., 100., .01))
    monkeypatch.setattr(module.time, "perf_counter", lambda: float(next(ticks)))
    result = cache.lookup(key())
    assert result.total_wall_s > .020 and not result.within_20ms
    assert not result.real_time_qualified and not result.physical_qualification


def test_resident_bytes_limit_and_invalid_keys_fail_closed(tmp_path):
    path = archive(tmp_path)
    with pytest.raises(CacheRejected, match="Uncompressed"):
        ExactReferenceCache([registration(path)], resident_bytes=32)
    for changes in ({"requested_ttc_s": .49}, {"requested_ttc_s": float("nan")},
                    {"target_w": (1., float("inf"), 2.)}, {"group_split": "nearest"}):
        with pytest.raises(CacheRejected):
            key(**changes)
