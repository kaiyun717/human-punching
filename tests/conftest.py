"""Reproducible Task 10 fixtures from declared source data, without dynamics.

The common bundle requires the committed four-specialist release, actual accepted
G1 snapshot payloads/nominal plant, and the pinned mjlab source export installed by
scripts/setup_wbc.py. It never reuses an untracked diagnostic run or qualifies a
reference for learning. All generated files and caches stay in human-punching.
"""
from pathlib import Path
import tempfile

import pytest


@pytest.fixture(scope="session")
def task10_common_bundle():
    import mujoco
    from human_punching.wbc.candidates import CandidateSpec, build_candidate_bundle
    from human_punching.wbc.provenance import PROJECT_ROOT, configure_local_caches, writable_path
    from human_punching.wbc.references import FAMILIES
    from human_punching.wbc.runtime.bundle import load_runtime_bundle

    configure_local_caches()
    inputs = {
        "release_root": PROJECT_ROOT/"outputs/four-punch-milestone",
        "bank_root": PROJECT_ROOT.parent/"robust_sav/constants/boxing/snapshots/v1_20260921",
        "g1_asset_root": PROJECT_ROOT/".cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1",
        "nominal_plant_path": PROJECT_ROOT.parent/"robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json",
    }
    absent = [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]
    if absent:
        pytest.fail("Task 10 common-bundle fixture requires the preserved raw-data prerequisites:\n" + "\n".join(absent))
    cache = writable_path(PROJECT_ROOT/".cache/task10/pytest")
    cache.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="common_bundle_", dir=cache))/"bundle"
    specs = [CandidateSpec(15744, family, .8, 0., ttc)
             for family, ttc in zip(FAMILIES, (.7, .7, 1.1, .95), strict=True)]

    def no_dynamics(*args, **kwargs):
        pytest.fail("Task 10 fixture generation must not integrate physics")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            patch.setattr(mujoco, name, no_dynamics)
        result = build_candidate_bundle(specs, **inputs, output_root=output,
                                        backend="refined_v3", ik_iterations=35, guard_iterations=120)
    if not result["all_families_generated"] or result["qualified_for_training"]:
        pytest.fail(f"Four-family diagnostic fixture generation failed: {result}")
    return load_runtime_bundle(output)
