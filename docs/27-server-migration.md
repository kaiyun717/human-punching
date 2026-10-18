# Server migration and artifact preservation

Audit date: 15 October 2026. This document distinguishes what is needed to continue development from what is needed to reproduce the research history. No upload, Git staging, commit, push, or ignore-rule change was performed. See [the handoff](26-project-handoff.md) for the scientific state and next milestones, and [the machine-readable inventory](migration-inventory.json) for selected sizes and hashes.

## Recommendation

Transfer the working source tree, the complete `outputs/four-punch-milestone/` release, the exact pinned mjlab source, and the standalone HTML report. Add the four selected DuoBox input files and the G1 asset source when developing new target-conditioned references. Preserve historical experiments in a separate archive; the whole existing `outputs/` tree is about 3.2 GiB allocated, so transferring it intact is a reasonable convenience if space permits. That avoids losing failed-run evidence and historical local links. There is no need to copy Python environments or compilation caches to resume human-policy development.

The current source, tests, scripts, lock files, documentation, and vendored SOMA23 description were already Git tracked at the start of this audit: 135 files, with a clean ordinary `git status`. The new handoff/migration/research-log work is additional working-tree content. Most missing-on-clone material is deliberately ignored generated data, rather than missing implementation code.

The frozen release is approximately **494 MB decimal** (about 473 MiB allocated on this filesystem). Its manifest inventories 1,264 artifacts totaling 493,177,707 bytes, in addition to the top-level manifest and README. It contains all four selected checkpoints, model/runtime XMLs, references, controller schemas, training source snapshots, independent 64-trial evaluations, matched zero-residual baselines, plots, and final videos. Each checkpoint is approximately 4.03 MB. Keeping the complete release is the simplest way to preserve the accepted result and its evidence.

## What to track in Git, LFS, or external storage

| Material | Current location/status | Recommendation |
|---|---|---|
| Implementation, tests, scripts, environment recipes | `src/`, `tests/`, `scripts/`, `pyproject.toml`, lock files; already tracked | Keep in ordinary Git. Include the new handoff and migration documents when you next choose to commit. |
| SOMA23 source description and license notices | `assets/vendor/protomotions/`; already tracked | Keep in Git. This is the required body-source subset; a separate full ProtoMotions checkout is unnecessary for the current pipeline. |
| Accepted scalar body XML, metadata, controller contract, reference manifests, small evaluation summaries | Currently under ignored `outputs/`, and copied into the frozen release | Good candidates for ordinary Git in a future curated path such as `assets/releases/four-punch-v1/`. The XMLs are tens of KB, and the four reference NPZs together are about 1.58 MB. Include checksums and notices. Do not force-add the whole `outputs/` tree or edit the frozen release in place. |
| Four selected checkpoints | `outputs/four-punch-milestone/*/checkpoint.pt`; ignored | Prefer Git LFS if the destination repository supports it, or a versioned artifact archive/object store. They are small enough for ordinary Git technically, but future binary checkpoint revisions will accumulate. |
| Complete accepted evaluations, rollout NPZs, videos, plots, baseline evidence | `outputs/four-punch-milestone/`; ignored | Upload the entire immutable release to the new server and preserve a durable archive. LFS is possible, but external artifact storage plus a tracked manifest is usually simpler for hundreds of MB of evidence. |
| Exact mjlab source export | `.cache/deps/mjlab-8ee51fbc/`; ignored, approximately 61 MiB allocated | **Preserve this source or the corresponding Git object.** It is an important exception to the general rule against copying caches. Prefer a recorded source pin and reproducible checkout/archive; do not casually vendor all of `.cache/`. |
| Original DuoBox source files | Sibling `../DuoBox/`, outside this repository | Transfer the selected four NPY files for new retargeting, or the complete dataset for broader motion/combination work. Preserve dataset provenance and access terms. Use dataset/archive storage or LFS as appropriate, rather than repeatedly committing a 2 GB directory. |
| Task 09 report figures | `docs/figures/task09/`; six new PNG copies plus provenance manifest | Keep these small, selected documentation figures in ordinary Git. They are unchanged copies of ignored evidence and make the LaTeX section portable without historical output directories. No files were staged. |
| Master report source and workspace HTML | `docs/master-report*`; source and workspace HTML already tracked | Keep the source, ledgers, guide, and small workspace HTML in Git. Its linked media still need separate transfer. |
| Standalone HTML report | `outputs/master-report/master-report-standalone.html`; ignored, 60,777,066 bytes | Upload as a convenient single-file research handoff. It embeds the selected images/videos and narrative; it does not embed checkpoints or all linked raw evidence. |
| Historical training runs and failed experiments | Numerous ignored `outputs/punch-*`, `outputs/jab-*`, diagnostic directories and Slurm logs | Archive if preserving exact research history or revisiting failed branches. They are not all required for the next milestone. Retain selected source snapshots/logs and manifests with any archived checkpoints. |
| Python environments and transient caches | `.venv/`, `.venv-physics/`, most `.cache/`, `__pycache__/`, build/egg-info/Ruff/pytest caches | Rebuild on the destination. Do not Git track or copy as the main environment transfer strategy. In particular, CUDA/Warp kernels are hardware-specific generated data. |
| Extracted PDF text | `docs/source-text/`; ignored | Optional convenience copy; reproducible from the PDFs and not required for training. |
| Accidental nested cache directory | `human-punching/human-punching/.cache/pytest/` relative to the workspace parent | Not project source; omit. No removal was performed. |
| Master research log | `../main_mjlab.tex`, outside the `human-punching` Git repository | Track in the research-log/Overleaf repository separately, alongside its bibliography, class/style files, and chosen figures. Copy the updated TeX explicitly; cloning `human-punching` does not include it. |

Do not make the ignored output tree broadly tracked just to migrate. A curated small release-description directory in Git, plus a checksummed binary release elsewhere, is easier to maintain. These are recommendations for a later user-controlled commit; this task did not stage anything.

## Dependency tiers

### A. Evaluate or fine-tune the accepted human policies

Required:

1. The current `human-punching` source and dependency recipes.
2. `outputs/four-punch-milestone/` in full.
3. Exact mjlab revision `8ee51fbcf806a7419189f706d9e394cbeb7790fa`, either as the existing `.cache/deps/mjlab-8ee51fbc/` source export or a Git checkout containing that object.
4. A newly created Python 3.12 environment and a qualified GPU/driver installation.

**Neither the raw DuoBox dataset nor `robust_sav` is needed to evaluate or fine-tune the frozen human references.** The human model uses primitive geometry; no missing human texture/mesh package is required. The frozen release contains the body metadata and reference trajectories needed by the learner.

The reproduction command in [the release README](../outputs/four-punch-milestone/README.md) explicitly supplies the bundled model, metadata, reference, checkpoint, and bank. Do not rely on the training wrapper's historical default `outputs/scalar-reference-v3/jab.npz`; that directory is not part of this minimal transfer. Use `--checkpoint` for evaluation or checkpoint-based training continuation. Training restores policy weights, optimizer state, and the update counter, but does not restore simulator state or random-number-generator state; this is not a bit-for-bit resume. Supply the recorded training arguments explicitly: evaluation restores saved controller arguments, whereas training uses the supplied arguments and checks compatibility. Use `--initialize-policy` only when intentionally transferring weights into a new training experiment with a fresh optimizer.

Historical absolute paths inside manifests and checkpoint arguments are provenance, not instructions to recreate the old `/home/gridsan/...` layout. Supply the destination's current paths explicitly. Preserve the immutable bundle as received rather than rewriting its internal historical records.

### B. Generate new targets, timing schedules, and guard transitions

Also preserve these four source files beneath a sibling `DuoBox/fbx_export_joints/` directory, or pass a different dataset root to the CLI:

| Punch | Raw file | Bytes |
|---|---|---:|
| Jab | `000_ma_jab_liu_defense_joints.npy` | 3,544,713 |
| Cross | `003_ma_cross_liu_defense_joints.npy` | 3,976,137 |
| Left hook | `006_ma_leadhook_liu_shanbi_joints.npy` | 6,915,465 |
| Right hook | `007_ma_rearhook_liu_shanbi_joints.npy` | 5,967,945 |

Together these are **20,404,260 bytes**, versus approximately 2.0 GiB allocated for the complete extracted DuoBox directory. The original zip occupies approximately 1.2 GiB; copying both the complete extraction and zip is redundant for active development. For genuine combination motion research, retaining the full dataset is useful because the four isolated crops alone do not provide combination transitions. Exact selected-source hashes are in `src/human_punching/motion.py` and [the migration inventory](migration-inventory.json).

For immediate reference work, also copy the small generated source banks `outputs/scalar-four-reference-v1/`, `outputs/scalar-hook-retime-v1/`, and `outputs/scalar-body/`. The accepted release already holds each chosen reference/body, but these directories preserve convenient pre-retiming comparisons and reference-generation manifests.

### C. Render a human alongside the G1 or regenerate combined scenes

The G1 visual asset is separate from any learned G1 policy. Its XML/meshes occupy about 19 MiB under `mjlab/src/mjlab/asset_zoo/robots/unitree_g1/`, with the mjlab license alongside the source tree. That asset is present both in the sibling checkout and the pinned source export.

The existing scene-building CLI defaults to a **sibling `../mjlab` checkout**. Copying that checkout is a straightforward way to preserve the current entry points and Git provenance. The current sibling HEAD is `033ae22a2c7a30a25a6fa77b16c113ed88dd1b55`; it contains the older accepted training pin as a Git object. Do **not** install this current HEAD as the training dependency: its requirements differ from the accepted stack. The training source is the older pinned export.

An explicit `mjlab_root` can instead be supplied to the Python `create_scene` API when using the pinned source export. An archive without `.git` still contains assets, but scene metadata cannot claim Git-tree equality from an unavailable Git object. Alternatively, transfer an existing self-contained generated scene **with its entire asset directory**, not just `scene.xml`.

The release includes final rendered videos, but does not include a complete G1 scene/mesh package. Replaying those videos needs no G1 asset. Rendering a human-only recorded rollout needs only the bundled human model. Rendering new G1 views requires the G1 asset/scene input described above.

### D. Integrate the actual learned G1 protagonist

Also transfer the `robust_sav` source/configuration and retrieve its accepted G1 checkpoint/export, actor/critic normalizers as applicable, settled stance/snapshot banks, and the selected downstream perception/control artifacts from their original storage. Keep source pins, artifact hashes, observation conventions, and selected nominal policy identity together.

**Those learned G1 payloads are not present in this workspace.** A read-only search of the supplied `robust_sav` tree found no `.pt`, `.pth`, `.ckpt`, or `.npz` payloads. Its README/configuration discuss accepted historical policies and banks, but those references are not the actual files. The human release does not contain them, and its standing G1 visualization is not a substitute for the accepted G1 controller or settled stance.

The current human and G1 clocks agree at 5 ms physics / 20 ms actions. They still require shared-scene qualification because the human pilot uses elliptic contacts and different solver settings from the accepted G1 recipe. Migration should preserve the accepted human release before adapting that configuration. See [the controller interface](24-five-ms-controller-interface.md).

## Environment portability and GPU qualification

Use `requirements-physics.lock` to rebuild the learning environment. It pins MuJoCo 3.11.0, MuJoCo-Warp 3.11.0, Warp 1.14.0, Torch 2.8.0+cu128, RSL-RL 5.5.0, and the local mjlab source export. The `requirements-physics-installed.lock` file is an observed installation record; it contains an absolute `file:///home/gridsan/.../.cache/deps/mjlab-8ee51fbc` entry and is not a portable installer without adjustment.

`scripts/setup_physics.sh` reads the exact mjlab commit from `../mjlab` and exports it into the ignored source-cache directory. It assumes that Git object exists and currently selects `/usr/bin/python3`. On another server, first ensure that interpreter is the intended Python 3.12, or create the environment explicitly using the already copied source export:

```bash
# Run on the destination, from its human-punching directory.
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache/physics-install/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
test -f .cache/deps/mjlab-8ee51fbc/pyproject.toml
uv venv --python /path/to/python3.12 .venv-physics
uv pip install --python .venv-physics/bin/python \
  --index-strategy unsafe-best-match -r requirements-physics.lock
uv pip check --python .venv-physics/bin/python

# A separate native MuJoCo/preview/test environment.
uv venv --python /path/to/python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-preview.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

The accepted jobs here ran on **V100s in `xeon-g6-volta`**, not on L40S. The existing Slurm wrappers contain this site's partition, GRES names and resource assumptions. On the destination, inspect available partitions/GPU types, adjust the wrapper, record driver/GPU versions, and run a small allocated GPU preflight before long jobs. Do not assume the same partition/account names or that old compiled kernels are portable.

The preflight should exercise Torch gradients, Warp kernels, MuJoCo-Warp stepping/contact, and rendering on the actual allocated GPU. `scripts/physics_preflight.py --output <new-run>/runtime.json --physical-model outputs/four-punch-milestone/jab/model/human.xml` can run without DuoBox or a G1 scene; it expects a Slurm allocation with `CUDA_VISIBLE_DEVICES` set. The existing `physics_probe_volta.sbatch` additionally supplies a historical G1 scene, so adapt/remove those optional scene arguments if using only the minimal transfer. EGL rendering and Warp camera rendering are separate paths; qualify whichever the next experiment uses.

After the preflight, re-evaluate each bundled checkpoint on its bundled 64-trial bank and re-run the independent audit with required 5/20 ms clocks. Retain the original results as the source-server baseline. A new GPU may change numerical behavior even when the source and package versions match; do not silently substitute new results for the frozen release.

## Example transfer commands

These are commands for a later user-initiated transfer; none were executed here. Replace `USER@NEW_SERVER:/absolute/path/HumanBox` with the destination, create that parent directory there, and run from the current `human-punching` project root. The first copy preserves current uncommitted/new documentation too; it omits `.git` history, environments, transient caches and generated outputs. If you want a Git checkout on the destination, clone there first and then overlay this working-tree copy.

```bash
destination='USER@NEW_SERVER:/absolute/path/HumanBox'

# Source working tree, including new handoff documents, without machine state.
rsync -av \
  --exclude='/.git/' --exclude='/.venv/' --exclude='/.venv-physics/' \
  --exclude='/.cache/' --exclude='/.ruff_cache/' --exclude='/.pytest_cache/' \
  --exclude='/outputs/' --exclude='/build/' --exclude='/dist/' \
  --exclude='/human-punching/' --exclude='__pycache__/' --exclude='*.egg-info/' \
  --exclude='/docs/source-text/' \
  ./ "$destination/human-punching/"

# Preserve relative destinations, including the essential source-cache exception.
rsync -avR \
  ./outputs/four-punch-milestone/ \
  ./outputs/master-report/master-report-standalone.html \
  ./outputs/master-report/manifest.json \
  ./.cache/deps/mjlab-8ee51fbc/ \
  "$destination/human-punching/"

# Optional but recommended for the next retargeting/transition milestone.
rsync -avR ./outputs/scalar-body/ ./outputs/scalar-four-reference-v1/ \
  ./outputs/scalar-hook-retime-v1/ "$destination/human-punching/"

# Four raw source files, preserving the sibling DuoBox layout.
(cd .. && rsync -avR \
  ./DuoBox/fbx_export_joints/000_ma_jab_liu_defense_joints.npy \
  ./DuoBox/fbx_export_joints/003_ma_cross_liu_defense_joints.npy \
  ./DuoBox/fbx_export_joints/006_ma_leadhook_liu_shanbi_joints.npy \
  ./DuoBox/fbx_export_joints/007_ma_rearhook_liu_shanbi_joints.npy \
  "$destination/")

# Optional read-only source dependencies and research log; reads the originals.
rsync -av ../mjlab/ "$destination/mjlab/"
rsync -av ../robust_sav/ "$destination/robust_sav/"
rsync -av ../main_mjlab.tex ../RobustSafeArrival-Vision5.pdf \
  ../Task09_Human_Adversary_Implementation_Plan.pdf "$destination/"
```

Copying the sibling mjlab checkout includes its Git history/objects, so `setup_physics.sh` can export the accepted pin. The pinned-source cache and the sibling checkout overlap; preserve both for convenience, or choose the cache-only installation path for the smallest learning-only transfer. Source repositories are read, not modified, by the commands above. No `--delete` is used.

The research-log source references `main.bib` and `ieeeconf.cls`, which are not beside `main_mjlab.tex` in this workspace. Retrieve the complete Overleaf project or those missing inputs separately. The six Task 09 PNGs now have small, byte-identical copies in `docs/figures/task09/` (1,027,216 bytes combined), with an origin/hash manifest. They are carried by the source-tree transfer above and can be tracked in ordinary Git. Upload them using the relative paths in the TeX section; see [the figure guide](task09-latex-figures.md). They are separate from the TeX source.

## Preserve the HTML evidence without all historical runs

The standalone report is sufficient for offline viewing of its narrative and selected embedded videos/images. It is not sufficient to rebuild the workspace HTML or follow every local raw-evidence link.

For a compact, rebuildable visual report, copy every relative media path listed by `outputs/master-report/manifest.json` in addition to the tracked report source. At this audit the manifest lists **59 media files totaling 41,549,262 bytes**. Preserve their relative paths. To prepare a transfer list inside this project:

```bash
python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path('outputs/master-report/manifest.json').read_text())
Path('outputs/master-report/media-files.txt').write_text(
    '\n'.join(sorted(manifest['media'])) + '\n'
)
PY
rsync -av --files-from=outputs/master-report/media-files.txt \
  ./ "$destination/human-punching/"
```

The list is generated from the current manifest, so later report updates can add media without relying on this document's snapshot count. Raw experiment artifacts linked in the narrative may still be absent; archive their original output directories if those links and full forensic reproduction are required.

## Destination acceptance checklist

1. Verify every release artifact against `outputs/four-punch-milestone/manifest.json` using the checksum command in its README; verify selected raw-motion hashes against `migration-inventory.json`.
2. Rebuild the preview/learning environments and record package, source, GPU and driver versions.
3. Adapt Slurm resource declarations and qualify the actual GPU/runtime with a small preflight.
4. Re-run the four frozen evaluation banks and independent physics/precision audits before modifying models, rewards, targets or timing.
5. Preserve that baseline, then develop shared guards, continuous handoffs, bounded aiming/timing variation and the tactical scheduler as described in [the handoff](26-project-handoff.md).

This migration supports continued human-policy research immediately. A physically integrated two-player experiment additionally requires the missing accepted G1 policy/state artifacts and shared-solver qualification.
