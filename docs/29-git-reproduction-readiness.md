# Git coverage and reproduction on another server

Audit: 18 October 2026. Repository inspected: `human-punching`, commit `e9205a63a188181099b2d08be15913153dbf0ccc`. The working tree was clean before this documentation update. No staging, commits, pushes, dependency installation, or GPU jobs were performed.

## Verdict

**The accepted four-punch results are now preserved in Git.** A checkout contains the selected policies, frozen references, bodies, evaluation evidence, and videos. It can support evaluation and further training after the pinned runtime is rebuilt. It is **not yet a self-contained installer or a complete archive for reproducing the historical training sequence from scratch**.

This changes the 15 October migration assessment: commit `25d0144` added the four-punch bundle, and `813cae8` added the standalone report. An `outputs/` ignore rule does not exclude files that are already tracked.

## What the current commit contains

There are **1,414 tracked files**, totaling **557,336,808 bytes** in the clean working tree. These are ordinary Git files; there is no repository Git LFS configuration or submodule configuration.

| Content | Tracked files | Coverage |
|---|---:|---|
| `outputs/four-punch-milestone/` | 1,266 | Four checkpoints; body and runtime XMLs; metadata; references; controller schemas; training arguments/source snapshots; frozen test banks; 256 learned and 256 baseline rollouts; audits, plots, videos; manifest and README |
| `outputs/master-report/master-report-standalone.html` | 1 | 60,777,066-byte report with selected embedded images/videos |
| `docs/` | 47 | Reports, handoff, figures, report sources, and the WBC audit |
| `src/`, `scripts/`, `tests/` | 88 | Implementation, runners, evaluation and packaging tools, tests |
| `assets/` | 6 | Vendored SOMA23 source subset and license/provenance files |
| Root files | 6 | README, project metadata, ignore rules, and three dependency records |

The release manifest is unchanged:

```text
16eebb942f71f8350bf3d886f8a083eac549a5bc3486fe9d5304b8cde29b0686
```

All **1,264 manifest-listed artifacts** are present in the commit, and every committed blob passes its recorded SHA256 check. The other two release files are its manifest and README. Verification read committed Git blobs, not merely the local files; there were no missing artifacts or hash mismatches.

## What “the same results” requires

| Intended task | Is the tracked content enough? | Additional requirement |
|---|---|---|
| View accepted videos, plots, metrics, and the standalone report | Yes | A browser/media viewer |
| Re-evaluate the four accepted policies at 5 ms physics / 20 ms actions | All experiment inputs are present | Exact mjlab source and a qualified Python/CUDA environment; adapt the runner to the new server |
| Fine-tune the accepted policies on their frozen references | All policy/reference inputs are present | Same runtime; supply explicit bundled paths and training settings |
| Re-render recorded human-only motion | Body and recorded motion are present | Preview dependencies and a working system OpenGL backend |
| Re-render with the standing G1 | Only the existing videos are bundled | G1 XML/meshes, available in the pinned mjlab source; regenerate or transfer the combined scene |
| Regenerate references from DuoBox motion | No | Raw source recordings and the conditioned intermediate references used by the scalar-reference builder |
| Re-run the actual historical learning sequence | No | Earlier initialization checkpoints, ancestor run configurations/source snapshots, and intermediate references |
| Obtain bit-identical retraining or continuation | Not guaranteed | Current saves omit RNG and simulator state; GPU/software numerical differences also matter |
| Run the learned G1 protagonist against the human | No | G1 policy/normalizers and integration work; this was never part of the accepted human-only result |

Neither raw DuoBox, `robust_sav`, a full ProtoMotions installation, nor additional human mesh assets are needed to evaluate the four frozen human policies.

## Remaining runtime gaps

1. **Pinned mjlab source is outside Git.** `requirements-physics.lock` references `.cache/deps/mjlab-8ee51fbc/`, which is ignored. Preserve its **62,168,556 bytes** of source, or recover revision `8ee51fbcf806a7419189f706d9e394cbeb7790fa` from an appropriate Git repository. The local sibling repository contains that object and records upstream `https://github.com/mujocolab/mjlab.git`; this audit did not test a fresh remote fetch. Do not substitute the latest mjlab release.
2. **The setup script assumes the current layout.** `scripts/setup_physics.sh` exports that commit from sibling `../mjlab` and selects `/usr/bin/python3`. Recreate that layout, or use the cache-only installation instructions in [report 27](27-server-migration.md) with an explicit Python 3.12 interpreter. Use `requirements-physics.lock`; `requirements-physics-installed.lock` is an observed installation record containing old absolute paths.
3. **GPU launch is site-specific.** The current wrapper selects `xeon-g6-volta` and `gpu:volta:1`. The trainer also requires `SLURM_JOB_ID` and CUDA. Adapt Slurm partition/GPU resources on another cluster; a non-Slurm server needs a small runner/launch-check change. Rebuild environments and compilation caches instead of copying the existing virtual environments. Package downloads and compatible system GPU/rendering drivers are still required.
4. **Historical defaults are not the portable entry point.** The wrapper defaults to ignored `outputs/scalar-body/` and `outputs/scalar-reference-v3/` paths. The [bundle README](../outputs/four-punch-milestone/README.md) overrides these with committed model, metadata, reference, checkpoint, and evaluation-bank paths. Follow that command, changing only the destination's allocation settings and output directory. Historical paths inside JSON files are provenance; do not recreate every old path for evaluation.

The new server must rerun the GPU preflight and all four frozen 64-trial evaluations, checking the existing physical/precision gates and clocks. This audit verifies repository coverage; it does not claim that an untested destination already reproduces the 64/64 pass counts.

## Additional files for training-history or reference reconstruction

The selected policies inherited earlier learned weights. These immediate predecessors are referenced in committed `arguments.json`/`initialization.json` but their bytes are **not tracked**:

| Final family | Missing immediate initialization checkpoint |
|---|---|
| Jab | `outputs/punch-5ms-5931681/checkpoint_00100.pt` |
| Cross | `outputs/punch-5ms-5931781/checkpoint_00300.pt` |
| Left hook | `outputs/punch-5ms-5931707/checkpoint_00600.pt` |
| Right hook | `outputs/punch-5ms-5931833/checkpoint_00300.pt` |

Preserve the parent chains as well if reproducing training from its beginning; copying only these four files covers only the last transfer stage. [Report 28](28-wbc-mjlab-code-audit.md) documents the lineage and update counts. Existing checkpoints save weights, optimizer state, update count, arguments and hashes, but not random-generator or simulator state; resuming starts fresh simulated trajectories.

For new retargeting, preserve the four selected raw DuoBox NPY files listed in [report 27](27-server-migration.md), totaling **20,404,260 bytes**. The scalar-reference builder also reads conditioned outputs from historical runs `physical-5890952`, `physical-5890924`, and `physical-5891006`; those are not tracked. Keep those inputs and their manifests, or archive the historical output tree if the full development record matters. Frozen final references already in Git avoid these inputs for evaluation/fine-tuning.

The workspace's `../main_mjlab.tex`, PDFs, and raw dataset are outside this repository. Copy the research-log project separately if desired. Early report links into ignored historical runs may not resolve from a clone; the standalone report preserves its embedded media, and the accepted release preserves its full numerical evidence.

## Recommended migration

For continuing from the accepted four policies: transfer/clone the audited commit, preserve the pinned mjlab source, rebuild the two environments, adapt the GPU runner, and validate the bundled banks. **No additional checkpoint upload is needed for the four final policies.**

For preserving the full experiment history: additionally archive raw DuoBox inputs and historical output runs. There is no need to put virtual environments, compiled kernels, or the complete transient cache into Git. A versioned source archive/submodule for pinned mjlab and a portable bootstrap would close the main clean-clone setup gap in a future implementation change.

Locally, `main` was one commit ahead of the cached `origin/main` reference (`25d0144`). The cached remote reference already includes the policy bundle, but not the latest documentation commit. No live remote check or push was made; transfer the desired commit explicitly or confirm the remote state before cloning elsewhere.
