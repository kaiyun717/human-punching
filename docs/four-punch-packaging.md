# Packaging the four-punch physical policy release

The release script packages **four explicitly selected checkpoints and their independently accepted evaluations**. It does not select checkpoints automatically or run training. `--check-only` can preflight one or more ready families; publication always requires all four.

From `human-punching`, substitute actual immutable training checkpoint paths and evaluation-run directories:

```bash
.venv-physics/bin/python -B scripts/package_punch_policies.py \
  --checkpoint jab=outputs/JAB_TRAIN/checkpoint_00800.pt \
  --checkpoint cross=outputs/CROSS_TRAIN/checkpoint_00800.pt \
  --checkpoint left_hook=outputs/LEFT_HOOK_TRAIN/checkpoint_00800.pt \
  --checkpoint right_hook=outputs/RIGHT_HOOK_TRAIN/checkpoint_00800.pt \
  --evaluation jab=outputs/JAB_EVALUATION \
  --evaluation cross=outputs/CROSS_EVALUATION \
  --evaluation left_hook=outputs/LEFT_HOOK_EVALUATION \
  --evaluation right_hook=outputs/RIGHT_HOOK_EVALUATION \
  --check-only
```

The evaluation directory must contain `independent/summary.json`, all detailed trial reports and rollouts, the frozen source/runtime XML, reference, bank, controller schema, body metadata and evaluation source records. It is the evaluation run directory, not its `independent` subdirectory.

After validation succeeds, use the same command without `--check-only` to publish `outputs/four-punch-milestone`. `--output outputs/another-version` chooses a new release directory. Existing releases are never overwritten. All four families are validated before writing; files are copied to a temporary directory inside `outputs`, verified against their preflight hashes, and renamed to the final destination only after every copy succeeds.

## Required acceptance

- Every family has **at least 32 independently accepted trials, with every trial passing** with all declared physical gates present and passing, including active-hand attack contact, supported guard recovery, joint limits, bounded actual torque, unactuated root and absence of external forces.
- Source and runtime models, reference, initial-state bank, checkpoint and controller schema agree with the corresponding evidence. Runtime XML hashes must match exactly between training and evaluation; older incorrectly exported runtime models are rejected.
- Native physics is **5 ms**, actor updates are **20 ms**, and uniform references and runtime playback use **1.0× source speed** by default. Recorded per-step clocks and actor-update indices establish this; filenames and arguments alone are insufficient.
- Every bank trial has a unique matching rollout and matching initial state. The archive must identify a learned policy and the selected checkpoint.
- New evaluations must match the recorded checkpoint SHA256. Legacy records without that field are accepted only for explicitly numbered checkpoint filenames with matching paths; the missing historical hash is prominently recorded as a provenance limitation. Mutable `checkpoint.pt` paths without recorded byte hashes are rejected.

Nonuniformly retimed references require the explicit `--allow-conditioned-timing` flag. Their local speed ranges and timing descriptions are preserved, and their effective scalar source speed is reported as null. This does not relax the 5 ms/20 ms clocks or any physical gate. Runtime playback remains 1.0× of the modified archive.

The ordinary release level is **accepted physical pilots**. Three tighter precision targets are reported separately: full-cycle active-glove RMS ≤2 cm, attack-window active-glove RMS ≤2 cm, and absolute interpolated contact-timing error ≤20 ms. Add `--require-precision` to require all three in every trial. The default does not silently turn aspirational precision targets into physical acceptance gates or imply that they passed.

## Contents and scope

Each family contains its checkpoint, frozen body and runtime XML, reference, body metadata, controller schema, training arguments and source snapshots, available learning curves/logs, evaluation bank/manifest, all recorded rollouts and all independent trial/summary metrics. A top-level manifest records SHA256, size and original path for each copied artifact, selected updates, pass counts, clock contracts and limitations. ProtoMotions license/notice records and the packaging script itself are included.

Paths inside original evidence are deliberately retained as historical provenance; the manifest maps those origins to the copied artifacts. The release is four isolated reference-conditioned policies, not a continuous combination controller, requested-rate scheduler, moving-G1 task or validated adversarial game.

## Optional paired baselines and visual evidence

Add `--baseline FAMILY=RUN_DIR` to preserve an independently audited zero-residual baseline. Its frozen controller schema, source/runtime models, reference, initial-state bank, native clocks and recorded actuation contract must match the learned evaluation. All paired raw rollouts and detailed audits are included. The baseline may fail balance, tracking or contact gates; those failures remain visible rather than being treated as release failures. This is a comparison against the same reference controller with its residual disabled, not a claim that no stronger classical controller could work.

Add `--media-dir FAMILY=DIR` for videos, images or numerical plots. Repeat it to include several directories for a family. Each copied artifact receives an origin path and SHA256. Render directories must identify the selected learned checkpoint or the already-attached zero-residual baseline; video records must identify the matching rollout and successful full decoding. Plot-only directories are supporting material and are not silently labeled as verified render provenance.

Use `--support-dir DIR` for global reports such as the compact final validation manifest and independent runtime audit. These are frozen with hashes under `supporting/`; they supplement the per-trial acceptance evidence. The bundle also preserves the exact release-selection shell command and Slurm wrappers.

The accepted jab provides a real, read-only preflight example:

```bash
.venv-physics/bin/python -B scripts/package_punch_policies.py \
  --checkpoint jab=outputs/punch-5ms-5931736/checkpoint_00100.pt \
  --evaluation jab=outputs/punch-5ms-5931806 \
  --baseline jab=outputs/punch-5ms-5931807 \
  --media-dir jab=outputs/jab-render-5931851 \
  --media-dir jab=outputs/jab-render-5931873 \
  --media-dir jab=outputs/punch-5ms-5931806/independent/nominal-plots \
  --check-only
```

The initial jab preflight, job `5931896`, passed all 64 physical trials, all 18 gates, all 64 checkpoint byte hashes, exact runtime/source/reference provenance and all three precision targets. Job `5931917` additionally validated the paired 0/64 zero-residual baseline, learned video and numerical plots; job `5931923` also validated and attached the baseline video. No four-family release was published by these checks.

The validator has eighteen focused synthetic-evidence tests covering rejected physical gates, wrong clocks, runtime/checkpoint changes, duplicate trials, separate precision targets, atomic publication, subset preflight, visual provenance, paired-baseline mismatches and the actual nested nonuniform-timing schema. These tests validate packaging logic, not physical motion. Run:

```bash
.venv/bin/python -B -m pytest tests/test_package_punch_policies.py -q \
  --basetemp=outputs/package-tests-tmp
```

## Final selected milestone

The reproducible selection is [package_four_punch_milestone.sh](../scripts/package_four_punch_milestone.sh). It requires all precision targets and explicitly admits declared nonuniform hook timing:

| Family | Training run / checkpoint update | Fresh learned evaluation | Paired zero-residual evaluation | Original-source timing |
|---|---|---|---|---|
| Jab | `5931736 / 00100` | `5931806` | `5931807` | Uniform 1.0× |
| Cross | `5931870 / 00100` | `5931915` | `5931916` | Uniform 1.0× |
| Left hook | `5931834 / 00300` | `5931934` | `5931935` | Declared nonuniform phase warp |
| Right hook | `5931884 / 00100` | `5931918` | `5931919` | Declared nonuniform phase warp |

Every learned family passed all 18 physical gates and all three precision targets in all 64 fresh trials. Both hook references retain `source_speed=null` in the release summary; 1.0× archive playback does not mean uniform original-capture playback. Each paired baseline passed 0/64 physical trials under the identical controller, model, reference and initial-state bank with the learned residual disabled.

Run a read-only preflight from the project root:

```bash
sbatch scripts/package_four_punch_milestone.sbatch --check-only
```

Removing `--check-only` atomically publishes `outputs/four-punch-milestone` and refuses to replace an existing directory. The strict four-family preflight first passed in job `5931967`; job `5931984` additionally passed with all final attachments: eight completely decoded videos, eight numerical plot directories, selected-still visual reviews, the compact final validation manifest and the independent runtime/body/force-law audit. The latter artifacts are supporting evidence; no full manual viewing of every video frame is claimed.

The generated bundle README contains artifact-hash checks and an evaluation command that loads the bundled checkpoint, reference, body metadata and initial-state bank. It uses the existing project environment, preserves all source snapshots, and does not assume the original experiment paths for those evaluation inputs.

The release was published atomically by job `5931987` at [outputs/four-punch-milestone](../outputs/four-punch-milestone/README.md). Its [manifest](../outputs/four-punch-milestone/manifest.json) records every copied artifact's SHA256, size and origin. The release contains all 256 accepted learned rollouts and all 256 paired baseline rollouts, eight videos and eight plot sets. Additional post-publication smoke and hash checks write their evidence outside this immutable directory.

Post-publication [verification job `5932008`](../outputs/four-punch-release-verification-5932008.json) re-read all **1,264 artifacts** and confirmed every size and SHA256. Artifact contents total 493,177,707 bytes. The release manifest SHA256 is `16eebb942f71f8350bf3d886f8a083eac549a5bc3486fe9d5304b8cde29b0686`.

GPU smoke job `5931997` then loaded the published right-hook checkpoint, model, body metadata and reference through the bundled Slurm wrapper. It ran one nominal world at native 5 ms physics and 20 ms policy updates, writing to a separate output directory. The [independent smoke audit](../outputs/punch-5ms-5931997/independent_smoke.json), completed by job `5932013`, passed every physical gate and all three precision targets: 5.72 mm full-cycle active-glove RMS, 7.16 mm attack RMS, and 0.272 ms contact-timing error. This checks the bundled evaluation paths; it does not add another 64-trial generalization claim.

The first smoke-audit wrapper, job `5932009`, incorrectly included the descriptive `affects_pilot_pass=false` field in a generic truth assertion. The evaluator had already reported physical and precision success. The wrapper was corrected to assert only the explicit precision results and attack RMS; no policy, trajectory, model or acceptance criterion changed.
