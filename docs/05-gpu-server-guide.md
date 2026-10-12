# Using GPUs on this server

Date: 2026-10-10. Scope: the isolated `human-punching` project.

The GPU partition on the connected server is **`xeon-g6-volta`**. Request a GPU with **`--gres=gpu:volta:1`**, following the supplied [`example_slurm.sh`](../../example_slurm.sh). The earlier `mit_normal_gpu` / L40S script targets a different cluster and does not apply here. A single allocated GPU is sufficient to render all four punch previews. The complete four-punch job was tested successfully on an allocated NVIDIA Tesla V100: 23 seconds of batch time, with 15.67 seconds inside the preview application.

## Resources verified on this server

These values were read from `sinfo`, `scontrol show partition xeon-g6-volta`, and `sacctmgr` during this task. They describe scheduler configuration, not GPUs that are immediately free.

| Setting | Observed value | Meaning for this project |
|---|---|---|
| Slurm cluster | `txe1` | The connected cluster |
| GPU partition | `xeon-g6-volta` | Use this partition for batch jobs |
| GPU resource type | `gpu:volta` | Preserve this type in submission and job-step requests |
| GPUs per compute node | 2 | An eight-GPU request cannot fit on one node |
| Partition inventory | 172 nodes / 344 Volta GPUs | Shared cluster inventory; not this user's allowance |
| User/account association | `kyun` / `default_group` | The supplied example uses the default account |
| Association limit | `GrpTRES=node=4` | Consistent with up to four GPU nodes for this association |
| Allowed QoS | `high,normal` | The script relies on the site's default selection |
| Partition wall-time limit | `4-04:00:00` | Preview jobs should request much less |
| Example environment module | `anaconda/Python-ML-2025a` | Python 3.10.14; not needed by the verified project environment |

The user's stated **eight GPUs** is consistent with the observed four-node allowance and two GPUs per node. This is an interpretation of the current limits, not a reservation or a guarantee that eight GPUs can start immediately. Other jobs using the association consume its allowance. The `normal` QoS query showed no additional explicit GPU limit.

To inspect current resources and limits yourself:

```bash
sinfo --format='%P|%a|%l|%D|%G'
scontrol show partition xeon-g6-volta
sacctmgr -n -P show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmitJobs
sacctmgr -n -P show qos \
  format=Name,MaxTRESPerUser,MaxTRESPerJob,GrpTRES,MaxJobsPerUser,MaxSubmitJobsPerUser
```

`--gres=gpu:volta:1` requests one GPU **per allocated node**. Slurm sets `CUDA_VISIBLE_DEVICES` for the allocation; GPU numbering inside a constrained job can differ from host numbering. Preserve the scheduler's assignment. [Slurm GRES documentation](https://slurm.schedmd.com/gres.html)

## Render all four punches on one GPU

The existing `.venv` is ready to use. For a fresh checkout, first follow the [README setup](../README.md#setup) inside this project. Do not use a training environment from a sibling repository.

```bash
cd /home/gridsan/kyun/Azizan_Lab/HumanBox/human-punching
job_id="$(bash scripts/submit_gpu.sh)"
echo "$job_id"
squeue -j "$job_id"
```

[`submit_gpu.sh`](../scripts/submit_gpu.sh) creates the log directory and submits [`render_volta.sbatch`](../scripts/render_volta.sbatch) from the correct project directory. The worker requests one node, one task, **one Volta GPU**, **4 CPUs**, **12 GiB RAM**, and **10 minutes**. It uses the existing `.venv`, establishes EGL rendering, then generates jab, right hook, left hook, and cross in one process. It needs neither a display server nor an Anaconda module.

Preview options can be passed after the wrapper name:

```bash
bash scripts/submit_gpu.sh --height 1.80
bash scripts/submit_gpu.sh --punches jab
```

The worker fixes the backend to `egl` and chooses a unique output directory for each job. Do not supply `--no-render` to this rendering worker: its completion check requires rendered media.

For job `JOB_ID`, the output is:

```text
outputs/slurm/human-punch-volta-JOB_ID.log
outputs/gpu-JOB_ID/
  nvidia-smi.csv       allocated hardware information
  gpu_runtime.json    CUDA/EGL mapping and actual rendering evidence
  egl_device.env      verified EGL device selection for the preview process
  complete.json       written only after successful rendering
  preview/
    manifest.json
    overview.png
    model/
    jab/jab.mp4
    right_hook/right_hook.mp4
    left_hook/left_hook.mp4
    cross/cross.mp4
```

Trajectories, still images, and per-punch manifests accompany the videos. Job-specific CUDA, shader, temporary and XDG caches stay under `.cache/gpu/JOB_ID`. The accepted CPU previews at `outputs/preview` remain available.

## Monitor or stop a job

Use the ID printed by `submit_gpu.sh`:

```bash
squeue -j "$job_id" -o '%.18i %.9P %.25j %.8T %.10M %.6D %R'
sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
cat "outputs/slurm/human-punch-volta-${job_id}.log"
cat "outputs/gpu-${job_id}/complete.json"
```

A finished successful job has `State=COMPLETED`, `ExitCode=0:0`, and a `complete.json` file. A missing job in `squeue` can simply mean it finished; inspect `sacct` and the artifacts. Logs appear when the worker starts. To cancel a job you no longer need:

```bash
scancel "$job_id"
```

A queued job may wait for resources, priority, or the association's node allowance. Inspect the final `squeue` column and `scontrol show job "$job_id"`. Do not change GPU visibility or request unallocated devices to address queue delays.

## Optional: one punch per array task

The default one-GPU job is enough for the current preview. To run the four punches as independently scheduled jobs:

```bash
array_id="$(bash scripts/submit_gpu.sh --array)"
squeue -r -j "$array_id"
sacct -j "$array_id" --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

This submits `--array=0-3%4`: at most four tasks can run at once, each requesting one GPU. IDs map to `0=jab`, `1=right_hook`, `2=left_hook`, and `3=cross`. Logs use `human-punch-volta-ARRAY_ID_TASK_ID.log`; outputs use `outputs/gpu-ARRAY_ID_TASK_ID/preview`. The worker adds the selected punch argument, so each task renders exactly one punch. There is no combined `overview.png` because each task produces only one clip.

The array path is for independent work. It does not distribute one render across multiple GPUs. The four current clips do not need all eight GPUs. Future eight-GPU workloads must fit the site's two-GPU-per-node topology and node allowance, and implement separate processes or distributed computation explicitly.

## Verify the allocated GPU is actually rendering

The worker runs [`gpu_preflight.py`](../scripts/gpu_preflight.py) before generating videos. The preflight:

1. Requires a Slurm GPU allocation and preserves `CUDA_VISIBLE_DEVICES`.
2. Uses the CUDA driver API to identify logical device 0, including its physical PCI address and UUID.
3. Matches that physical address to an EGL device; CUDA logical indices and EGL indices are distinct.
4. Sets `MUJOCO_EGL_DEVICE_ID`, creates a real MuJoCo render context, and confirms that its EGL display uses the selected device.
5. Checks the actual OpenGL vendor/renderer and rejects software rendering; renders a nonconstant 160 × 120 test image.
6. Records the evidence in `gpu_runtime.json` and exports the verified selection through `egl_device.env`.

A successful `nvidia-smi` command alone is not evidence that MuJoCo used the GPU. Inspect `gpu_runtime.json`: `status` must be `passed`, `egl_display_matches_selected_device` must be `true`, and `opengl` must identify NVIDIA hardware. A mapping or context failure stops the worker before preview rendering and records the error in the JSON report.

## Troubleshooting

| Symptom | Action |
|---|---|
| `Invalid partition` for `mit_normal_gpu` | Use `scripts/submit_gpu.sh` and the current `xeon-g6-volta` worker. The L40S script belongs to another cluster. |
| Missing project `.venv` | Complete the README setup in `human-punching`, then resubmit. |
| `CUDA_VISIBLE_DEVICES` missing / no CUDA GPU | Use `sbatch` through the wrapper. Running the worker directly on the login host does not allocate a GPU. |
| EGL device mismatch or initialization error | Read `gpu_runtime.json` and the job log; preserve scheduler visibility and use the supplied preflight. Do not guess an EGL index. |
| `llvmpipe`, `softpipe`, or another software renderer | Treat the job as a failed GPU validation. The preflight rejects this case. |
| `TIMEOUT` | Inspect the log for the stage that stalled. For a deliberately longer workload, adjust the worker's `--time` request within the partition limit. |
| `OUT_OF_MEMORY` | Inspect `sacct` and the failing stage before increasing the worker's `--mem` request; host RAM and GPU memory are separate resources. |
| `complete.json` is missing | Check `sacct`, the Slurm log and `gpu_runtime.json`. Completion is recorded only after the preview manifest contains rendered media. |

## What GPU support means for this milestone

The current preview renders the articulated light-blue mannequin with native **MuJoCo 3.11.0**. Its headless GPU backend is **EGL**, selected before importing MuJoCo. There is no desktop display or interactive viewer in a batch job. MuJoCo's Python package provides explicit OpenGL contexts for offscreen rendering. [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html#rendering)

Retargeting, `mj_forward`, text overlays, and video encoding still include CPU work. This milestone replays recorded poses; it does not train a policy or test mjlab/MuJoCo-Warp GPU physics. Successfully rendering on a Volta GPU does not establish that the future training stack, its CUDA wheels, or the Warp camera pipeline support this hardware.

More allocated GPUs do not make the current single-process preview use them automatically. Run the four previews on one GPU unless there is enough independent work to justify parallel jobs. For future parameter sweeps, a Slurm array can limit concurrent tasks with a `%` suffix; each task still needs its own GPU allocation and separate output directory. [Slurm job-array documentation](https://slurm.schedmd.com/job_array.html)

## Environment and project boundaries

All project environments, logs, caches, models and videos belong under `human-punching`. The sibling `robust_sav`, `mjlab` and DuoBox inputs remain read-only. Slurm scripts must resolve the project directory from their submission configuration: Slurm may execute a spool copy of the script, so its runtime `BASH_SOURCE` is not a reliable way to find this repository.

The existing project `.venv` works on the compute node unchanged: Python 3.12.3, MuJoCo 3.11.0 and NumPy 2.5.3 were imported successfully. There is no need to load `anaconda/Python-ML-2025a` for this preview. The example module provides Python 3.10.14; switching to it would use a different interpreter from the tested environment. `requirements-preview.lock` records the Python 3.12 environment. The worker enters the project directory and uses `.venv/bin/python` explicitly, preserving the isolated dependencies.

## Verified GPU probe

Initial probe [Slurm job **5884571**](../outputs/slurm/hp-volta-probe-5884571.log) completed with exit code **0** in **6 seconds** on node `d-12-12-1`. It used the existing project environment and reported:

| Check | Result |
|---|---|
| Allocated GPU | NVIDIA Tesla V100-PCIE-32GB |
| GPU memory | 32,768 MiB |
| Driver | 580.159.04 |
| Python | 3.12.3 |
| MuJoCo | 3.11.0 |
| NumPy | 2.5.3 |
| EGL/OpenGL vendor | NVIDIA Corporation |
| EGL/OpenGL renderer | Tesla V100 |
| Render test | Valid 64 × 64 image |

The scheduler supplied a **GPU UUID** in `CUDA_VISIBLE_DEVICES`, rather than an integer, and `SLURM_JOB_GPUS=0`. `nvidia-smi` exposed only the allocated GPU at index 0. This is why the rendering setup must resolve the allocation to the corresponding EGL device instead of copying `CUDA_VISIBLE_DEVICES` directly into an integer-only EGL setting. Do not hard-code this job's UUID or assume the next allocation uses the same physical device.

## Verified full four-punch run

Slurm job **5884573** completed with state **`COMPLETED`**, exit code **`0:0`**, and **23 seconds** of batch wall time on `d-12-12-1`. The preview application generated all four clips in **15.67 seconds**. This includes retargeting, GPU rendering and video creation; it is not a controlled GPU-versus-CPU benchmark.

The production preflight passed on the allocated Tesla V100-PCIE-32GB, compute capability 7.0. Its CUDA PCI address matched the EGL device through `EGL_NV_device_cuda`, and the active EGL display matched that selected device. The rendered preflight image was 160 × 120 RGB with finite, nonconstant pixels. The OpenGL renderer was `Tesla V100-PCIE-32GB/PCIe/SSE2`, vendor `NVIDIA Corporation`.

Evidence: [Slurm log](../outputs/slurm/human-punch-volta-5884573.log), [GPU preflight report](../outputs/gpu-5884573/gpu_runtime.json), [completion record](../outputs/gpu-5884573/complete.json), and [preview manifest](../outputs/gpu-5884573/preview/manifest.json).

Rendered outputs: [overview](../outputs/gpu-5884573/preview/overview.png), [jab](../outputs/gpu-5884573/preview/jab/jab.mp4), [right hook](../outputs/gpu-5884573/preview/right_hook/right_hook.mp4), [left hook](../outputs/gpu-5884573/preview/left_hook/left_hook.mp4), and [cross](../outputs/gpu-5884573/preview/cross/cross.mp4).

This validates the current mannequin and punch replay in native MuJoCo EGL on this server's GPU. It does not validate physical adversary training, multi-GPU training, or the mjlab/MuJoCo-Warp camera and simulation pipeline.

## Verified four-task array

Array **5884574** completed all four tasks with state **`COMPLETED`** and exit code **`0:0`**. The scheduler placed two one-GPU tasks on each of two nodes, using four GPUs in total. Every task passed the CUDA-to-EGL physical-device check and generated its selected punch video.

| Task | Punch | Node | GPU PCI address | EGL device index | Batch time | Runtime evidence |
|---|---|---|---|---:|---:|---|
| `5884574_0` | jab | `d-12-12-1` | `0000:86:00.0` | 0 | 7 s | [JSON](../outputs/gpu-5884574_0/gpu_runtime.json) |
| `5884574_1` | right hook | `d-12-12-1` | `0000:d8:00.0` | 1 | 7 s | [JSON](../outputs/gpu-5884574_1/gpu_runtime.json) |
| `5884574_2` | left hook | `d-12-13-1` | `0000:86:00.0` | 0 | 10 s | [JSON](../outputs/gpu-5884574_2/gpu_runtime.json) |
| `5884574_3` | cross | `d-12-13-1` | `0000:d8:00.0` | 1 | 10 s | [JSON](../outputs/gpu-5884574_3/gpu_runtime.json) |

All four processes selected their CUDA logical device 0, while two required **EGL device 1**. This confirms why assigning EGL index 0 unconditionally would be incorrect on this server. The JSON reports include the per-node GPU UUIDs and the active-display verification.

The [Slurm accounting record](../outputs/gpu-slurm-accounting.txt) contains allocation details, exit codes and elapsed times for the probe, single job and array tasks. This test verifies four GPUs across two nodes; an eight-GPU run was not needed or performed.

All **eight generated MP4s** across the single-job and array runs passed full decoding and metadata checks: H.264, 1280 × 720, 30 fps, with the expected frame counts. Their saved trajectories passed comparison against the accepted CPU references at a `1e-12` tolerance. The [combined media and trajectory validation report](../outputs/gpu-validation.json) records per-video checks and SHA256 hashes, alongside GPU UUIDs and PCI addresses.

## Standing-G1 targeting preview

The newer head-targeted preview uses the same tested Slurm allocation and EGL-device preflight:

```bash
bash scripts/submit_g1_gpu.sh
```

It runs `human_punching.target_cli` through `scripts/target_g1_volta.sbatch`, requesting one Volta GPU, 4 CPUs, 12 GiB RAM and 15 minutes. Logs are `outputs/slurm/human-g1-target-JOBID.log`; the completed scene, trajectories and four videos are under `outputs/g1-gpu-JOBID/preview`. This workflow includes CPU arm IK followed by native GPU rendering. It does not train a policy or run Warp physics. See the [targeting report](07-standing-g1-targeting.md).

## Physical simulation follow-up

The [physics runtime report](09-physics-runtime.md) now verifies the exact pinned training stack on these V100s, including actual MuJoCo-Warp physics, cameras, and a controlled jab. Use `scripts/setup_physics.sh`, `scripts/physics_probe_volta.sbatch`, and `scripts/warp_tracking_volta.sbatch` for that environment. Native physical videos use `scripts/physical_pilot_volta.sbatch`; their GPU allocation accelerates rendering while the native solver runs on the CPU. The Warp worker explicitly runs GPU dynamics.
