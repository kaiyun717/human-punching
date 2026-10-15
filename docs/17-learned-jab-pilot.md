# One-jab learned imitation pilot

This is a small, motion-conditioned residual-control experiment. It uses the
constrained SOMA-derived scalar-joint body and one complete retargeted jab cycle
in the pinned mjlab/MuJoCo Warp environment. It is not yet the Task 09 adversarial
planner, a target-generalizing punching policy, or a biomechanical validation.

## Controller and learning contract

`src/human_punching/jab_learning.py` implements a Gaussian actor and value network,
generalized advantage estimation, and clipped PPO. Its physics state stays on the
allocated GPU through `mjlab.sim.Simulation`. There are no CPU physics steps in
training. Native MuJoCo is used once during initialization to evaluate reference
FK and an offline support-torque estimate.

The action is a bounded joint-angle residual around the committed reference.
Joint position targets are clamped to the body's anatomical ranges, and final
motor commands are clamped to each joint's torque range. All motors have unit
gear and actuate scalar hinges. The free root has no motor. Neither external
Cartesian forces nor generalized applied forces are used. Contact is simulated
through MuJoCo Warp.

The optional feedforward term estimates gravity/velocity compensation with
assumed double-foot support, using reference-pose contact Jacobians. The estimated
support forces are **never applied to the simulator**: they only produce bounded
joint motor torques. `--no-feedforward` disables that term. This calculation is an
engineering initializer, not a physically guaranteed whole-body controller.

An optional `--balance-feedback` extension uses reference contact Jacobians and
root mass blocks to turn root position/orientation/velocity error into additional
joint torques. Its matrix has only actuated-joint output rows. Root acceleration
requests are clipped to ±3, and final motor commands retain the same torque
limits. It remains an approximate double-support controller; actual contact
forces are neither overwritten nor imposed. This option is off in the first
PPO experiment, and must be assessed as a distinct controller configuration.

Observations contain root pose/velocity, joint pose/velocity, reference errors,
elapsed phase, the previous action, samples 0.10 and 0.25 seconds into the
reference, and both glove position errors. Reward separately measures joint,
root, body, glove-position and glove-orientation tracking. A late-cycle velocity
penalty encourages quiet return to guard. Actions are updated every 20 ms;
bounded PD torques are recomputed every 2 ms.

Training resets sample either guard or a reference phase, with small state
perturbations. Falling terminates a training episode. This improves sample
coverage but does not itself prove an uninterrupted full cycle. Evaluation
therefore starts from guard, performs exactly one initial state reset per world,
and runs the entire cycle plus final hold without resetting after a fall. Its
first world starts at the exact guard and others use reproducible perturbations.
Every evaluation physics step records state, torque, free-root actuator force,
applied forces, and actual foot-floor contact support.

The elapsed phase is measured from the initial guard reset. It is **not** the
paper's outward-launch time-to-contact definition. The fixed head center is used
for the existing 0.14 m contact predicate; 0.16 m clearance and return-to-guard
acceptance are checked independently by `jab_evaluation.py`.

## Reproducing a run

Run from `human-punching` after generating the scalar body and validated reference:

```bash
sbatch scripts/train_jab_volta.sbatch \
  --reference outputs/scalar-reference-v3/jab.npz \
  --num-envs 128 --updates 200 --horizon 64 --save-every 25
```

The script uses this server's available `xeon-g6-volta` V100 partition, not an
unavailable L40S partition. It respects Slurm's assigned CUDA devices and places
caches under the project. It defaults to `.venv-physics`, the previously validated
Torch 2.8 / CUDA 12.8 and MuJoCo Warp 3.11 stack. See `05-gpu-server-guide.md` for
the server details.

Each output directory contains a copied model directory, reference archive,
body metadata, learner source, command arguments, and controller schema. The
checkpoint also includes model/reference hashes, network dimensions, gains,
joint order, action scale, observation layout, and timing. `training.jsonl`
records PPO progress; baseline and learned rollouts are separate artifacts.

Reproduce the accepted checkpoint's fixed-bank evaluation:

```bash
sbatch scripts/train_jab_volta.sbatch --mode evaluate \
  --model outputs/jab-learning-5895253/model/human.xml \
  --reference outputs/jab-learning-5895253/reference.npz \
  --body-metadata outputs/jab-learning-5895253/body_metadata.json \
  --checkpoint outputs/jab-learning-5895253/checkpoint.pt \
  --evaluation-bank outputs/scalar-evaluation-bank/guard_trials.npz \
  --recorded-worlds 32 --attack-window .5 3.3888888888888893
```

Omit `--checkpoint` in evaluation mode for the matched reference-PD baseline.
The checkpoint restores its timing/scales and validates its effective gains,
joint order, action/observation schema, body and reference. Changed controller
settings require a separate experiment. A fresh baseline can probe 5 ms with
`--timestep .005`; checkpoint evaluation preserves its trained 2 ms setting.
Compatibility with G1's accepted 5 ms physics requires measured validation.

## Verified primary result

The original PPO-200 checkpoint **passes all declared pilot gates in 32/32
independently audited fixed-bank trials**. Its matched zero-residual controller
passes 0/32. Every learned trial stays upright, respects the tested torque/joint
limits, reaches the head-contact predicate during the declared attack window,
keeps sole drift within the pilot limit, and returns to a supported quiet guard.
Every recorded free-root actuator force and applied generalized/Cartesian force
is zero. Evaluation contains one initial reset and no subsequent state overwrite.

The nominal learned trial reaches 0.11399 m glove/head center distance, with first
contact at 1.75 seconds from the initial guard clock. Maximum sole drift is
4.95 mm and final guard dwell is 1.804 seconds. Final glove errors measured in the
Chest frame are 6.35/4.80 mm and 2.14/1.41 degrees, with relative speeds
0.0042/0.0121 m/s. This is actual learned feedback: the same reference and PD
controller with its network residual removed falls in all 32 matched trials.

The selected checkpoint is `outputs/jab-learning-5895253/checkpoint.pt`.
Paired full-state evaluations are `outputs/jab-learning-5895279` (reference PD)
and `outputs/jab-learning-5895280` (learned policy). The independent bank is
`outputs/scalar-evaluation-bank/guard_trials.npz`, SHA256
`e8ae40abcdbce50b42cea94a5f4b5e0e6535a4bec03fe61cb11cdbda73786c8a`.
It contains one exact initial guard and 31 fixed small perturbations: hinge
standard deviation 0.005 rad with anatomical clamping, and velocity standard
deviation 0.02 in the corresponding generalized-coordinate units. This tests
local robustness around one body, guard, target and jab; it does not establish
new targets, timings, morphologies, other punch families, or mixed sequences.

The declared attack window is 0.5–3.3888889 seconds of reference time. It excludes
the explicit appended return blend but is not yet a precisely annotated
outward/follow-through window. Reported contact times remain guard-clock times,
not the paper's launch-relative TTC. There is no simulated forceful head impact:
the accepted analytic center-distance contract supplies the contact event.

The accepted physics step is **2 ms** with a 20 ms actor period. Native 5 ms
tests failed; integration into G1's accepted 5 ms physics has not been achieved.
First Warp initialization included several minutes of kernel compilation and
Lustre I/O waits. The reported 444-second training duration excludes this cold
startup and the baseline/final evaluations; it is not end-to-end wall time.

Reference v3 uses the previously conditioned, head-directed DuoBox jab at
**60% of its source playback speed**, with inserted quiet guard and return blend.
The learner's `--speed 1` means normal playback of that already slowed reference.
This milestone does not establish full-speed human punching. The reference lasts
4.989 seconds and evaluation adds one second of final guard hold.

## Experiment history

- Job 5895225: eight-world synthetic-standing API smoke test; one PPO update
  completed on actual GPU physics, with a saved checkpoint and continuous
  evaluation. Both baseline and trained one-second standing trials had zero
  falls. This tests software plumbing and is not evidence of learned punching.
- Job 5895242: cancelled during initialization, before training, when an
  additional audit found scalar-joint discontinuities in reference v2. Small
  Cartesian tracking error alone was insufficient to accept that reference.
- Job 5895253: real-jab experiment on smoothed reference v3; 128 worlds,
  200 PPO updates, 2 ms physics, checkpoint every 25 updates. Completed
  1,638,400 transitions in 444 seconds of training after initialization and
  baseline evaluation. The 128-world guard-start learned evaluation had zero
  falls and 99.22% any-cycle contact, compared with 100% falls in its baseline.
  Mean continuous-evaluation reward increased from -0.474 to 0.977. These initial
  states are small perturbations of one morphology/guard/reference, not a broad
  generalization benchmark. Strict timing and guard gates are checked separately.
- Job 5895265: checkpoint-50 guard-start evaluation; all 16 worlds still fell.
  This intermediate failure is retained; improved training reward alone did not
  establish success.
- Job 5895277: separate 256-world balance-feedback PPO comparison. Intentionally
  stopped after checkpoint 100 was successfully loaded, at exactly 1,638,400
  training transitions, matching the primary experiment's experience budget.
  Its fixed-bank evaluation is job 5895286. This is an exploratory controller
  comparison, not a repeated-seed statistical study. Its independent audit also
  passes 32/32 trials on all frozen gates, with mean continuous-evaluation reward
  0.9883. The primary policy remains the original no-balance-feedback checkpoint,
  which demonstrates that the learned residual acquired the needed balance.
- Jobs 5895279/5895280: paired zero-residual/learned-200 evaluations on the frozen
  32-trial guard bank, recording every world and a declared attack window.

An independent native-controller diagnostic found that 2 ms inverse-statics
feedforward plus reference PD initially stands, but develops slow root drift and
falls after about 3.7 seconds during a long quiet trial. A 5 ms variant produces
rapid joint oscillations and fails earlier. The initial PPO run therefore has a
real balance problem to learn, rather than merely reproducing a successful
classical whole-cycle controller. Results must be judged on guard-start physical
evaluation, not training rewards alone.

The paired evaluations were independently checked against the immutable initial
states, all 32 world IDs, model/reference hashes and bank hash before acceptance.

The first v3 guard-start reference-PD baseline has 100% falls over its six-second
evaluation across 128 initial states. Some trajectories pass the 0.14 m glove
predicate, but that does not make them valid punch-and-return demonstrations.
The nominal rollout records exactly zero free-root actuator force and exactly
zero generalized/Cartesian applied forces. Training rewards use episode resets;
their values cannot be interpreted as whole-cycle success rates.
