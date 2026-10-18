# Four-punch 5 ms physical policy release

Each family has its own checkpoint, frozen source/runtime XML, reference, controller schema, training source and complete independent evaluation evidence (at least 32 trials per family). Read `manifest.json` for hashes, artifact origins, acceptance and precision-target counts. Physics uses 5 ms; policy updates use 20 ms; frozen references replay at 1.0×. Original-source timing and any explicitly allowed nonuniform timing are reported separately for each family.

These are four isolated physical pilots. They do not provide requested punch rates, continuous combinations, moving-target generalization or a complete adversarial opponent. Historical source paths inside copied evidence are retained for provenance.

## Check bundled artifact hashes

Run from the `human-punching` project root:

```bash
.venv/bin/python -B - <<'PY'
import hashlib, json
from pathlib import Path
bundle = Path('outputs/four-punch-milestone')
manifest = json.loads((bundle / 'manifest.json').read_text())
for relative, metadata in manifest['artifacts'].items():
    assert hashlib.sha256((bundle / relative).read_bytes()).hexdigest() == metadata['sha256'], relative
print('All bundled artifact hashes match')
PY
```

## Evaluate a bundled policy on the GPU

Run from the same project root with the existing project environments. This submits a new evaluation using the bundled model, reference, checkpoint and frozen 64-trial bank. The runner restores the checkpoint's controller settings and requires 5 ms physics / 20 ms actions. Change `punch_family` to `cross`, `left_hook` or `right_hook` for another family. Results go to a new job directory; the release remains unchanged.

```bash
punch_family=jab
bundle_root=outputs/four-punch-milestone
family_bundle="$bundle_root/$punch_family"
sbatch "$bundle_root/reproduction/train_punch_5ms_volta.sbatch" \
  --mode evaluate --model "$family_bundle/model/human.xml" \
  --body-metadata "$family_bundle/body_metadata.json" \
  --reference "$family_bundle/reference.npz" \
  --checkpoint "$family_bundle/checkpoint.pt" \
  --evaluation-bank "$family_bundle/evaluation/evaluation_bank.npz" \
  --recorded-worlds 64
```

The bundled Slurm wrapper uses the existing repository runner and installed dependencies. Exact training/evaluation source snapshots are preserved per family for provenance; this bundle does not vendor the complete software environment. Original selection/publication commands are preserved under `reproduction/`; their historical input paths are not needed by the bundled-policy evaluation command above.
