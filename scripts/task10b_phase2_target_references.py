#!/usr/bin/env python3
"""Execute one immutable, bounded snapshot-target reference plan."""
import argparse
import json
import os
from pathlib import Path

from human_punching.phase2_target_catalog import generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,required=True)
    options=parser.parse_args()
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use a counted, explicitly bounded Phase2 Slurm GPU allocation")
    result=generate(options.plan,options.output,wall_seconds=options.wall_seconds)
    print(json.dumps({k:result[k] for k in ("complete","planned_cells","executed_cells","exception")}))
    return 0 if result["complete"] and not result["exception"] else 1


if __name__=="__main__":raise SystemExit(main())
