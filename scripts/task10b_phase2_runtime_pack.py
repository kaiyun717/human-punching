#!/usr/bin/env python3
"""Build one explicitly planned Phase2 runtime reference pool."""
import argparse
import json
import os
from pathlib import Path
from human_punching.phase2_runtime_pack import pack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,required=True)
    args=parser.parse_args()
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use an authorized counted reference-generation allocation")
    result=pack(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="packed" else 1


if __name__=="__main__":raise SystemExit(main())
