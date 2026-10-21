#!/usr/bin/env python3
"""Generate a fresh Task10B Phase2 D0 timing library in an allocated session."""
import argparse
import json
import os
from pathlib import Path

from human_punching.phase2_reference import generate_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predicate", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full reference generation requires the root-authorized allocated session")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The declared Phase2 generation allocation must have its counted GPU available")
    result = generate_catalog(args.bundle, args.release, args.output, args.predicate)
    print(json.dumps({"cells": len(result["cells"]), "manifest": str(args.output/"manifest.json")}))


if __name__ == "__main__":
    main()
