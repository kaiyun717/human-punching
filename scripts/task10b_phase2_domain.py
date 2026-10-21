#!/usr/bin/env python3
"""Freeze a read-only G1 snapshot domain, without G1 inference or stepping."""
from pathlib import Path
import argparse
import json
from human_punching.phase2_domain import build_domain

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_domain(args.output, project_root=root,
        bank_root=root.parent/"robust_sav/constants/boxing/snapshots/v1_20260921",
        g1_asset_root=root/".cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1",
        nominal_plant_path=root.parent/"robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json"), indent=2))

if __name__ == "__main__":
    main()
