#!/usr/bin/env python3
"""Run an explicitly selected, bounded original-PPO D0 timing pilot."""
import argparse
import json
from pathlib import Path
from human_punching.phase2_timing_pilot import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    result=execute(options.plan,options.output)
    print(json.dumps({k:result[k] for k in ('status','exception','fresh_transitions','phase2_complete')}))
    return 0 if result['status']=='bounded_stop' else 1


if __name__=='__main__':raise SystemExit(main())
