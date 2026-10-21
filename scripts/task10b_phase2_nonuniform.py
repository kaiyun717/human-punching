#!/usr/bin/env python3
"""Generate the six immutable contact-anchored nonuniform comparison references."""
import argparse
import json
from pathlib import Path
from human_punching.phase2_nonuniform import generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args()
    result=generate(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps({k:result[k] for k in ('complete','planned_new_cells','executed_new_cells','exception')}))
    return 0 if result['complete'] and result['exception'] is None else 1


if __name__=='__main__':raise SystemExit(main())
