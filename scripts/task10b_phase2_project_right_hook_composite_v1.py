#!/usr/bin/env python3
"""Project the exact audited RH54+10 cache, retaining both counted histories."""
import argparse
import json

from human_punching.phase2_right_hook_composite_projection_v1 import project, validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    build=sub.add_parser("project")
    for name in ("catalog","generation-execution","alias-probe","output","saved-audit"):
        build.add_argument("--"+name,required=True)
    sub.add_parser("verify").add_argument("--catalog",required=True)
    args=parser.parse_args()
    result=(project(args.catalog,args.generation_execution,args.alias_probe,args.output,
                    saved_audit_path=args.saved_audit) if args.action=="project" else validate(args.catalog))
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
