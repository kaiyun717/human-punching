#!/usr/bin/env python3
"""Create or verify a metadata-only view of a completed family-probe-v2 .210 catalog."""
import argparse
import json

from human_punching.phase2_family_projection_v2 import project, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("project")
    build.add_argument("--catalog", required=True)
    build.add_argument("--generation-execution", required=True)
    build.add_argument("--alias-probe", required=True)
    build.add_argument("--output", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--catalog", required=True)
    args = parser.parse_args()
    result = (project(args.catalog, args.generation_execution, args.alias_probe, args.output)
              if args.action == "project" else validate(args.catalog))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
