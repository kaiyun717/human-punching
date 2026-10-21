#!/usr/bin/env python3
"""Review stored Phase2 evidence or publish the unchanged qualified domain."""
import argparse
import json
from pathlib import Path

from human_punching.phase2_review import review_panel,review_family,publish_catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("panel","family","catalog"))
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result={"panel":review_panel,"family":review_family,"catalog":publish_catalog}[args.mode](args.plan,args.output)
    print(json.dumps(result,allow_nan=False,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
