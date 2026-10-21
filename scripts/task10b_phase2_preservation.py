#!/usr/bin/env python3
"""Verify prior sources and accepted/failed artifacts without rewriting receipts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from human_punching.wbc.provenance import writable_path,write_json,sha256_file
PREP=ROOT/'outputs/task10b/p2prep_20261021_01'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=writable_path(args.output);out.relative_to(ROOT/'outputs/task10b')
    out.mkdir(parents=True,exist_ok=False)
    sources=json.loads((PREP/'initial_sources.json').read_text())
    prior=json.loads((PREP/'prior_phase_artifacts_before.json').read_text())
    baseline=json.loads((PREP/'phase0_baseline_verification.json').read_text())
    sets={'initial_source_files':sources['files'],'prior_phase_artifacts':prior['files'],
          'inherited_baseline':{r['path']:{k:r[k] for k in ('sha256','bytes')} for r in baseline['files']}}
    results={};seen={}
    for name,members in sets.items():
        failures=[];total=0
        for path,expected in members.items():
            p=Path(path)
            if not p.is_absolute():p=ROOT/p
            try:
                resolved=p.resolve(strict=True)
                if str(resolved) not in seen:
                    h=hashlib.sha256()
                    with resolved.open('rb') as stream:
                        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                    seen[str(resolved)]={'sha256':h.hexdigest(),'bytes':resolved.stat().st_size}
                actual=seen[str(resolved)]
                if any(actual[k]!=expected[k] for k in ('sha256','bytes')):
                    failures.append(dict(path=str(p),expected=expected,actual=actual))
                total+=actual['bytes']
            except Exception as exc:failures.append(dict(path=str(p),error=repr(exc)))
        results[name]=dict(passed=not failures,checked=len(members),bytes=total,failures=failures)
    old_roots={Path(p).parts[2] for p in prior['files'] if len(Path(p).parts)>3}
    added=[]
    for name in sorted(old_roots):
        for p in (ROOT/'outputs/task10b'/name).rglob('*'):
            if p.is_file() and not p.is_symlink() and str(p.relative_to(ROOT)) not in prior['files']:
                added.append(str(p.relative_to(ROOT)))
    phases=Counter(Path(p).parts[2][:2] for p in prior['files'])
    result=dict(schema='Task10BPhase2PreservationVerification/v1',utc=datetime.now(timezone.utc).isoformat(),
        passed=all(r['passed'] for r in results.values()) and not added,sets=results,
        added_files_inside_prior_phase_packets=added,
        prior_manifest_sha256=sha256_file(PREP/'prior_phase_artifacts_before.json'),
        prior_manifest_actual_prefix_counts=dict(phases),
        prior_manifest_label_correction='The inherited schema/scope text mentions Phase0 only; the immutable file map actually includes both p0 and p1 artifacts. This addendum clarifies the label without rewriting it.',
        old_receipts_rewritten=False,source_sha256=sha256_file(__file__))
    write_json(out/'preservation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sets'},indent=2))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
