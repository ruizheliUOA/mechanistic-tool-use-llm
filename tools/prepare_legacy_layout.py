#!/usr/bin/env python3
"""Copy retained artifacts to a new historical-layout workspace; never run experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

def prepare(root, out):
    if out.exists():
        raise ValueError('Destination already exists; choose a new directory')
    root=root.resolve(); out=out.resolve()
    if out.is_relative_to(root):
        raise ValueError('Destination must be outside this artifact checkout')
    records=json.loads((root/'manifests/retained_artifacts.json').read_text())['files']
    plan={}
    for row in records:
        rel=Path(row['legacy_path']); source=root/row['path']
        if rel.is_absolute() or '..' in rel.parts or '.git' in rel.parts:
            raise ValueError('Unsafe relative path in retained-artifact manifest')
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['published_sha256']:
            raise ValueError('Changed artifact: '+row['path'])
        if rel in plan and plan[rel]!=data:
            raise ValueError('Conflicting historical-layout destination')
        plan[rel]=data
    out.mkdir(parents=True,exist_ok=False)
    for rel,data in plan.items():
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print(json.dumps({'files_copied':len(plan),'experiment_executed':False,
                      'missing_external_inputs_restored':False},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    try:prepare(a.root,a.out)
    except (OSError,ValueError,KeyError) as e:
        print('FAIL: '+str(e),file=sys.stderr);return 1
    return 0

if __name__=='__main__':sys.exit(main())
