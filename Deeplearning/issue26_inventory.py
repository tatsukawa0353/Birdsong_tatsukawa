"""Create the read-only starting snapshot needed by Issue26 prepare."""
import argparse
from pathlib import Path
import subprocess

from .issue26_data import ROOT,OUT
from .issue19_protection import workspace_inventory
from .issue15_analysis import write_json
from .issue16 import digest


def inventory(destination):
    destination=Path(destination)
    if destination.exists() or OUT.exists():raise FileExistsError('Starting snapshot and study output must both be new')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True)
    paths={ROOT/p for p in git('ls-files','-z').split('\0') if p and p not in ('README.md','.gitignore')}
    paths.update(p for name in ('results','reports') for p in (ROOT/name).rglob('*') if p.is_file())
    records=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths) if p.is_file() and not p.is_symlink()]
    write_json(destination,dict(workspace=workspace_inventory(),git_status=git('status','--porcelain=v1','--untracked-files=all'),
        git_head=git('rev-parse','HEAD').strip(),unpushed=git('log','--oneline','@{upstream}..HEAD'),files=records))
    print(f'Protected {len(records)} existing files; no sound generation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,default=Path('/tmp/birdsong-issue26-before.json'));a=p.parse_args();inventory(a.destination)
