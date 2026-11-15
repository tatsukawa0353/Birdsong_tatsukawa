"""Redraw only the canonical, single-condition 4096-step figures."""
import argparse
import json
from pathlib import Path
from .issue17 import DEST
from .issue17_figures import figures
from .issue17_artifacts import refresh_checksums


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=DEST)
    args=parser.parse_args()
    config=json.loads((args.results/'config.json').read_text())
    if config['episodes']!=4096:
        parser.error('Use the 4096-step canonical results; retired pilot plots are not redrawn')
    figures(args.results)
    if (args.results/'checksums.json').exists():refresh_checksums(args.results)
    print('Redrew separate single-condition figures; no training or evaluation.')
