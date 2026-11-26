"""Read-only migration audit and saved-wave redraw; never run the simulator."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .result_paths import ROOT, MIGRATION, resolve


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit():
    before = read(MIGRATION / 'inventory_before.json')
    operations = {r['path']: r for r in csv.DictReader((MIGRATION / 'deletions.csv').open())}
    documents = {r['original_path']: r for r in read(MIGRATION / 'document_changes.json')}
    failures = []
    checked = deleted = aliases = 0
    for row in before['files']:
        old = row['path']
        path = ROOT / resolve(old)
        operation = operations.get(old, {})
        if operation.get('action') == 'delete_regenerable':
            deleted += 1
            if path.exists() or path.is_symlink():
                failures.append([old, 'retired artifact unexpectedly present'])
            continue
        if not path.is_file():
            failures.append([old, 'retained artifact missing'])
            continue
        expected = documents.get(old, {}).get('after_sha256', row['sha256'])
        if digest(path) != expected:
            failures.append([old, 'artifact hash mismatch'])
        if operation.get('action') == 'deduplicate_symlink':
            aliases += 1
            expected_target = ROOT / operation['replacement']
            if not path.is_symlink() or os.path.isabs(os.readlink(path)) or path.resolve() != expected_target.resolve():
                failures.append([old, 'relative alias mismatch'])
        checked += 1
    for name, expected in before['native_sha256'].items():
        if digest(ROOT / name) != expected:
            failures.append([name, 'original physical/acoustic implementation changed'])
    for name, entry in read(MIGRATION / 'source_changes.json').items():
        if digest(ROOT / name) != entry['after_sha256']:
            failures.append([name, 'unrecorded source change'])
    for entry in (read(MIGRATION / 'document_changes.json')
                  + read(MIGRATION / 'config_changes.json')
                  + read(MIGRATION / 'extra_reference_changes.json')):
        if digest(ROOT / entry['path']) != entry['after_sha256']:
            failures.append([entry['path'], 'unrecorded reference edit'])
    manifest = MIGRATION / 'migration_manifest.json'
    if manifest.exists():
        for name, expected in read(manifest)['management_sources'].items():
            if digest(ROOT / name) != expected:
                failures.append([name, 'migration management source hash mismatch'])
    result = dict(passed=not failures, retained_artifacts_checked=checked,
                  deleted_regenerable=deleted, relative_aliases_checked=aliases,
                  physics_and_acoustics_unchanged=True if not any('implementation changed' in r[1] for r in failures) else False,
                  generated_sounds=0, training_steps=0, failures=failures)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def redraw(input_path, output, reference=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from .acoustic_views import display_psd, relative_db
    from .plot_style import context, colorbar

    input_path = resolve(Path(input_path).absolute())
    reference = resolve(Path(reference).absolute()) if reference else input_path
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Use a new output directory; completed results must remain immutable')
    # Refuse results/ destinations for this inspection command.
    if output.is_relative_to(ROOT / 'results'):
        raise ValueError('Choose a separate output directory (for example /tmp/birdsong-redraw)')

    def wave(path):
        with np.load(path) as z:
            key = 'audio' if 'audio' in z else 'target'
            rate = int(z['sample_rate']) if 'sample_rate' in z else int(z['rate']) if 'rate' in z else 50000
            if rate != 50000:
                raise ValueError('This renderer uses the Phase1 50 kHz display protocol')
            return z[key].copy()

    audio, target = wave(input_path), wave(reference)
    f, t, power, _ = display_psd(audio)
    ref = display_psd(target)[3]
    output.mkdir(parents=True)
    with context():
        fig, ax = plt.subplots(figsize=(18, 12))
        mesh = ax.pcolormesh(t, f / 1000., relative_db(power, ref), shading='auto',
                             cmap='magma', vmin=-40, vmax=0, rasterized=True)
        ax.set(xlabel='Time [s]', ylabel='Frequency [kHz]', ylim=(0, 10), xlim=(t[0], t[-1]))
        colorbar(fig, mesh, [ax])
        fig.savefig(output / 'spectrogram.png', dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.plot(np.arange(len(audio)) / 50000., audio)
        ax.set(xlabel='Time [s]', ylabel='Pressure [model units]', xlim=(0, len(audio) / 50000.))
        fig.savefig(output / 'waveform.png', dpi=120)
        plt.close(fig)
    record = dict(input=str(input_path), input_sha256=digest(input_path),
                  reference=str(reference), reference_sha256=digest(reference),
                  reference_psd=ref, limits_db=[-40, 0], style_module='Deeplearning.plot_style',
                  generated_sounds=0, training_steps=0, physical_simulation=False)
    (output / 'redraw.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('audit')
    p = sub.add_parser('redraw')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference')
    args = parser.parse_args()
    if args.command == 'audit':
        if not audit()['passed']:
            raise SystemExit(1)
    else:
        redraw(args.input, args.output, args.reference)


if __name__ == '__main__':
    main()
