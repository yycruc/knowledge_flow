"""Single entry point: python run.py [--output PATH]."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Reproduce every final quantitative result and Figure 1–7 offline.')
    parser.add_argument('--output', type=Path, default=root / 'results', help='Output directory (default: results beside this script)')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == root or output == root / 'data':
        raise ValueError('Choose a dedicated output directory, not the code or source-data directory')
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf8'))
    for name, item in manifest['files'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Source input is missing or has changed: {name}')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    from analysis import compute
    from plots import draw_all
    plot_data, tables, summary = compute(root / 'data', output)
    draw_all(plot_data, output / 'figures')
    import numpy, pandas, scipy, matplotlib, seaborn, statsmodels
    summary['software'] = {module.__name__: module.__version__ for module in
                           [numpy, pandas, scipy, matplotlib, seaborn, statsmodels]}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'Done. All results written to: {output}', flush=True)


if __name__ == '__main__':
    main()
