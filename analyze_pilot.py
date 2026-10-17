"""Recompute paired pilot metrics from saved predictions (no model/API calls)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/real-pilot-2026-10-16')
    parser.add_argument('--output', default='reports/real_pilot_metrics.json')
    args = parser.parse_args()
    root = Path(args.run)
    records = {name: json.loads((root / f'{name}_test_predictions.json').read_text())
               for name in ('baseline', 'aft_rbf')}
    labels = np.asarray(records['baseline']['labels'])
    assert np.array_equal(labels, records['aft_rbf']['labels'])
    assert len(labels) == 10000
    correct = {name: np.asarray(record['predictions']) == labels for name, record in records.items()}
    delta = correct['aft_rbf'].astype(int) - correct['baseline'].astype(int)
    rng = np.random.default_rng(20261016)
    bootstrap = np.concatenate([delta[rng.integers(0, len(delta), (100, len(delta)))].mean(1)
                                for _ in range(20)]) * 100
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['train_indices']) == 10000
    assert len(manifest['validation_indices']) == 2000
    assert not set(manifest['train_indices']) & set(manifest['validation_indices'])
    result = {'examples': len(labels), 'correct': {name: int(x.sum()) for name, x in correct.items()},
              'accuracy_percent': {name: float(x.mean() * 100) for name, x in correct.items()},
              'aft_minus_baseline_pp': float(delta.mean() * 100),
              'aft_only_correct': int((delta == 1).sum()),
              'baseline_only_correct': int((delta == -1).sum()),
              'paired_bootstrap_95_percentile_interval_pp': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
              'bootstrap': {'seed': 20261016, 'resamples': 2000,
                            'scope': 'test-example sampling conditional on this single training seed; not training-seed uncertainty'},
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in root.glob('*_test_predictions.json')}}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
