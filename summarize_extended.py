"""Summarize all pre-specified seeds, refusing incomplete runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    args.runs_dir=args.runs_dir.resolve()
    output = ROOT / 'reports/extended_30ep'
    output.mkdir(exist_ok=True)
    rows = []
    manifests = []
    histories = {}
    for seed in (42, 43, 44):
        run = args.runs_dir / f'extended-30ep-seed{seed}'
        results = json.loads((run/'results.json').read_text())
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['seed'] == seed and manifest['epochs'] == 30 and manifest['split_seed'] == 42
        manifests.append(manifest)
        histories[str(seed)] = results
        labels = None
        row = {'seed': seed}
        for arm in ('baseline','aft_rbf'):
            assert len(results[arm]['history']) == 30
            predictions = json.loads((run/f'{arm}_test_predictions.json').read_text())
            if labels is not None: assert labels == predictions['labels']
            labels = predictions['labels']
            count = sum(a==b for a,b in zip(predictions['predictions'],labels))
            assert len(labels)==10000
            best = results[arm]['best_epoch']
            row[arm] = {'test_accuracy_percent': count/100,
                        'test_correct': count,'test_cross_entropy':results[arm]['test']['loss'],'best_epoch':best,
                        'validation_accuracy_percent': results[arm]['history'][best-1]['validation']['accuracy']*100,
                        'epoch5_validation_percent':results[arm]['history'][4]['validation']['accuracy']*100,
                        'epoch30_training_percent':results[arm]['history'][-1]['train_accuracy']*100}
        row['aft_minus_baseline_pp'] = row['aft_rbf']['test_accuracy_percent']-row['baseline']['test_accuracy_percent']
        rows.append(row)
    assert all(m['train_indices']==manifests[0]['train_indices'] and m['validation_indices']==manifests[0]['validation_indices'] for m in manifests)
    summary = {'seeds':rows,'seed_count':3,'epochs':30,'split_seed':42,
               'scope':'fixed 10k train/2k validation/10k test; exploratory comparison after earlier test inspection',
               'mean':{arm:float(np.mean([r[arm]['test_accuracy_percent'] for r in rows])) for arm in ('baseline','aft_rbf')},
               'sample_std':{arm:float(np.std([r[arm]['test_accuracy_percent'] for r in rows],ddof=1)) for arm in ('baseline','aft_rbf')},
               'mean_paired_delta_pp':float(np.mean([r['aft_minus_baseline_pp'] for r in rows])),
               'paired_delta_sample_std_pp':float(np.std([r['aft_minus_baseline_pp'] for r in rows],ddof=1)),
               'initial_state_hashes':{str(m['seed']):m['initial_state_sha256'] for m in manifests}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'histories.json').write_text(json.dumps(histories,indent=2)+'\n')
    evidence={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for seed in (42,43,44) for f in (args.runs_dir/f'extended-30ep-seed{seed}').glob('*.json')}
    (output/'evidence_sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':main()
