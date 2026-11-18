"""Fit one global temperature from saved logits; no model inference is repeated."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np

from .data import setting_key
from .io import sha256, write_json
from .metrics import report


def fit_temperature(rows, bounds=(.05, 20.)):
    groups = sorted({setting_key(r) for r in rows})
    width = max(len(r['choices']) for r in rows)
    logits = np.full((len(rows), width), -np.inf, dtype=np.float64)
    correct = []
    for i, row in enumerate(rows):
        z = np.asarray(row['option_logits'], dtype=np.float64)
        if z.shape != (len(row['choices']),) or not np.isfinite(z).all():
            raise ValueError('Invalid saved option logits')
        logits[i, :len(z)] = z-z.max()
        correct.append(row['choices'][int(z.argmax())] == row['gt_answer'])
    g = np.asarray([groups.index(setting_key(r)) for r in rows])
    counts = np.bincount(g, minlength=len(groups))
    cache = {}
    def objective(log_t):
        log_t = float(log_t)
        if log_t not in cache:
            confidence = 1/np.exp(logits/math.exp(log_t)).sum(axis=1)
            bins = np.searchsorted(np.arange(1, 11)/10, confidence, side='left')
            residual = np.bincount(g*10+bins, weights=np.asarray(correct)-confidence,
                                   minlength=len(groups)*10).reshape(len(groups), 10)
            cache[log_t] = float(np.mean(np.abs(residual).sum(axis=1)/counts))
        return cache[log_t]
    lo, hi = map(math.log, bounds)
    grid = np.linspace(lo, hi, 257)
    values = [objective(t) for t in grid]
    objective(0.0)
    minima = [i for i in range(1, 256) if values[i] <= values[i-1] and values[i] <= values[i+1]]
    for i in sorted(minima, key=lambda j: (values[j], abs(grid[j])))[:8]:
        left, right = grid[i-1], grid[i+1]
        for _ in range(6):
            points = np.linspace(left, right, 33)
            scores = [objective(t) for t in points]
            best = min(range(33), key=lambda j: (scores[j], abs(points[j])))
            left, right = points[max(0, best-1)], points[min(32, best+1)]
    best = min(cache, key=lambda t: (cache[t], abs(t)))
    return {'temperature': math.exp(best), 'fit_ece10_before': objective(0.), 'fit_ece10_after': cache[best],
            'search_evaluations': len(cache), 'bounds': list(bounds),
            'objective': 'equal-setting ECE10 on the selected calibration records',
            'search': 'deterministic log-grid and local refinement; approximate, not a global certificate'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fraction', type=float, default=.1)
    p.add_argument('--seed', type=int, default=20261112)
    args = p.parse_args()
    if not 0 < args.fraction <= 1:
        p.error('--fraction must be in (0,1]')
    source, output = Path(args.predictions).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty calibration output directory')
    rows = [json.loads(line) for line in source.open(encoding='utf-8')]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Predictions must be nonempty and have unique IDs')
    complete_path = source.parent/'completed.json'
    if not complete_path.exists() or json.loads(complete_path.read_text())['predictions_sha256'] != sha256(source):
        raise ValueError('Use predictions from a completed evaluation run')
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:max(1, math.ceil(len(rows)*args.fraction))]
    selected = [rows[i] for i in indices]
    fit = fit_temperature(selected)
    run = json.loads((source.parent/'run.json').read_text())
    fit.update(population=len(rows), fit_n=len(selected), seed=args.seed, fraction=args.fraction,
               selected_ids=[r['id'] for r in selected], predictions_sha256=sha256(source),
               model_files=run['model_files'], adapter_files=run['adapter_files'],
               full_metrics_include_calibration_samples=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'temperature.json', fit)
    # Freeze T before applying it to the full input set. No refitting on full metrics.
    result = []
    with (output/'predictions.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            z = np.asarray(row['option_logits'], dtype=np.float64)
            probabilities = np.exp((z-z.max())/fit['temperature'])
            probabilities /= probabilities.sum()
            new = {**row, 'probabilities': probabilities.tolist(), 'temperature': fit['temperature']}
            handle.write(json.dumps(new, ensure_ascii=False, allow_nan=False)+'\n')
            result.append(new)
    metrics = report(result)
    metrics['calibration'] = {key: fit[key] for key in ('fit_n', 'population', 'seed', 'temperature', 'full_metrics_include_calibration_samples')}
    write_json(output/'metrics.json', metrics)
    print(json.dumps({key: fit[key] for key in ('temperature', 'fit_n', 'fit_ece10_before', 'fit_ece10_after')}, indent=2))


if __name__ == '__main__':
    main()
