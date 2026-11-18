"""Sample-weighted and equal-setting metrics, with label-aware F1."""
from collections import Counter, defaultdict
import math
import numpy as np

from .data import setting_key


def summarize(rows):
    if not rows:
        raise ValueError('No predictions to summarize')
    correct, confidences, briers = [], [], []
    truth, predicted, tp = Counter(), Counter(), Counter()
    for row in rows:
        probabilities = np.asarray(row['probabilities'], dtype=np.float64)
        if (probabilities.shape != (len(row['choices']),) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or abs(probabilities.sum()-1) > 1e-5):
            raise ValueError('Invalid prediction probabilities')
        index = int(probabilities.argmax())
        label, target = row['choices'][index], row['gt_answer']
        if row['prediction'] != label:
            raise ValueError('Prediction differs from the largest option probability')
        hit = label == target
        truth[target] += 1; predicted[label] += 1
        tp[target] += int(hit)
        correct.append(hit); confidences.append(float(probabilities[index]))
        one_hot = np.zeros(len(probabilities)); one_hot[row['choices'].index(target)] = 1
        briers.append(float(np.square(probabilities-one_hot).sum()))
    labels = sorted(set(truth) | set(predicted))
    f1 = {label: 2*tp[label]/(truth[label]+predicted[label]) for label in labels}
    confidence = np.asarray(confidences)
    hit = np.asarray(correct, dtype=float)
    bins = np.searchsorted(np.arange(1, 11)/10, confidence.clip(0, 1), side='left')
    residual = np.bincount(bins, weights=hit-confidence, minlength=10)
    return {'n': len(rows), 'correct': int(sum(correct)), 'accuracy': float(hit.mean()),
            'macro_f1': float(np.mean(list(f1.values()))),
            'weighted_f1': sum(f1[label]*truth[label] for label in labels)/len(rows),
            'ece10': float(np.abs(residual).sum()/len(rows)),
            'brier': float(np.mean(briers))}


def report(rows):
    settings, sources = defaultdict(list), defaultdict(list)
    for row in rows:
        settings[setting_key(row)].append(row)
        sources[row['source']].append(row)
    per_setting = [dict(source=key[0], subtask=key[1], modality=list(key[2]), **summarize(value))
                   for key, value in sorted(settings.items())]
    keys = ('accuracy', 'macro_f1', 'weighted_f1', 'ece10', 'brier')
    return {'overall_sample_weighted': summarize(rows),
            'mean_over_settings': {k: sum(s[k] for s in per_setting)/len(per_setting) for k in keys},
            'setting_count': len(per_setting), 'per_setting': per_setting,
            'per_source': {source: summarize(value) for source, value in sorted(sources.items())},
            'definitions': {'scale': '0..1; multiply by 100 for percent',
                            'ece10': '10 equal-width bins (0,.1],...,(.9,1]',
                            'brier': 'sum over classes, without the 1/2 factor',
                            'setting': 'source + subtask + target modality; distinguishes both CREMA-D settings'}}
