import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                              precision_recall_curve, brier_score_loss)


def choose_threshold(y, scores, target_recall=.90):
    """Validation only: max precision at recall >= target; ties: largest threshold."""
    y, scores = np.asarray(y).ravel(), np.asarray(scores).ravel()
    if len(np.unique(y)) != 2:
        raise ValueError('Threshold selection requires both classes')
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0,1]')
    precision, recall, thresholds = precision_recall_curve(y, scores)
    candidates = np.flatnonzero(recall[:-1] >= target_recall)
    best = max(candidates, key=lambda i: (precision[i], thresholds[i]))
    return float(thresholds[best])


def binary_metrics(y, scores, threshold):
    y, scores = np.asarray(y).ravel(), np.asarray(scores).ravel()
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid score shape or non-finite score')
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def ratio(a, b):
        return float(a/b) if b else None
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=ratio(tp+tn, len(y)), precision=ratio(tp, tp+fp),
                recall=ratio(tp, tp+fn), specificity=ratio(tn, tn+fp),
                f1=ratio(2*tp, 2*tp+fp+fn),
                ap=float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
                auroc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
                brier=float(brier_score_loss(y, scores)))


def patient_bootstrap_ap(frame, score_a='score', score_b=None, repeats=200, seed=42):
    """Image-level AP, whole-patient resampling; paired delta is B minus A."""
    rng = np.random.default_rng(seed)
    blocks = [g.index.to_numpy() for _, g in frame.reset_index(drop=True).groupby('patient_id')]
    frame = frame.reset_index(drop=True)
    values = []
    for _ in range(repeats):
        indexes = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        sample = frame.iloc[indexes]
        if sample.target.nunique() < 2:
            continue
        a = average_precision_score(sample.target, sample[score_a])
        values.append(average_precision_score(sample.target, sample[score_b]) - a if score_b else a)
    if len(values) < max(10, repeats // 2):
        return {'low': None, 'high': None, 'valid_replicates': len(values), 'requested': repeats}
    lo, hi = np.quantile(values, [.025, .975])
    return {'low': float(lo), 'high': float(hi), 'valid_replicates': len(values), 'requested': repeats}
