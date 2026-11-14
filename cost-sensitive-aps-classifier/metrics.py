"""Shared policy metrics: offline training and the UI use identical arithmetic."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def cost_row(y, scores, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and scores must be nonempty aligned vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores.")
    if ((scores < 0) | (scores > 1)).any() or not np.isfinite(threshold):
        raise ValueError("Invalid score range or threshold.")
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn),
                tp=int(tp), cost=int(10 * fp + 500 * fn))


def threshold_curve(y, scores):
    # Exact candidate cutoffs; equivalent tied scores are never split. Include
    # the no-inspection policy even if a score is exactly one.
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    cost_row(y, scores, .5)  # validate inputs
    y = y.astype(int)
    order = np.argsort(scores, kind="stable")
    s, labels = scores[order], y[order]
    thresholds = np.unique(np.r_[0., s, np.nextafter(1., 2.)])
    left = np.searchsorted(s, thresholds, side="left")
    cum_pos = np.r_[0, np.cumsum(labels)]
    fn = cum_pos[left]
    tn = left - fn
    tp = labels.sum() - fn
    fp = len(y) - labels.sum() - tn
    return pd.DataFrame(dict(threshold=thresholds, tn=tn, fp=fp, fn=fn,
                             tp=tp, cost=10 * fp + 500 * fn))


def select_threshold(y, scores):
    curve = threshold_curve(y, scores)
    best = curve.sort_values(["cost", "fn", "fp", "threshold"]).iloc[0]
    return float(best.threshold), curve


def report(y, scores, threshold):
    scores = np.asarray(scores)
    pred = scores >= threshold
    return dict(**cost_row(y, scores, threshold),
                roc_auc=float(roc_auc_score(y, scores)),
                average_precision=float(average_precision_score(y, scores)),
                recall=float(recall_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)))
