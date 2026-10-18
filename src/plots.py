from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve


def history_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.epoch, history.train_weighted_loss, label='Train weighted BCE')
    axes[1].plot(history.epoch, history.val_bce, label='Validation unweighted BCE')
    axes[2].plot(history.epoch, history.val_ap, label='Validation AP')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    # Weighted train loss and unweighted val BCE are deliberately on separate axes.
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.show()


def evaluation_plot(y, scores, metrics, path):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    p, r, _ = precision_recall_curve(y, scores)
    axes[0].plot(r, p); axes[0].axhline(np.mean(y), ls='--', color='gray')
    axes[0].set(xlabel='Recall', ylabel='Precision', title='PR curve / prevalence reference')
    fpr, tpr, _ = roc_curve(y, scores)
    axes[1].plot(fpr, tpr); axes[1].plot([0,1], [0,1], '--', color='gray')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC curve')
    matrix = [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]]
    axes[2].imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(matrix[i][j]), ha='center', va='center')
    axes[2].set(xticks=[0,1], yticks=[0,1], xlabel='Predicted', ylabel='True', title=f"Threshold={metrics['threshold']:.3f}")
    prob_true, prob_pred = calibration_curve(y, scores, n_bins=10, strategy='uniform')
    axes[3].plot(prob_pred, prob_true, 'o-'); axes[3].plot([0,1], [0,1], '--', color='gray')
    counts, _ = np.histogram(scores, bins=np.linspace(0,1,11))
    axes[3].set(xlabel='Mean score (10 equal-width bins)', ylabel='Observed positive fraction', title='Reliability: inspect support CSV')
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.show()
    return counts
