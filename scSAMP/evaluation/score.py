import numpy as np
from sklearn.metrics import confusion_matrix


def specific_scores(y: np.array,
                    y_pred: np.array,
                    target: str
                    ) -> tuple:
    """
    Calculate evaluation scores.
    Including 'Accuracy', 'Precision', 'Recall', 'F1-score', 'cohen-kappa'.

    Parameters
    ----------
    y
        Ground truth value in :class:`~numpy.array` format, 1-d array
    y_pred
        Predicted value in :class:`~numpy.array` format, 1-d array
    target
        Specific clsuter label.

    Returns
    -------
        Score tuple of specific cluster.

    Examples
    --------
    >>> specific_scores(["A", "B", "C", "A", "B"], ["A", "B", "C", "B", "A"], "A")
    (0.2, 0.5, 0.5, 0.5, 0.04761904761904763)
    >>> specific_scores(["A", "B", "C", "A", "B"], ["B", "B", "C", "B", "A"], "A")
    (0.0, 0.0, 0.0, 0, 0.08695652173913043)
    """

    confusion = confusion_matrix(y, y_pred)

    clsts = np.unique(np.vstack([y, y_pred]))
    loc = None
    for i in range(len(clsts)):
        if clsts[i] == target:
            loc = i
            break

    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    total = np.sum(confusion)

    TP = confusion[loc][loc]
    FP = sum0[loc] - TP
    FN = sum1[loc] - TP

    precision = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0 if TP + FN == 0 else TP / (TP + FN)
    F1 = 0 if TP == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, F1


