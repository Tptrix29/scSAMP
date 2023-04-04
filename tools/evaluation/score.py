import numpy as np
from sklearn.metrics import confusion_matrix


def specific_scores(y: np.array,
                    y_pred: np.array,
                    target: str
                    ) -> tuple:
    """
    Calculate evaluation scores.
    :param y: True value
    :param y_pred: Prediction
    :param target: Specific type
    :return: (accuracy, precision, recall, F1-score, cohen-kappa)

    Usage:
    >>> specific_scores(["A", "B", "C", "A", "B"], ["A", "B", "C", "B", "A"], "A")
    # (0.2, 0.5, 0.5, 0.5, 0.04761904761904763)
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
    p_observe = TP / total
    p_expect = sum0[loc] * sum1[loc] / (total ** 2)

    acc = TP / total
    precision = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0 if TP + FN == 0 else TP / (TP + FN)
    F1 = 0 if TP == 0 else 2 * precision * recall / (precision + recall)
    kappa = np.abs(p_observe - p_expect) / (1 - p_expect)

    return acc, precision, recall, F1, kappa


