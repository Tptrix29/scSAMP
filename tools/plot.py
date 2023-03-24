import numpy as np
import matplotlib.pyplot as plt

from typing import Union


def line_eval(
        x: Union[np.array, list],
        y_array: np.array,
        line_labels: list[str],
        prob_lim: bool,
        title: str = None,
        xlab: str = None,
        ylab: str = None
) -> None:
    x = np.array(x).reshape(-1)
    if x.shape[0] != y_array.shape[1]:
        raise ValueError("shape of 'x' and 'y_array' is inconsistent")

    plt.figure()
    for i in range(y_array.shape[0]):
        y = y_array[i, :]
        plt.plot(x, y, '-o')
    plt.legend(line_labels, loc=2, bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if prob_lim:
        plt.ylim([0, 1])
    plt.show()


def kernel_distribution_eval(X, multi=False):
    import scipy.stats as st
    import matplotlib.pyplot as plt
    X = np.array(X)
    if not multi:
        kde = st.gaussian_kde(X)
        X.sort()
        dens = kde.evaluate(X)
        plt.plot(X, dens)
    else:
        for i in range(X.shape[0]):
            kernel_distribution_eval(X[i])