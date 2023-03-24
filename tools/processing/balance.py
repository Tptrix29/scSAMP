import numpy as np


def balance_power(
        x: np.array,
        a: float = 3
) -> np.array:
    """
    Balance function in power format. (Symmetric)
    :param x: 1-d array
    :param a: scale factor
    :return: 1-d array
    """
    p = (x-0.5) * (a*x**2-a*x+1) + 0.5
    return p / p.sum()


def balance_inv_sigmoid(
        x: np.array
) -> np.array:
    """
    Balance function in inverse sigmoid format. (Asymmetric)
    :param x: 1-d array
    :return: 1-d array
    """
    p = np.log(1/(1-x)-1)+0.5
    p = p - p.min()
    return p / p.sum()


def balance_tan(
        x: np.array
) -> np.array:
    """
    Balance function in tangent format. (Symmetric)
    :param x: 1-d array
    :return: 1-d array
    """
    p = np.tan(np.pi*(x-0.5)) + 0.5
    p = p - p.min()
    return p / p.sum()


