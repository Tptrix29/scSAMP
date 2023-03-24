import numpy as np
import pandas as pd


def _compactness(data: np.array) -> float:
    avg_expr = data.mean(axis=0)
    return np.abs(np.corrcoef(avg_expr, data)[0, :]).mean()


def _proto_concave(X: np.array) -> np.array:
    return (np.exp(X) - 1) / (np.e - 1)


def _proto_convex(X: np.array) -> np.array:
    return - X * (np.log(X) - 1)


def _entropy_1d(X: np.array) -> np.array:
    return -1 / np.log(X.shape[0]) * np.sum(X * np.log(X))


def compactness_factor(
        info_data: pd.DataFrame,
        numeric_arr: np.array,
        label: str,
) -> (pd.Index, np.array):
    """
    Compactness Factor Calculation
        compactness-factor = 1 - compactness
    :param info_data: dataframe including cluster info
    :param numeric_arr: expression matrix
    :param label: cluster label name
    :return: (cluster types info, compactness factor array)
    """
    summary = info_data.groupby(label).apply(
        lambda x: 1 - _compactness(numeric_arr[np.array(x.index.astype(int).tolist()), :])
    )
    return summary.index, summary / summary.sum()


def complexity_factor(
        info_data: pd.DataFrame,
        numeric_arr: np.array,
        label: str,
) -> (pd.Index, np.array):
    """
    Complexity Factor Calculation:
        complexity-factor = complexity
    :param info_data: dataframe including cluster info
    :param numeric_arr: expression matrix
    :param label: cluster label name
    :return: (cluster types info, complexity factor array)
    """
    avg = info_data.groupby(label).apply(
        lambda x: np.mean(numeric_arr[np.array(x.index.astype(int).tolist()), :], axis=0)
    )
    # calculate inter-cluster complexity
    coef = np.abs(np.corrcoef(np.array(avg.to_list())))
    for i in range(coef.shape[0]):
        coef[i, i] = 0
    summary = pd.Series(coef.max(axis=1), index=avg.index)  # complexity
    return summary.index, summary / summary.sum()


def concave_2var(vec1: np.array, vec2:np.array) -> np.array:
    """
    Concave Integration
    :param vec1: compactness factor
    :param vec2: complexity factor
    :return: concave index
    """
    return (_proto_concave(vec1) + _proto_concave(vec2)) / 2


def convex_2var(vec1: np.array, vec2: np.array) -> np.array:
    """
    Convex Integration
    :param vec1: compactness factor
    :param vec2: complexity factor
    :return: convex index
    """
    return (_proto_convex(vec1) + _proto_convex(vec2)) / 2


def entropy_2var(vec1: np.array, vec2: np.array) -> np.array:
    """
    Entropy Integration
    :param vec1: compactness factor
    :param vec2: complexity factor
    :return: integrated index with entropy weight
    """
    d1 = 1 - _entropy_1d(vec1)
    d2 = 1 - _entropy_1d(vec2)
    return (d1 * vec1 + d2 * vec2) / (d1 + d2)