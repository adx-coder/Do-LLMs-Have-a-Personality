import numpy as np
from scipy import stats


def calculate_gini(array):
    array = np.sort(np.array(array))
    n = len(array)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * array)) / (n * np.sum(array)) - (n + 1) / n

