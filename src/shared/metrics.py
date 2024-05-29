import Levenshtein
import numpy as np


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def average_levenshtein_distance(array1, array2):
    if not array1 or not array2:
        return 0  # Avoid division by zero if any array is empty

    total_distance = 0
    count = 0

    for i in range(len(array1)):
        for j in range(len(array2)):
            if i <= j:
                distance = Levenshtein.distance(array1[i], array2[j])
                total_distance += distance
                count += 1

    average_distance = total_distance / count
    return average_distance
