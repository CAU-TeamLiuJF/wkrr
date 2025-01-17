import numpy as np

from numba import njit, prange


@njit('float64[:](float32[:, :])', parallel=True, fastmath=True)
def cal_euclidean_distance(X: np.ndarray[:, :]):
    """ Calculate Euclidean distance with parallel.
    
    Parameters
    ----------
    X : np.ndarray, dtype=np.float32
        Genotypes of samples with shape (n_samples, n_snps)

    Return
    ------
    np.ndarray[:], dtype=np.float64 : euclidean_distance
    """
    n_samples, n_features = X.shape

    euclidean_distance = np.zeros(int(n_samples * (n_samples+1) / 2))

    for i in prange(1, n_samples+1):
        for j in prange(i, n_samples+1):
            uppder_indice = int(n_samples * (i-1) + j - i * (i-1) / 2 - 1)
            for k in prange(n_features):
                euclidean_distance[uppder_indice] += (X[i-1, k] - X[j-1, k])**2
    return euclidean_distance


def generate_candidate_powers(X: np.ndarray[:, :]):
    """ Generate list of candidate powers.

    candidate_powers = [
        np.floor(expected_power) - 1,
        np.floor(expected_power), 
        np.ceil(expected_power), 
        np.ceil(expected_power) + 1
    ]
    
    Parameters
    ----------
    X : np.ndarray, dtype=np.float32
        Genotypes of samples with shape (n_samples, n_snps)

    Return
    ------
    np.ndarray[:], dtype=np.float64 : candidate_powers
    """
    euclidean_distance = cal_euclidean_distance(X)
    nonzero_indices = euclidean_distance.nonzero()
    median_distance = np.median(euclidean_distance[nonzero_indices])

    expected_power = np.log2(1 / median_distance)
    
    power_lb = np.floor(expected_power) - 3
    candidate_powers = power_lb + np.arange(0, 8, 1)
    return candidate_powers
