import collections
import timeit

import numpy as np
import scipy as sp

from joblib import Parallel, delayed

from ._eval import eval_mle_cython_w, eval_mle_cython_x
from ._eval import eval_mle_1st_cython_w, eval_mle_1st_cython_x
from ._test import likelihood_ratio_test


def fit_optimization_w(UTw, UTy, S, n_intervals=10, ldelta_lb=-5, ldelta_ub=5):
    # grid eval with first order
    mle_1st_grid = np.ones(n_intervals+1) * -np.inf
    ldelta_grid = np.arange(n_intervals+1)/(n_intervals*1.) * (ldelta_ub-ldelta_lb) + ldelta_lb
    for i in np.arange(n_intervals)+1:
        mle_1st_grid[i] = eval_mle_1st_cython_w(ldelta_grid[i], S, UTw, UTy)
    
    # find region with changed sign
    candidate_indices = [i for i in np.arange(n_intervals-1) if not np.isinf(mle_1st_grid[i]) and np.sign(mle_1st_grid[i]) != np.sign(mle_1st_grid[i+1])]
    candidate_ldeltas = ldelta_grid[candidate_indices]

    # find root on first derivative by Brent
    brent_roots = np.ones(candidate_ldeltas.size) * np.inf
    for i, ldelta in enumerate(candidate_ldeltas):
        brent_roots[i] = sp.optimize.brentq(
            eval_mle_1st_cython_w, ldelta, ldelta+1, args=(S, UTw, UTy), full_output=False)
    
    # find the global optimal ldelta / mle
    opt_mles = np.array([eval_mle_cython_w(ldelta, S, UTw, UTy) for ldelta in brent_roots])
    mle_max = opt_mles.max()
    ldelta_opt_glob = brent_roots[opt_mles.argmax()]

    return mle_max, ldelta_opt_glob


def fit_optimization_x(UTw, UTx, UTy, S, n_intervals=10, ldelta_lb=-5, ldelta_ub=5):
    # grid eval with first order
    mle_1st_grid = np.ones(n_intervals+1) * -np.inf
    ldelta_grid = np.arange(n_intervals+1)/(n_intervals*1.) * (ldelta_ub-ldelta_lb) + ldelta_lb
    for i in np.arange(n_intervals)+1:
        mle_1st_grid[i] = eval_mle_1st_cython_x(ldelta_grid[i], S, UTw, UTx, UTy)
    
    # find region with changed sign
    candidate_indices = [i for i in np.arange(n_intervals-1) if not np.isinf(mle_1st_grid[i]) and np.sign(mle_1st_grid[i]) != np.sign(mle_1st_grid[i+1])]
    candidate_ldeltas = ldelta_grid[candidate_indices]

    # find root on first derivative by Brent
    brent_roots = np.ones(candidate_ldeltas.size) * np.inf
    for i, ldelta in enumerate(candidate_ldeltas):
        brent_roots[i] = sp.optimize.brentq(
            eval_mle_1st_cython_x, ldelta, ldelta+1, args=(S, UTw, UTx, UTy), full_output=False)
    
    # find the global optimal ldelta / mle
    opt_mles = np.array([eval_mle_cython_x(ldelta, S, UTw, UTx, UTy) for ldelta in brent_roots])
    mle_max = opt_mles.max()
    ldelta_opt_glob = brent_roots[opt_mles.argmax()]

    return mle_max, ldelta_opt_glob


def run_gwas(X, y, rss):
    n_samples, n_features = X.shape
    w = np.ones((n_samples, 1))

    K = X @ X.T / n_features
    S, U = np.linalg.eigh(K)
    UTy = (U.T @ y.reshape(-1, 1)).flatten()
    UTw = (U.T @ w.reshape(-1, 1)).flatten()
    UTX = np.asarray(U.T @ X, dtype=np.float64)

    # H0 model: null model
    # time_begin = timeit.default_timer()

    l_mle_H0, ldelta_H0 = fit_optimization_w(UTw, UTy, S)

    # print("H0", timeit.default_timer()-time_begin)

    # H1 model
    result_dict = collections.defaultdict(list)
    for (i, rs) in enumerate(rss):
        # time_begin = timeit.default_timer()

        l_mle_H1, ldelta_H1 = fit_optimization_x(UTw, UTX[:, i], UTy, S)

        result_dict['rs'].append(rs)
        result_dict['l_mle_H1'].append(l_mle_H1)

        # print(i, rs, l_mle_H0, l_mle_H1, timeit.default_timer()-time_begin)
    
    result_dict['l_mle_H0'] = l_mle_H0
    result_dict['p_lrt'] = likelihood_ratio_test(l_mle_H0, result_dict['l_mle_H1'])

    return result_dict


def parallel_gwas(X, y, rss, n_jobs=1):
    n_samples, n_features = X.shape

    w = np.ones((n_samples, 1))

    K = X @ X.T / n_features
    S, U = np.linalg.eigh(K)
    UTy = (U.T @ y.reshape(-1, 1)).flatten()
    UTw = (U.T @ w.reshape(-1, 1)).flatten()
    UTX = np.asarray(U.T @ X, dtype=np.float64)

    # H0 model: null model
    l_mle_H0, ldelta_H0 = fit_optimization_w(UTw, UTy, S)

    # H1 model
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=2*n_jobs)
    with parallel:
        out = parallel(
            delayed(fit_optimization_x)(
                UTw, UTX[:, i], UTy, S
            )
            for (i, rs) in enumerate(rss)
        )

    # organize results
    result_dict = collections.defaultdict(list)
    for i, (l_mle_H1, ldelta) in enumerate(out):
        result_dict['rs'].append(rss[i])
        result_dict['l_mle_H1'].append(l_mle_H1)

    result_dict['l_mle_H0'] = l_mle_H0
    result_dict['p_lrt'] = likelihood_ratio_test(l_mle_H0, result_dict['l_mle_H1'])

    return result_dict
