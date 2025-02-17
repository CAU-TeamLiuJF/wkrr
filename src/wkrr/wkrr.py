import collections
import itertools
import timeit

import numpy as np
import scipy as sp

from joblib import Parallel, delayed
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

from .models.kernels import generate_candidate_powers
from .models.mle import parallel_gwas


def weighted_krr(X_train, y_train, X_test, p_values=None, top_quantile=0.01, base=np.e, gamma=1., alpha=1.):
    X_merge = np.concatenate([X_train, X_test], axis=0)
    n_samples_train, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    if p_values is not None:
        p_threshold = np.quantile(p_values, top_quantile)
        significant_indices = np.where(p_values<=p_threshold)

        weights = np.ones(p_values.size)
        weights[significant_indices] = 1 - np.log(p_values[significant_indices]) / np.log(base) + np.log(p_threshold) / np.log(base)
        weighted_X_merge = X_merge * np.sqrt(weights)

        K = rbf_kernel(weighted_X_merge, gamma=gamma)
        K_train = K[:n_samples_train, :n_samples_train]
        K_test = K[n_samples_train:, :n_samples_train]

        model = KernelRidge(kernel='precomputed', alpha=alpha)
        model = model.fit(K_train, y_train)
        pred = model.predict(K_test)

    else:
        K = rbf_kernel(X_merge, gamma=gamma)
        K_train = K[:n_samples_train, :n_samples_train]
        K_test = K[n_samples_train:, :n_samples_train]

        model = KernelRidge(kernel='precomputed', alpha=alpha)
        model = model.fit(K_train, y_train)
        pred = model.predict(K_test)

    return pred


def fit_and_score(X_train, y_train, X_val, y_val, p_values=None, top_quantile=0.01, base=np.e, gamma=1., alpha=1.):
    X_merge = np.concatenate([X_train, X_val], axis=0)
    n_samples_train, n_features = X_train.shape
    n_samples_val = X_val.shape[0]

    pred = weighted_krr(X_train, y_train, X_val, p_values=p_values, top_quantile=top_quantile, base=base, gamma=gamma, alpha=alpha)
    acc = np.corrcoef(pred, y_val)[0, 1]

    update_top_quantile = top_quantile if p_values is not None else np.inf
    update_base = base if p_values is not None else np.inf
    return (update_top_quantile, update_base, gamma, alpha, acc)


def run_wkrr(X_train, 
             y_train, 
             X_test, 
             rss, 
             n_splits=5, 
             n_repeats=1, 
             n_jobs=1):
    n_samples_train, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    time_begin = timeit.default_timer()

    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    candidate_alphas = np.logspace(-5, 3, 9, base=10)
    candidate_powers = generate_candidate_powers(X_train)

    grid_search_dict = collections.defaultdict(list)
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_sub, y_train_sub = X_train[train_idx, :], y_train[train_idx]
        X_val_sub, y_val_sub = X_train[val_idx, :], y_train[val_idx]

        result_dict = parallel_gwas(X_train_sub, y_train_sub, rss, n_jobs=n_jobs)
        p_values = np.array(result_dict['p_lrt'])

        # weighted version
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=2*n_jobs)
        with parallel:
            out = parallel(
                delayed(fit_and_score)(
                    X_train_sub, y_train_sub, 
                    X_val_sub, y_val_sub, 
                    p_values=p_values,
                    top_quantile=top_quantile, base=base, gamma=gamma, alpha=alpha
                )
                for top_quantile, base, gamma, alpha in itertools.product(
                    [0.0001, 0.001, 0.01], 
                    [np.e, 10.0], 
                    np.power(2., candidate_powers), 
                    candidate_alphas
                )
            )
        
        for (top_quantile, base, gamma, alpha, acc) in out:
            grid_search_dict['cv_no'].append(i+1)
            grid_search_dict['top_quantile'].append(top_quantile)
            grid_search_dict['base'].append(base)
            grid_search_dict['gamma'].append(gamma)
            grid_search_dict['alpha'].append(alpha)
            grid_search_dict['acc'].append(acc)
        
        # unweighted version
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=2*n_jobs)
        with parallel:
            out = parallel(
                delayed(fit_and_score)(
                    X_train_sub, y_train_sub, 
                    X_val_sub, y_val_sub, 
                    p_values=None, 
                    gamma=gamma, alpha=alpha
                )
                for gamma, alpha in itertools.product(
                    np.power(2., candidate_powers), 
                    candidate_alphas
                )
            )
        
        for top_quantile, base, gamma, alpha, acc in out:
            grid_search_dict['cv_no'].append(i+1)
            grid_search_dict['top_quantile'].append(top_quantile)
            grid_search_dict['base'].append(base)
            grid_search_dict['gamma'].append(gamma)
            grid_search_dict['alpha'].append(alpha)
            grid_search_dict['acc'].append(acc)
    
    import pandas as pd
    pd.DataFrame(grid_search_dict).to_csv("grid_search.csv", index=False)


    # find the best params
    best_params = None
    best_score = -np.inf
    for (top_quantile, base, gamma, alpha), grid_search_df in pd.DataFrame(grid_search_dict).groupby(['top_quantile', 'base', 'gamma', 'alpha']):
        score = np.mean(grid_search_df['acc'])
        if score > best_score:
            best_score = score
            best_params = {
                'top_quantile': top_quantile,
                'base': base,
                'gamma': gamma, 
                'alpha': alpha
            }
    
    if np.isinf(best_params['top_quantile']):  # unweighted
        pred = weighted_krr(X_train, y_train, X_test, p_values=None, **best_params)

    else:  # weighted
        result_dict = parallel_gwas(X_train, y_train, rss, n_jobs=n_jobs)
        p_values = np.array(result_dict['p_lrt'])

        pred = weighted_krr(X_train, y_train, X_test, p_values=p_values, **best_params)
    
    print("best_params: ", best_params)
    return pred


