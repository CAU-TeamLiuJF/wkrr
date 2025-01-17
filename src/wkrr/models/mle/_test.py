import numpy as np
import scipy as sp


def likelihood_ratio_test(l_mle_H0, l_mle_H1):
    likelihood_ratio = 2 * (l_mle_H1 - l_mle_H0)
    p_lrt = sp.stats.chi2.sf(likelihood_ratio, df=1)

    return p_lrt
