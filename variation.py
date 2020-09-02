import numpy as np

import sys

sys.path.insert(1, '../logistic_models')
from neutrality_analysis import JensenShannon

from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def variation_coefficient(ts):
    if np.any(np.isnan(ts)):
        return (np.nan,) * 4

    d = ts.drop('time', axis=1)
    x = d.mean(axis=0)
    y = d.std(axis=0) / x
    return np.mean(y), np.std(y), np.min(y), np.max(y)


def JS(ts, verbose=False):
    if np.any(np.isnan(ts)) or np.all(ts.iloc[-1, 1:] == 0):
        return (np.nan,) * 5

    ts_t = ts.drop('time', axis=1)

    # pseudo-count
    if np.any(ts == 0):
        ts_t += 1e-6 * np.min(ts_t[ts_t > 0])

    JS = np.zeros(500)
    ti = np.zeros(500)

    for i in range(len(JS)):
        a = np.random.randint(len(ts_t))
        b = a

        while np.abs(a - b) < 3:
            b = np.random.randint(len(ts))

        ti[i] = np.abs(a - b)  # time interaval
        JS[i] = JensenShannon(ts_t.iloc[a], ts_t.iloc[b])

    if sum(~np.isnan(JS)) > 2:
        corr, pval = pearsonr(JS[~np.isnan(JS)], ti[~np.isnan(JS)])
    else:
        corr, pval = np.nan, np.nan

    if verbose:
        plt.figure()
        plt.scatter(ti, JS)
        plt.title('corr = %.3E, \n pval = %.3E' % (corr, pval))
        plt.xlabel('Time intervals')
        plt.ylabel('JS distance')

    return np.mean(JS), np.std(JS), np.min(JS), np.max(JS), pval
