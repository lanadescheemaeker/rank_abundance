from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(1, '../plot_functions/')
from piecewise_normalizations import PiecewiseLogNorm, PiecewiseNormalize

import powerlaw

normalizations = {'width': PiecewiseLogNorm([0.1, 2, 15], [0, 0.5, 1]),
                  'pval': PiecewiseNormalize([0, 0.05, 1], [0, 0.5, 1]),
                  'p': PiecewiseNormalize([0, 0.05, 1], [0, 0.5, 1]),
                  'diff_width': PiecewiseLogNorm([-10, 0, 10], [0, 0.5, 1], linthresh=1e-2),
                  'filling': PiecewiseLogNorm([0.1, 1, 10], [0, 0.5, 1]),
                  'number': PiecewiseNormalize([0, 100, 200], [0, 0.5, 1]),
                  # 'stability' : PiecewiseLogNorm([-0.1,0,0.1], [0, 0.5, 1], linthresh=1e-3),
                  'stability': PiecewiseNormalize([0, 1], [0, 1]),
                  'diff_JS': None, 'diff_pval': None, 'diff_filling': None,
                  'diff_number': None, 'diff_stability': None,
                  'R': PiecewiseLogNorm([-1e3, 0, 1e3], [0, 0.5, 1], linthresh=1),
                  'JS_mean': PiecewiseNormalize([0, 0.3, 1], [0, 0.5, 1]),
                  'variation_mean': PiecewiseLogNorm([0.1, 1, 10], [0, 0.5, 1]),
                  'variationnorm_mean': PiecewiseLogNorm([0.1, 1, 10], [0, 0.5, 1]),
                  None: None, 'JS_stab': PiecewiseNormalize([0, 0.05, 1], [0, 0.5, 1])
                  }


def rescale_ra(input_ra):
    ra = input_ra[input_ra > 0]
    if len(ra) == 0:
        return np.array([])

    ra /= 10 ** np.median(np.log10(ra))
    return ra


def fit_heavytail(input_ra, func='lognorm', discrete=False):
    ra = input_ra.copy()
    if discrete:
        ra = ra[ra > 0]
        ra += np.random.uniform(0, 1, ra.shape)  # otherwise big jumps for KS test
    ra = rescale_ra(ra)

    if func == 'lognorm':
        if np.any(np.isnan(ra)) or np.all(ra == 0):
            return (np.nan,) * 5
        s, loc, scale = stats.lognorm.fit(ra, 1, floc=0, fscale=1)
        stat, pval = stats.kstest(ra, 'lognorm', args=((s, loc, scale)))

        params = (s, loc, scale, stat, pval)
    elif func == 'norm':
        if np.any(np.isnan(ra)) or np.all(ra == 0):
            return (np.nan,) * 4
        loc, scale = stats.norm.fit(ra, loc=1, scale=1)
        stat, pval = stats.kstest(ra, 'norm', args=((loc, scale)))

        params = (loc, scale, stat, pval)
    elif func == 'expon':
        if np.any(np.isnan(ra)) or np.all(ra == 0):
            return (np.nan,) * 4
        loc, scale = stats.expon.fit(ra)
        stat, pval = stats.kstest(ra, 'expon', args=((loc, scale)))

        params = (loc, scale, stat, pval)
    elif func == 'powerlaw':
        if np.any(np.isnan(ra)) or np.all(ra == 0):
            return (np.nan,) * 5
        a, loc, scale = stats.powerlaw.fit(ra, 0.1, floc=0)
        stat, pval = stats.kstest(ra, 'powerlaw', args=((a, loc, scale)))

        params = (a, loc, scale, stat, pval)
    elif func == 'trunc_powerlaw':
        if np.any(np.isnan(ra)) or np.all(ra == 0):
            return (np.nan,) * 6
        results = powerlaw.Fit(ra, xmin=np.min(ra), xmax=np.max(ra))
        alpha = - results.power_law.alpha

        def cdf(x):
            rescale = (alpha + 1) * 1 / (np.max(ra) ** (alpha + 1) - np.min(ra) ** (alpha + 1))
            y = rescale * (x ** (alpha + 1) - np.min(ra) ** (alpha + 1)) / (alpha + 1)
            y[x < np.min(ra)] = 0
            y[x > np.max(ra)] = 1
            return y

        stat, pval = stats.kstest(ra, cdf)

        # noinspection PyTypeChecker,PyTypeChecker
        params = (-results.power_law.alpha, results.power_law.sigma) + \
                 results.distribution_compare('power_law', 'lognormal') + \
                 (stat, pval)
    elif func == 'pareto':
        if np.any(np.isnan(ra)) or np.all(ra == 0):
            return (np.nan,) * 5
        b, loc, scale = stats.pareto.fit(ra, 0.1, loc=0)
        stat, pval = stats.kstest(ra, 'pareto', args=((b, loc, scale)))

        params = (b, loc, scale, stat, pval)
    else:
        raise ValueError("Function %s is not defined." % func)

    return params


def plot_heavytail(input_ra, params, ax=None, func='lognorm', add_label=True,
                   xscale='log', yscale='log', discrete=False, **args):
    ra = input_ra.copy()
    if not discrete:
        ra = rescale_ra(ra)
    else:
        ra = ra[ra > 0]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if xscale == 'log':
        bins = np.logspace(np.log10(np.min(ra)) - 1, np.log10(np.max(ra)) + 1, 21)
        x_fit = np.logspace(np.log10(np.min(ra)) - 1, np.log10(np.max(ra)) + 1, 500)
    elif xscale == 'linear':
        bins = np.linspace(1.1 * np.min(ra) - 0.1 * np.max(ra), 1.1 * np.max(ra) - 0.1 * np.min(ra), 21)
        x_fit = np.linspace(1.1 * np.min(ra) - 0.1 * np.max(ra), 1.1 * np.max(ra) - 0.1 * np.min(ra), 500)
    else:
        raise ValueError('Scale of x-axis not known: %s' % xscale)

    hist = ax.hist(ra, alpha=0.2, density=True, bins=bins, color='grey')

    bin_heights = hist[0]

    cmap = plt.cm.get_cmap('coolwarm')
    norm = normalizations['pval']

    if func == 'lognorm':
        s, loc, scale, stat, pval = params

        pdf_fitted = stats.lognorm.pdf(x_fit, s, loc, scale)
        label = 's = %.2f' % s
        c = cmap(norm(pval))
    elif func == 'norm':
        loc, scale, stat, pval = params

        pdf_fitted = stats.norm.pdf(x_fit, loc, scale)
        label = r'$\sigma$' + ' = %.2f' % scale
        c = cmap(norm(pval))
    elif func == 'expon':
        loc, scale, stat, pval = params

        pdf_fitted = stats.expon.pdf(x_fit, loc, scale)
        label = 'scale = %.2f' % scale
        c = cmap(norm(pval))
    elif func == 'powerlaw':
        a, loc, scale, stat, pval = params
        pdf_fitted = stats.powerlaw.pdf(x_fit, a, loc, scale)
        label = r'$\alpha$' + ' = %.2f' % a
        c = 'gray'
    elif func == 'trunc_powerlaw':
        alpha, sigma, R, p, stat, pval = params
        rescale = (alpha + 1) * 1 / (np.max(ra) ** (alpha + 1) - np.min(ra) ** (alpha + 1))
        pdf_fitted = rescale * x_fit ** alpha
        pdf_fitted[x_fit < np.min(ra)] = 0
        pdf_fitted[x_fit > np.max(ra)] = 0
        label = r'$\alpha$' + ' = %.2f' % alpha
        c = cmap(norm(pval))
    elif func == 'pareto':
        b, loc, scale, stat, pval = params
        pdf_fitted = stats.pareto.pdf(x_fit, b, loc, scale)

        label = r'$\alpha$' + ' = %.2f' % (-b - 1)
        c = cmap(norm(pval))
    else:
        raise ValueError("Function %s is not defined." % func)

    ax.plot(x_fit, pdf_fitted, label=label if add_label == True else None, color=c, **args)
    ax.set_xscale('log')
    ax.legend()

    if xscale == 'log':
        ax.set_ylim(np.nanmin(bin_heights[bin_heights > 0]) / 100,
                    np.nanmax(bin_heights[bin_heights > 0]) * 100)
    elif xscale == 'linear':
        ax.set_ylim(1.1 * np.nanmin(bin_heights[bin_heights > 0]) - 0.1 * np.nanmax(bin_heights[bin_heights > 0]),
                    1.1 * np.nanmax(bin_heights[bin_heights > 0]) - 0.1 * np.nanmin(bin_heights[bin_heights > 0]))

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def plot_cdf_heavytail(input_ra, params, ax=None, func='lognorm', xscale='log', discrete=False, **args):
    ra = input_ra.copy()
    if not discrete:
        ra = rescale_ra(ra)
    else:
        ra = ra[ra > 0]
        # ra += np.random.uniform(0,1,ra.shape)

    x_fit = np.logspace(np.log10(np.min(ra) / 10), np.log10(np.max(ra) * 10), 200)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if xscale == 'log':
        x_fit = np.logspace(np.log10(np.min(ra)) - 1, np.log10(np.max(ra)) + 1, 500)
    elif xscale == 'linear':
        x_fit = np.linspace(1.1 * np.min(ra) - 0.1 * np.max(ra), 1.1 * np.max(ra) - 0.1 * np.min(ra), 500)

    cmap = plt.cm.get_cmap('coolwarm')
    norm = normalizations['pval']

    ax.plot(np.sort(ra), np.arange(1, len(ra) + 1) / len(ra), color='k')

    if func == 'lognorm':
        s, loc, scale, stat, pval = params

        cdf_fitted = stats.lognorm.cdf(x_fit, s, loc, scale)
        c = cmap(norm(pval))
    elif func == 'norm':
        loc, scale, stat, pval = params

        cdf_fitted = stats.norm.cdf(x_fit, loc, scale)
        c = cmap(norm(pval))
    elif func == 'expon':
        loc, scale, stat, pval = params

        cdf_fitted = stats.expon.cdf(x_fit, loc, scale)
        c = cmap(norm(pval))
    elif func == 'powerlaw':
        a, loc, scale, stat, pval = params
        cdf_fitted = stats.powerlaw.cdf(x_fit, a, loc, scale)
        c = 'gray'
    elif func == 'trunc_powerlaw':
        alpha, sigma, R, p, stat, pval = params
        rescale = (alpha + 1) * 1 / (np.max(ra) ** (alpha + 1) - np.min(ra) ** (alpha + 1))
        cdf_fitted = rescale * (x_fit ** (alpha + 1) - np.min(ra) ** (alpha + 1)) / (alpha + 1)
        cdf_fitted[x_fit < np.min(ra)] = 0
        cdf_fitted[x_fit > np.max(ra)] = 1
        c = cmap(norm(pval))
    elif func == 'pareto':
        b, loc, scale, stat, pval = params
        cdf_fitted = stats.pareto.cdf(x_fit, b, loc, scale)
        c = cmap(norm(pval))
    else:
        raise ValueError("Function %s is not defined." % func)

    ax.plot(x_fit, cdf_fitted, color=c, **args)
    ax.set_xscale('log')
    ax.legend()

    ax.set_xscale(xscale)
