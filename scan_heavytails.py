num_cores = 12 # Number of processors used for multiprocessing

import multiprocessing
import warnings

import numpy as np
from scipy import stats
import itertools
import functools

import sys

sys.path.insert(1, '../sglv_timeseries')

import sglv_timeseries.glv.Timeseries as glv
import sglv_timeseries.ibm.Timeseries as ibm
from sglv_timeseries.noise_parameters import NOISE
from sglv_timeseries.models import MODEL

from heavytails import fit_heavytail

from variation import variation_coefficient, JS

debug = False

noise_implementation = NOISE.LANGEVIN_LINEAR  # CONSTANT

if debug:
    import matplotlib.pyplot as plt
    from sglv_timeseries.timeseries_plotting import PlotTimeseries
    from heavytails import plot_heavytail


def random_parameter_set(S, connectance=0.3, minint=-0.5, maxint=0.5,
                         minmigration=0.4, maxmigration=0.4,
                         minextinction=0.5, maxextinction=0.5, growth_rate=1.5):
    """ Return a set of random parameters to generate glv time series """

    # Interaction matrix
    interaction = np.random.uniform(minint, maxint, [S, S])

    # Impose connectance: set interaction matrix elements to zero such that the percentage of non-zero elements
    # is equal to the connectance
    interaction *= np.random.choice([0, 1], interaction.shape, p=[1 - connectance, connectance])

    # Self-interaction is -1 for all species.
    np.fill_diagonal(interaction, -1.)

    # Growth rate is equal for all species (value is growth_rate).
    growth_rate = np.full([S, 1], growth_rate)

    # Uniform immigration and extinction rates.
    immigration = np.random.uniform(minmigration, maxmigration, [S, 1])
    extinction = np.random.uniform(minextinction, maxextinction, [S, 1])

    return {'interaction_matrix': interaction, 'immigration_rate': immigration,
              'extinction_rate': extinction, 'growth_rate': growth_rate}


def random_parameter_set_ibm(S, connectance=0.3, minint=-0.5, maxint=0.5,
                             minmigration=0.4, maxmigration=0.4,
                             minextinction=0.5, maxextinction=0.5, growth_rate=1.5, SIS=[], SISfactor=200):

    params = random_parameter_set(S, connectance, minint, maxint,
                         minmigration, maxmigration, minextinction, maxextinction, growth_rate)

    # Generate strongly-interacting-species (SIS) vector.
    SISvector = np.ones(S, dtype=int)
    SISvector[SIS] *= SISfactor

    params['SISvector'] = SISvector

    return params


def random_parameter_set_logistic(S, width_growth=1):
    # Set growth rates.
    if width_growth == 0:
        growth_rate = np.ones([S, 1])
    else:
        growth_rate = stats.lognorm.rvs(loc=0, s=width_growth, size=[S, 1])

    # No interactions becauce logistic model
    interaction = np.zeros([S, S])

    # Calculate and set self-interactions.
    if width_growth == 2:
        self_int = np.ones(S)
    else:
        self_int = stats.lognorm.rvs(loc=0, s=np.sqrt(4 - width_growth ** 2), size=S)
    np.fill_diagonal(interaction, -self_int)

    # No immigration or extinction.
    immigration = np.zeros([S, 1])
    extinction = np.zeros([S, 1])

    return {'interaction_matrix': interaction, 'immigration_rate': immigration,
              'extinction_rate': extinction, 'growth_rate': growth_rate}


def add_SIS(interaction, SISvector):
    interaction_SIS = interaction * SISvector
    np.fill_diagonal(interaction_SIS, np.diag(interaction))
    return interaction_SIS


def line_statistics(params, model=MODEL.GLV):
    """ Generates a time series with the given parameters and returns a string with all statistical parameters
    of this time series."""

    # Initiate empty line
    line = ''

    # First simulate without noise to allow system to go to steady state
    params_nonoise = params.copy()  # parameters without noise
    for noise in ['noise_linear', 'noise_constant']:
        if noise in params_nonoise:
            params_nonoise[noise] = 0

    if model in [MODEL.GLV, MODEL.MAX, MODEL.MAX_IMMI]:
        discrete = False

        # Find steady state without noise
        ts = glv.Timeseries(params_nonoise, T=250, dt=0.01, tskip=99, model=model)

        if debug:
            PlotTimeseries(ts.timeseries)

        # Determine deterministic stability: stable if less than 10% change for last 50 time points.
        deterministic_stability = (np.max(np.abs((ts.timeseries.iloc[-50, 1:] - ts.timeseries.iloc[-1, 1:]) / ts.timeseries.iloc[-50,
                                                                                           1:])) < 0.1)
        line += ',%d' % deterministic_stability

        # Find steady state with noise
        # Set steady state to deterministic steady state
        params['initial_condition'] = ts.endpoint.values.astype('float')

        ts = glv.Timeseries(params, T=500, dt=0.01, tskip=99, model=model, noise_implementation=noise_implementation)

        if debug:
            PlotTimeseries(ts.timeseries)
    elif model == MODEL.IBM:
        discrete = True

        # Time series to find "steady state", transient dynamics
        params['initial_condition'] = ibm.Timeseries(params, T=50).endpoint.values.astype('int').flatten()

        # Time series for IBM.
        ts = ibm.Timeseries(params, T=250)
    else:
        raise ValueError("Unknown model: %s" % model.__name__)

    endpoint = ts.endpoint

    # Remove species that are "extinct", by definition: smaller than 6 orders of magnitude smaller than maximal abundance
    col_to_drop = endpoint.index[endpoint.endpoint < 1e-6 * np.max(endpoint.endpoint)]

    with warnings.catch_warnings(): # Ignore the NAN-warnings when removing species.
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        endpoint = endpoint.values.astype('float').flatten()
        endpoint = endpoint[endpoint > 1e-6 * np.nanmax(endpoint)]

    if model in [MODEL.GLV, MODEL.MAX, MODEL.MAX_IMMI]:
        ts_trimmed = ts.timeseries.drop(columns=col_to_drop)
    elif model == MODEL.IBM:
        ts_trimmed = ts.timeseries
    else:
        raise ValueError("Unknown model: %s" % model.__name__)

    # Diversity is number of remaining species.
    diversity = len(endpoint)
    line += ',%d' % diversity

    # Normalized time series.
    ts_norm = ts.timeseries.div(
        ts.timeseries.loc[:, [col for col in ts.timeseries.columns if col.startswith('species')]].sum(axis=1), axis=0)
    ts_norm.time = ts.timeseries.time

    # Calculate variation coefficient for time series, and normalized time series.
    for tsi in [ts_trimmed, ts_norm]:
        params = variation_coefficient(tsi)
        for par in params:
            line += ',%.3E' % par

    # Calculate Jensen Shannon distance.
    params_JS = JS(ts_trimmed)
    for par in params_JS:
        line += ',%.3E' % par

    # Calculate parameters for fitting heavy tailed distributions.
    for func in ['lognorm', 'pareto', 'powerlaw', 'trunc_powerlaw', 'expon', 'norm']:
        params = fit_heavytail(endpoint, func=func, discrete=discrete)

        for par in params:
            line += ',%.3E' % par

    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        params = fit_heavytail(endpoint, func='lognorm', discrete=discrete)
        plot_heavytail(endpoint, params, func='lognorm', ax=ax, discrete=discrete)
        print("Width lognorm:", params[0])
        print("Stat lognorm:", params[-2])
        for f in ['expon', 'norm', 'powerlaw', 'pareto']:
            params = fit_heavytail(endpoint, func=f, discrete=discrete)
            plot_heavytail(endpoint, params, func=f, ax=ax, discrete=discrete)
            print("Stat %s:" % f, params[-2])
        params = fit_heavytail(endpoint, func='trunc_powerlaw', discrete=discrete)
        plot_heavytail(endpoint, params, func='trunc_powerlaw', ax=ax, discrete=discrete)
        print("Stat trunc powerlaw:", params[-2])
        print("R powerlaw (negative -> lognormal):", params[2])
        plt.show()
    return line


def initial_condition(S_, model, max_cap, absent_init):
    initcond = np.random.uniform(0, 1, [S_, 1])

    if 'MAX' in model.name:
        # Rescale initial condition with maximum capacity.
        initcond *= min(1., 1. * max_cap / S_)

    if absent_init:
        # Set random species to zero, they may enter the system through immigration.
        initcond *= np.random.choice([0, 1], size=initcond.shape, p=[0.2, 0.8])

    return initcond


def one_set_glv(input_pars, file='', N=1, S=None, model=MODEL.GLV, absent_init=False, use_lognormal_params=False):
    """ Generates N time series of glv systems according to input parameters and writes summary
    of the statistics of time series to file."""

    connectance, immigration, noise, int_strength, max_cap = input_pars

    if 'MAX' in model.name and np.isinf(max_cap):
        model = MODEL.GLV

    # Reduce number of species (S_) until more than half of the solutions are good (not all 0 or NaN abundances).
    S_ = S
    Ngood_solutions = 0

    while Ngood_solutions < N / 2 and S_ > 1:
        Ngood_solutions = 0

        line_stat = ''

        for k in range(N):
            # Set parameters.
            params = random_parameter_set(S=S_,
                                          minmigration=immigration, maxmigration=immigration, connectance=connectance,
                                          minint=-int_strength, maxint=int_strength)

            # Maximum capacity parameter.
            if 'MAX' in model.name:
                params['maximum_capacity'] = max_cap

            # Noise parameters.
            if noise_implementation == NOISE.LANGEVIN_LINEAR:
                params['noise_linear'] = noise
            elif noise_implementation == NOISE.LANGEVIN_CONSTANT:
                params['noise_constant'] = noise

            if use_lognormal_params:
                # Set growth rate and self-interaction parameters to lognormally distributed parameters.
                np.fill_diagonal(params['interaction_matrix'], -stats.lognorm.rvs(loc=0, s=1, size=S_))
                params['growth_rate'] = stats.lognorm.rvs(loc=0, s=1, size=[S_, 1])

            # Set initial condition
            params['initial_condition'] = initial_condition(S_, model, max_cap, absent_init)

            line_stat += line_statistics(params, model)

            # Check whether solution is good (not all abundances 0 or NAN)
            if np.any([number not in ['NAN', '0', ''] for number in line_stat.split(',')]):
               Ngood_solutions += 1

        # Reduce S_ for next iteration if there were not enough 'good' solutions
        S_ = int(0.95 * S_) if S_ > 10 else (S_ - 1)

    # Write results to file.
    line = '%.3E,%.3E,%.3E,%.3E,%3E' % input_pars + line_stat + '\n'
    with open(file, 'a') as f:
        f.write(line)


def one_set_logistic(input_pars, file='', N=10, S=None):
    """ Generates N time series of logistic systems according to input parameters and writes summary
    of the statistics of time series to file."""

    width_growth, noise = input_pars

    line_stat = ''

    for k in range(N):
        # Set parameters.
        params = random_parameter_set_logistic(S=S, width_growth=width_growth)

        # Set initial condition.
        params['initial_condition'] = np.random.uniform(0, 1, [S, 1])

        # Set noise paramters.
        if noise_implementation == NOISE.LANGEVIN_CONSTANT:
            params['noise_constant'] = noise
        elif noise_implementation == NOISE.LANGEVIN_LINEAR:
            params['noise_linear'] = noise

        line_stat += line_statistics(params, model=MODEL.GLV)

    # Write data to file.
    line = '%.3E,%.3E' % input_pars + line_stat + '\n'
    with open(file, 'a') as f:
        f.write(line)


def one_set_ibm(input_pars, file='', N=10, S=None):
    """ Generates N time series of IBM according to input parameters and writes summary
        of the statistics of time series to file."""

    connectance, immigration, int_strength, sites = input_pars

    line_stat = ''
    for k in range(N):
        # Set parameters.
        params = random_parameter_set_ibm(S=S, minmigration=immigration, maxmigration=immigration,
                                          connectance=connectance,
                                          minint=-int_strength, maxint=int_strength)

        # Set initial condition, assert it does not exceed the number of sites.
        initcond = np.random.randint(0, int(0.66 * sites / S), S)
        assert np.sum(initcond) <= sites
        params['initial_condition'] = initcond

        params['sites'] = sites

        # Generate time series and its statistics.
        line_stat += line_statistics(params, model=MODEL.IBM)

    # Write data to file.
    line = '%.3E,%.3E,%.3E,%.3E' % input_pars + line_stat + '\n'
    with open(file, 'a') as f:
        f.write(line)

def header_time_series(N, ibm=False):
    """ Return string for header of statistical parameters of N time series."""

    line = ""

    # Statistical parameters for one time series.
    subline = 'number_%d,' \
              'variation_mean_%d,variation_std_%d,variation_min_%d,variation_max_%d,' \
              'variationnorm_mean_%d,variationnorm_std_%d,variationnorm_min_%d,variationnorm_max_%d,' \
              'JS_mean_%d,JS_std_%d,JS_min_%d,JS_max_%d,JS_stab_%d,' \
              'log_width_%d,log_loc_%d,log_scale_%d,log_stat_%d,log_pval_%d,' \
              'pareto_a_%d,pareto_loc_%d,pareto_scale_%d,pareto_stat_%d,pareto_pval_%d,' \
              'pow_a_%d,pow_loc_%d,pow_scale_%d,pow_stat_%d,pow_pval_%d,' \
              'tpow_a_%d,tpow_scale_%d,tpow_R_%d,tpow_p_%d,tpow_stat_%d,tpow_pval_%d,' \
              'exp_loc_%d,exp_scale_%d,exp_stat_%d,exp_pval_%d,' \
              'norm_loc_%d,norm_scale_%d,norm_stat_%d,norm_pval_%d'
    Npars = 43 # number of paramters in line

    # Add stability paramter if not for ibm.
    if not ibm:
        subline = ',stability_%d,' + subline
        Npars += 1

    # Add statistical parameters for number of time series N.
    for i in range(1, N + 1):
        line += subline % ((i,) * Npars)
    return line

def setup_glv(file, N=10):
    """ Adds header line for gLV time series."""

    # Fixed parameters for line.
    line = 'connectance,immigration,noise,interaction,max_cap' + header_time_series(N) + '\n'

    # Write header to file.
    with open(file, 'w') as f:
        f.write(line)

def setup_logistic(file, N=10):
    """ Adds header line for logistic time series."""

    # Fixed parameters for line.
    line = 'width_growth,noise' + header_time_series(N) + '\n'

    # Write header to file.
    with open(file, 'w') as f:
        f.write(line)

def setup_ibm(file, N=10):
    """ Adds header line for ibm time series."""

    # Fixed parameters for line.
    line = 'connectance,immigration,interaction,sites' + header_time_series(N, ibm=True) + '\n'

    # Write header to file.
    with open(file, 'w') as f:
        f.write(line)

def scan_ibm(file, N):
    setup_ibm(file, N)

    S = 200.

    sitess = [500, 1000, 5000, 10000, 50000]
    connectances = np.linspace(0, 1, 5)
    immigrations = np.hstack(([0], np.logspace(-2, 0, 5)))
    interaction_strengths = np.linspace(0, 1, 5)

    combinations = itertools.product(connectances, immigrations, interaction_strengths, sitess)

    combinations = itertools.islice(combinations, 0, 250)
    combinations = itertools.islice(combinations, 250, 500)
    combinations = itertools.islice(combinations, 500, 751)

    one_set_ = functools.partial(one_set_ibm, file=file, N=N, S=S)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_glv_interactions(file, N):
    setup_glv(file, N)

    S = 200

    connectances = np.linspace(0, 1, 11)
    interaction_strengths = np.linspace(0, 1, 11)
    immigrations = [0., 0.1]  # np.hstack(([0],np.logspace(-2,0,5))) #10)))
    noises = [0, 0.5]
    max_caps = [np.inf, 1e2, 1000, 200]  # 1e-2, 1e-1, 1e0, 5, 1e1, 5e1, 1e2, 5e2, 1e3]

    combinations = itertools.product(connectances, immigrations, noises, interaction_strengths, max_caps)

    model = MODEL.MAX_IMMI
    one_set_ = functools.partial(one_set_glv, file=file, N=N, S=S, model=model)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_glv_maxcap(file, N):
    setup_glv(file, N)

    S = 200

    connectances = np.linspace(0, 1, 11)
    interaction_strengths = [0.5]
    immigrations = [0.1]
    noises = [0, 0.5]
    max_caps = [np.inf, 1e2, 1000, 200, 500, 10, 50]

    model = MODEL.MAX_IMMI
    one_set_ = functools.partial(one_set_glv, file=file, N=N, S=S, model=model)

    combinations = itertools.product(connectances, immigrations, noises, interaction_strengths, max_caps)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)

    connectances = [0.]
    interaction_strengths = [0.]
    immigrations = [0.]
    noises = np.linspace(0, 1, 11)
    max_caps = [np.inf, 1e2, 5000, 1000, 200, 500, 10, 50]

    combinations = itertools.product(connectances, immigrations, noises, interaction_strengths, max_caps)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_glv_immigration(file, N):
    setup_glv(file, N)

    S = 200

    connectances = np.linspace(0, 1, 11)
    interaction_strengths = [0.5]
    immigrations = np.hstack(([0], np.logspace(-2, 1, 7)))  # 10)))
    noises = [0, 0.5]
    max_caps = [np.inf, 1e2, 1000, 200]  # , 1000, 200, 500, 10, 50]

    combinations = itertools.product(connectances, immigrations, noises, interaction_strengths, max_caps)

    model = MODEL.MAX_IMMI
    one_set_ = functools.partial(one_set_glv, file=file, N=N, S=S, model=model)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_logistic(file, N):
    setup_logistic(file, N)

    S = 200

    noises = np.linspace(0, 1, 11)  # np.hstack(([0], np.logspace(-3, 0, 13)))
    width_growth = [0, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]  # np.hstack(([0.], np.logspace(-3, np.log10(2), 10)))

    combinations = itertools.product(width_growth, noises)

    one_set_ = functools.partial(one_set_logistic, file=file, N=N, S=S)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_diversity(file, N):
    setup_glv(file, N)

    S = 500

    connectances = np.logspace(-3, 0, 31)
    interaction_strengths = [0.2, 0.5]
    immigrations = [0, 1e-4, 1e-1]  # np.hstack(([0],np.logspace(-2,1,7))) #10)))
    noises = [0.5]
    max_caps = [np.inf, 200, 1000]

    combinations = itertools.product(connectances, immigrations, noises, interaction_strengths, max_caps)

    model = MODEL.MAX_IMMI
    one_set_ = functools.partial(one_set_glv, file=file, N=N, S=S, model=model)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_diversity_ibm(file, N):
    setup_ibm(file, N)

    S = 500

    connectances = np.logspace(-3, 0, 31)
    interaction_strengths = [0.2, 0.5, 1.0]
    immigrations = [0, 1e-4, 1e-3, 1e-1]  # np.hstack(([0],np.logspace(-2,1,7))) #10)))
    sitess = [10000, 1000]

    combinations = itertools.product(connectances, immigrations, interaction_strengths, sitess)

    model = MODEL.MAX_IMMI
    one_set_ = functools.partial(one_set_ibm, file=file, N=N, S=S, model=model)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


def scan_diversity_lognormal(file, N):
    setup_glv(file, N)

    S = 500

    connectances = np.logspace(-3, 0, 31)
    interaction_strengths = [0.2, 0.5]
    immigrations = [0, 1e-4, 1e-1]
    noises = [0.5]
    max_caps = [np.inf, 200, 1000]

    combinations = itertools.product(connectances, immigrations, noises, interaction_strengths, max_caps)

    model = MODEL.MAX_IMMI
    one_set_ = functools.partial(one_set_glv, file=file, N=N, S=S, model=model, use_lognormal_params=True)

    p = multiprocessing.Pool(num_cores)
    p.map(one_set_, combinations)


# Test functions

def test_absent_species_initial_condition(file):
    connectance = 0.5
    immigration = 0.1
    int_strength = 0.5
    noise = 0.01
    max_cap = np.inf
    setup_glv(file, N=2)

    model = MODEL.MAX_IMMI

    one_set_glv((connectance, immigration, noise, int_strength, max_cap), file=file,
                N=2, S=200, model=model, absent_init=True)


def test_glv(file):
    connectance = 0.2
    immigration = 0.01
    int_strength = 0.1
    noise = 0.01
    max_cap = np.inf
    setup_glv(file, N=2)
    one_set_glv((connectance, immigration, noise, int_strength, max_cap), file=file, N=2, S=200, model=MODEL.MAX_IMMI)


def test_ibm(file):
    setup_ibm(file, N=1)

    S = 200
    sites = 1000
    connectance = 0.3
    immigration = 0.01
    interaction_strength = 0.1

    setup_ibm(file, N=1)
    one_set_ibm((connectance, immigration, interaction_strength, sites), file=file, N=1, S=S)


# Main function performs different tests

def main():
    # test_absent_species_initial_condition('test_absent_species_initial_condition.csv')
    test_glv('test_glv.csv')
    # test_ibm('test_ibm.csv')


if __name__ == "__main__":
    main()
