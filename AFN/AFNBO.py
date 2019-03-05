from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec


def posterior(bayes_optimizer, x_obs, y_obs, grid):
    bayes_optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = bayes_optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(bayes_optimizer, xrange,
            y=None,
            title=None,
            plot_target=True,
            plot_observations=True,
            plot_prediction=True,
            plot_ci=True,
            plot_utility=True,
            plot_next_best=True):

    x = np.linspace(xrange[0], xrange[1], 10000).reshape(-1, 1)

    fig = plt.figure(figsize=(16, 10))
    steps = len(bayes_optimizer.space)
    fig.suptitle(
        '{} After {} Iterations'.format(title or "Bayesian Optimization", steps),
        fontsize=30,
    )

    if plot_utility:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    else:
        gs = gridspec.GridSpec(1, 1, height_ratios=[1])

    axis = plt.subplot(gs[0])

    x_obs = np.array([[res["params"]["x"]] for res in bayes_optimizer.res])
    y_obs = np.array([res["target"] for res in bayes_optimizer.res])

    mu, sigma = posterior(bayes_optimizer, x_obs, y_obs, x)

    if y is not None and plot_target:
        axis.plot(x, y, linewidth=3, label='Target')

    if plot_observations:
        axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')

    if plot_prediction:
        axis.plot(x, mu, '--', color='k', label='Prediction')

    if plot_ci:
        axis.fill(np.concatenate([x, x[::-1]]),
                  np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
                  alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim(xrange)
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})
    axis.legend(loc=0, borderaxespad=0.6)

    if plot_utility:
        acq = plt.subplot(gs[1])
        utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
        utility = utility_function.utility(x, bayes_optimizer._gp, 0)
        acq.plot(x, utility, label='Utility Function', color='purple')

        if plot_next_best:
            acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
                     label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

        acq.set_xlim(xrange)
        acq.set_ylim((0, np.max(utility) + 0.5))
        acq.set_ylabel('Utility', fontsize=20)
        acq.set_xlabel('x', fontsize=20)
        acq.legend(loc=0, borderaxespad=0.6)

    plt.show()
