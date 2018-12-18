import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import logging


class Optimization:

    def __init__(self, optimizer, obj_func):
        self.optimizer = optimizer
        self.obj_func = obj_func
        self.instances = []
        self.count = 0
        self.gaussian_mu = 0
        self.gaussian_sigma = 0

    def run_one_step(self):
        ins = self.optimizer.run_one_step(obj_fct=self.obj_func)
        self.instances.append(ins)
        self.count += 1
        self.fit_gaussian()

    @property
    def best_so_far(self, size=3):
        return self.instances.sort(key=lambda x: x.get_fitness(), reverse=True)[:size]

    @property
    def best_instance(self):
        if len(self.best_so_far) == 0:
            return None

        return self.best_so_far[0]

    def fit_gaussian(self):
        self.gaussian_mu, self.gaussian_sigma = norm.fit(self.best_so_far)


def _ucb_value(optimization, sample_time):
    ei = _expectation_improvement(optimization.best_instance.get_fitness(), optimization.gaussian_mu,
                                  optimization.gaussian_sigma)
    return ei + np.sqrt(2 * np.log(sample_time) / optimization.count)


def _expectation_improvement(best_so_far_value, mu, sigma):
    return quad(lambda x: x * norm.pdf(x, loc=mu, scale=sigma), best_so_far_value, np.inf)[0]


def bandit_selection(optimizations, sample_budget, initial_steps=5):
    # initialize each process, use dictionary to record each optimizer's result
    for optimization in optimizations:
        # do initialization
        for i in range(initial_steps):
            optimization.run_one_step()

    # update distribution
    for i in range(sample_budget):
        # get all ucb values and get next selection according to it
        ucb_values = [_ucb_value(o, i + 1) for o in optimizations]
        next_selection = optimizations[np.argmax(ucb_values)]

        # run one step in selected model, run_one_step() function will do the rest step automatically
        next_selection.run_one_step()

    # find best instance and return it
    best_optimization = sorted(optimizations, key=lambda x: x.best_instance.get_fitness())[-1]
    return best_optimization.best_instance


def test_bandit_selection():
    pass
