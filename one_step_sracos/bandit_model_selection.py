import logging

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

logger = logging.getLogger('bandit.process')


class Optimization:

    def __init__(self, optimizer, obj_func, name=None):
        self.optimizer = optimizer
        self.obj_func = obj_func
        self.instances = []
        self.count = 0
        self.gaussian_mu = 0
        self.gaussian_sigma = 0
        self.name = name

    def run_one_step(self):
        ins = self.optimizer.run_one_step(obj_fct=self.obj_func)
        self.instances.append(ins)
        self.count += 1

        logger.debug('{} - {}, times : {}'.format(self.name, str(ins), self.count))

    @property
    def best_so_far(self, size=3):
        self.instances.sort(key=lambda x: x.get_fitness(), reverse=True)
        return self.instances[:size]

    @property
    def best_instance(self):
        if len(self.best_so_far) == 0:
            return None

        return self.best_so_far[0]

    def fit_gaussian(self):
        self.gaussian_mu, self.gaussian_sigma = norm.fit(list(map(lambda x: x.get_fitness(), self.best_so_far)))
        logger.debug("mu is {}, sigma is {}".format(self.gaussian_mu, self.gaussian_sigma))


def _ucb_value(optimization, sample_time):
    # ei = _expectation_improvement(optimization.best_instance.get_fitness(), optimization.gaussian_mu,
    #                               optimization.gaussian_sigma)

    return optimization.gaussian_mu + np.sqrt(2 * np.log(sample_time) / optimization.count)


def _expectation_improvement(best_so_far_value, mu, sigma):
    return quad(lambda x: x * norm.pdf(x, loc=mu, scale=sigma), best_so_far_value, np.inf)[0]


def bandit_selection(optimizations, sample_budget, initial_steps=3):
    logger.debug("initialization")

    # initialize each process, use dictionary to record each optimizer's result
    for optimization in optimizations:
        # do initialization
        logger.debug("initialize {}".format(optimization.name))
        for i in range(initial_steps):
            optimization.run_one_step()

        optimization.fit_gaussian()

    logger.debug("initialization done")

    # update distribution
    logger.debug("begin bandit selection")
    for i in range(sample_budget):
        # get all ucb values and get next selection according to it
        ucb_values = [_ucb_value(o, i + 1) for o in optimizations]
        next_selection = optimizations[np.argmax(ucb_values)]

        logger.debug('next selection is {}'.format(next_selection.name))

        # run one step in selected model, run_one_step() function will do the rest step automatically
        next_selection.run_one_step()

    logger.debug('bandit selection done')
    # find best instance and return it
    best_optimization = sorted(optimizations, key=lambda x: x.best_instance.get_fitness())[-1]

    logger.info('result - model: {}, instance: {}'.format(best_optimization.name, str(best_optimization.best_instance)))
    return best_optimization.best_instance


def test_bandit_selection():
    pass
