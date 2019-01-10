import numpy as np
import pandas as pd

from framework.random_search import random_search

EVALUATION_CRITERIA = 'Accuracy'


class RandomOptimization:

    def __init__(self, model_generator, name=None):
        self.model_generator = model_generator
        self.name = name
        self.count = 0
        self.time_out_count = 0

        # Evaluation results
        self.instances = pd.DataFrame()

        # Gaussian parameters
        self.mu = 0
        self.sigma = 0

        # Mean of the sum of the squares of evaluation result
        self.square_mean = 0

    def __str__(self):
        return 'Model {}\nBudget: {}\nTimeout count: {}\nBest result: {}\n' \
               'Gaussian mu: {}\nGaussian sigma: {}\nmu_Y: {}'.format(self.name, self.count, self.time_out_count,
                                                                      self.best_evaluation, self.mu, self.sigma,
                                                                      self.square_mean)

    @property
    def best_evaluation(self):
        return self.instances.sort_values(by=EVALUATION_CRITERIA, ascending=False).iloc[0]

    def run_one_step(self, train_x, train_y):
        evaluation_result = random_search(self.model_generator, train_x, train_y, search_times=1)

        while evaluation_result.empty:
            # The result is empty because some errors like timeout occurred
            self.time_out_count += 1
            evaluation_result = random_search(self.model_generator, train_x, train_y, search_times=1)

        self.instances = self.instances.append(evaluation_result, ignore_index=True)

        # update count
        previous_count = self.count
        self.count += 1

        self._update_parameter(previous_count, evaluation_result[EVALUATION_CRITERIA])

    def _update_parameter(self, previous_count, new_eval_result):
        self.mu = (previous_count * self.mu + new_eval_result) / (previous_count + 1)
        self.sigma = self.instances[EVALUATION_CRITERIA].var()
        self.square_mean = (previous_count * self.square_mean + new_eval_result ** 2) / (previous_count + 1)


def _new_func(optimization, t):
    return _ucb_func(optimization, t) + np.sqrt(optimization.square_mean + ucb_item)


def _ucb_func(optimization, t):
    return optimization.mu + np.sqrt(2 * np.log(t - 1) / optimization.count)


class BanditModelSelection:
    _update_functions = ['new', 'ucb']

    def __init__(self, optimizations, update_func='new'):
        self.optimizations = optimizations
        self.update_func = self._get_update_function(update_func)

    def fit(self, train_x, train_y, budget=200):
        self._init_each_optimizations(train_x, train_y)

        for t in range(len(self.optimizations) + 1, budget + 1):
            next_model = self._next_selection(t)
            next_model.run_one_step(train_x, train_y)

        return self._best_selection()

    def show_models(self):
        models_info = ''
        for optimization in self.optimizations:
            models_info += str(optimization)
            models_info += '\n\n'

    def _best_selection(self):
        best_results = [r.best_evaluation[EVALUATION_CRITERIA] for r in self.optimizations]
        best_index = np.argmax(best_results)

        return self.optimizations[best_index]

    def _init_each_optimizations(self, train_x, train_y):
        for optimization in self.optimizations:
            optimization.run_one_step(train_x, train_y)

    def _next_selection(self, current_count):
        values = [self.update_func(o, current_count) for o in self.optimizations]
        return self.optimizations[np.argmax(values)]

    def _get_update_function(self, update_func_name):
        if update_func_name == 'new':
            return _new_func
        elif update_func_name == 'ucb':
            return _ucb_func
        else:
            raise ValueError("Unknown update function: {}".format(self.update_func))
