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

        # Parameter change record
        self.param_change_info = []

    def __str__(self):
        return 'Model {}\nBudget: {}\nTimeout count: {}\n======Best result======:\n {}\n============\n' \
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

        eval_value = evaluation_result[EVALUATION_CRITERIA].values[0]
        self._update_parameter(previous_count, eval_value)

    def clear(self):
        self.count = 0
        self.time_out_count = 0

        self.instances = pd.DataFrame()

        self.mu = 0
        self.sigma = 0

        self.square_mean = 0

        self.param_change_info = []

    def _update_parameter(self, previous_count, new_eval_result):
        self.mu = (previous_count * self.mu + new_eval_result) / (previous_count + 1)
        self.sigma = self.instances[EVALUATION_CRITERIA].std()
        self.square_mean = (previous_count * self.square_mean + new_eval_result ** 2) / (previous_count + 1)


def _new_func(optimization, t, theta=1, record=None):
    third_term = np.sqrt(2 * np.log(t - 1) / optimization.count)
    forth_term = (1 / theta) * third_term
    sqrt_mu_y = np.sqrt(optimization.square_mean)
    result = optimization.mu + (1 / theta) * sqrt_mu_y + third_term + forth_term

    if record is not None:
        assert isinstance(record, list)
        record.append((optimization.name, optimization.mu, sqrt_mu_y,
                       third_term, forth_term, third_term + forth_term, result))

    return result


def _ucb_func(optimization, t, record=None):
    second_term = np.sqrt(2 * np.log(t - 1) / optimization.count)
    result = optimization.mu + second_term

    if record is not None:
        assert isinstance(record, list)
        record.append((optimization.name, optimization.mu, second_term, result))

    return result


class BanditModelSelection:
    _update_functions = ['new', 'ucb']

    def __init__(self, optimizations, update_func='new', theta=1):
        self.optimizations = optimizations
        self.update_func = self._get_update_function(update_func)
        self.param_change_info = []
        self.theta = theta

    def fit(self, train_x, train_y, budget=200):
        self._clean()  # clean history data
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

        return models_info

    def statistics(self):
        data = [(o.name, o.mu, o.sigma, o.square_mean, o.count, o.best_evaluation[EVALUATION_CRITERIA])
                for o in self.optimizations]
        return pd.DataFrame(data=data, columns=['name', 'mu', 'sigma', 'mu_Y', 'budget', 'best X'])

    def _wrap_selection_information(self, data):
        if self.update_func is _new_func:
            return pd.DataFrame(data=data, columns=['name', 'mu', 'sqrt(mu_Y)', 'third term',
                                                    'forth term', 'sum of last two', 'sum all'])
        elif self.update_func is _ucb_func:
            return pd.DataFrame(data=data, columns=['name', 'mu', 'second_term', 'sum all'])

    def _best_selection(self):
        best_results = [r.best_evaluation[EVALUATION_CRITERIA] for r in self.optimizations]
        best_index = np.argmax(best_results)

        return self.optimizations[best_index]

    def _init_each_optimizations(self, train_x, train_y):
        for optimization in self.optimizations:
            optimization.clear()  # clear history data
            optimization.run_one_step(train_x, train_y)

    def _next_selection(self, current_count):
        selection_record = []  # used to record values of the terms of the equation for each models
        if self.update_func is _new_func:
            values = [self.update_func(o, current_count, theta=self.theta, record=selection_record)
                      for o in self.optimizations]
        else:
            values = [self.update_func(o, current_count, selection_record) for o in self.optimizations]

        self.param_change_info.append(self._wrap_selection_information(selection_record))
        return self.optimizations[np.argmax(values)]

    def _clean(self):
        self.param_change_info = []

    def _get_update_function(self, update_func_name):
        if update_func_name == 'new':
            return _new_func
        elif update_func_name == 'ucb':
            return _ucb_func
        else:
            raise ValueError("Unknown update function: {}".format(self.update_func))
