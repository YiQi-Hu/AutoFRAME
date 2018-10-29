"""
Author: Yi-Qi Hu
this is a template for model evaluation
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import abc

_FLOAT_PARAM = 0
_INTEGER_PARAM = 1
_CATEGORICAL_PARAM = 2


def data_collector(index_list, features, labels):
    """
    re-collect data according to index
    :param index_list: the data index
    :param features: original features
    :param labels: original labels
    :return: the features and labels collected by index_list
    """

    feature_dim = features.shape[1]

    trans_features = np.zeros((len(index_list), feature_dim))
    trans_labels = np.zeros(len(index_list))

    for i in range(len(index_list)):
        trans_features[i, :] = features[index_list[i], :]
        trans_labels[i] = labels[index_list[i], :]

    return trans_features, trans_labels


class ModelEvaluator:

    def __init__(self, model_generator=None, train_x=None, train_y=None, criterion='accuracy', valid_k=5):
        """
        :param model_generator: an instantiation son class of ModelGenerator
        :param train_x: train feature, type -- array
        :param train_y: train label, type -- array
        :param criterion: evaluation metric, type -- string
        :param valid_k: k-validation, type -- int
        """
        self.model_generator = model_generator
        self.train_x = train_x
        self.train_y = train_y
        self.criterion = criterion
        self.validation_kf = StratifiedKFold(n_splits=valid_k, shuffle=False)
        return

    def evaluate(self, x):
        """
        evaluate the hyperparameter x by k-fold validation
        :param x: the hyperparameter list, type -- list
        :return: the evaluation value according to the metric, type -- float
        """

        this_model = self.model_generator.generate_model(x)

        eval_values = []
        for train_index, valid_index in self.validation_kf.split(self.train_x, self.train_y):
            x, y = data_collector(train_index, self.train_x, self.train_y)
            valid_x, valid_y = data_collector(valid_index, self.train_x, self.train_y)

            this_model = this_model.fit(x, y)

            predictions = this_model.predict(valid_x)

            if self.criterion == 'accuracy':
                eval_value = accuracy_score(valid_y, predictions)
            elif self.criterion == 'auc':
                eval_value = roc_auc_score(valid_y, predictions)
            else:
                eval_value = 0.0
            eval_values.append(eval_value)

        eval_mean = np.mean(np.array(eval_values))

        return eval_mean


class ModelGenerator:
    """
    This is the father class of each model implementation. Each specific model implementation should overwrite the two
    basic functions: set_hyper-parameter and generate_model.
    """

    def __init__(self, hp_space, model_initializer):
        self.hp_space = hp_space
        self._model_initializer = model_initializer

    @abc.abstractmethod
    def generate_model(self, param_values):
        return


class HyperParameter:

    def __init__(self, name, param_range, param_type):
        self.name = name
        self._param_range = param_range
        self.param_type = param_type

    @classmethod
    def int_param(cls, name, param_range):
        return cls(name, param_range, _INTEGER_PARAM)

    @classmethod
    def float_param(cls, name, param_range):
        return cls(name, param_range, _FLOAT_PARAM)

    @classmethod
    def categorical_param(cls, name, param_range):
        return cls(name, param_range, _CATEGORICAL_PARAM)

    @property
    def param_bound(self):
        """Get lower bound and upper bound for a parameter

        Returns
        -------
        lower_bound: int or float
            lower_bound is inclusive
        upper_bound: int or float
            if parameter is categorical, the upper_bound is exclusive,
            otherwise the upper_bound is inclusive
        """
        if self.param_type == _CATEGORICAL_PARAM:
            return 0, len(self._param_range)
        else:
            return self._param_range

    def in_range(self, value):
        """Test whether the parameter's value is in a legal range

        Parameters
        ---------
        value : str or int or float
            value of parameter

        Returns
        -------
        is_in_range: bool
            True if value is in range
        """
        if self.param_type == _CATEGORICAL_PARAM:
            return 0 <= int(value) < len(self._param_range)
        else:
            assert len(self._param_range) == 2
            return self._param_range[0] <= value <= self._param_range[1]

    def convert_raw_param(self, raw_param):
        """Cast raw parameter value to certain type

        Parameters
        ----------
        raw_param : str or int or float
            value which can be any type

        Returns
        -------
        param : str or int or float
            casted value
        """
        if self.param_type == _INTEGER_PARAM:
            return int(raw_param)
        elif self.param_type == _FLOAT_PARAM:
            return float(raw_param)
        elif self.param_type == _CATEGORICAL_PARAM:
            return self._param_range[int(raw_param)]
        else:
            assert False

    def retrieve_raw_param(self):
        if self.param_type == _CATEGORICAL_PARAM:
            return [0, 0, _CATEGORICAL_PARAM, list(range(len(self._param_range)))]
        else:
            lower_bound, upper_bound = self.param_bound
            return [lower_bound, upper_bound, self.param_type, None]
