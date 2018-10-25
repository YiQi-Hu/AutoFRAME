from framework.base import ModelGenerator, HyperParameter

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class GBDTModelGenerator(ModelGenerator):

    def __init__(self, hp_space, model_initializer):
        super().__init__(hp_space, model_initializer)

    def generate_model(self, param_values):
        model = self._model_initializer()
        assert len(param_values) == len(self.hp_space)
        # Check and set each parameters for the new model
        for value, param in zip(param_values, self.hp_space):
            if not param.in_range(value):
                raise ValueError('Value of parameter {} is not in range'.format(param.name))

            # set_params should be both defined in LGBMClassifier and XGBClassifier
            # XGBoost may not supported set_params in current released version
            # If want to use XGBoost generator, wait for the next version or replace
            # sklearn.py in your python site-packages with the file
            # https://raw.githubusercontent.com/dmlc/xgboost/master/python-package/xgboost/sklearn.py
            assert hasattr(model, 'set_params')
            model.set_params(**{param.name: param.convert_raw_param(value)})

        return model


class LightGBM(GBDTModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('boosting_type', ('gbdt', 'dart', 'goss', 'rf')),
            HyperParameter.int_param('num_leaves', (10, 300)),
            HyperParameter.int_param('max_depth', (-1, 100)),
            HyperParameter.float_param('learning_rate', (0.01, 1)),
            HyperParameter.int_param('n_estimators', (10, 1000)),
            HyperParameter.int_param('subsample_for_bin', (20000, 2000000)),
            HyperParameter.float_param('min_split_gain', (0., 10.)),
            HyperParameter.float_param('min_child_weight', (1e-4, 1e-2)),
            HyperParameter.int_param('min_child_samples', (1, 20)),
            HyperParameter.float_param('subsample', (0.1, 1.0)),
            HyperParameter.int_param('subsample_freq', (-1, 10)),
            HyperParameter.float_param('colsample_bytree', (0.1, 1.0)),
            HyperParameter.float_param('reg_alpha', (0.0, 1e4)),
            HyperParameter.float_param('reg_lambda', (0.0, 1e4)),
            HyperParameter.int_param('max_bin', (20, 2000)),
            HyperParameter.float_param('drop_rate', (0.0, 1.0)),
            HyperParameter.int_param('max_drop', (0, 500)),
            HyperParameter.float_param('skip_drop', (0.0, 1.0)),
            HyperParameter.float_param('top_rate', (0.0, 1.0)),
            HyperParameter.float_param('other_rate', (0.0, 1.0)),
            HyperParameter.int_param('min_data_in_bin', (3, 30)),
            HyperParameter.float_param('sparse_threshold', (0.1, 1.0))
        ]

        initializer = LGBMClassifier
        super().__init__(hp_space, initializer)


class XGBoost(GBDTModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('max_depth', (0, 60)),
            HyperParameter.float_param('learning_rate', (0, 1)),
            HyperParameter.int_param('n_estimators', (50, 1000)),
            HyperParameter.categorical_param('booster', ('gbtree', 'gblinear', 'dart')),
            HyperParameter.float_param('gamma', (0, 10000)),
            HyperParameter.int_param('min_child_weight', (0, 100)),
            HyperParameter.int_param('max_delta_step', (0, 10)),
            HyperParameter.float_param('subsample', (0.1, 1)),
            HyperParameter.float_param('colsample_bytree', (0.1, 1)),
            HyperParameter.float_param('colsample_bylevel', (0.1, 1)),
            HyperParameter.float_param('reg_alpha', (0.0, 1e4)),
            HyperParameter.float_param('reg_lambda', (0.0, 1e4)),
            HyperParameter.categorical_param('tree_method', ('exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist')),
            HyperParameter.float_param('sketch_eps', (0.003, 1)),
            HyperParameter.categorical_param('grow_policy', ('depthwise', 'lossguide')),
            HyperParameter.int_param('max_leaves', (0, 100)),
            HyperParameter.int_param('max_bin', (20, 2000)),
            HyperParameter.categorical_param('sample_type', ('uniform', 'weighted')),
            HyperParameter.categorical_param('normalize_type', ('tree', 'forest')),
            HyperParameter.float_param('rate_drop', (0, 1)),
            HyperParameter.float_param('skip_drop', (0, 1)),
            HyperParameter.categorical_param('updater', ('shotgun', 'coord_descent')),
            HyperParameter.categorical_param('feature_selector', ('cyclic', 'shuffle', 'random', 'greedy', 'thrifty'))
        ]

        initializer = XGBClassifier
        super().__init__(hp_space, initializer)
