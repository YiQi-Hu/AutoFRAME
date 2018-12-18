"""
this file covers optimizers in tf.train.
the range of the paramter is all closed.
use (1e-6,upperboundary) to imply the parameter should be larger than 0.
Author: Ting Bao
"""

import numpy as np
import tensorflow.train
from framework.base import HyperParameter,ModelGenerator


#the definition of this class is the same of "SKLearnModelGenerator" 
class OptimizerGenerator(ModelGenerator):

    def __init__(self, hp_space, model_initializer):
        super().__init__(hp_space, model_initializer)

    def generate_model(self, param_values):
        optimizer = self._model_initializer()
        assert len(param_values) == len(self.hp_space)
        # Check and set each parameters for the new model
        for value, param in zip(param_values, self.hp_space):
            if not param.in_range(value):
                raise ValueError('Value of parameter {} is not in range'.format(param.name))

            assert hasattr(optimizer, param.name), 'optimizer is {}, invalid parameter is {}'.format(optimizer, param.name)
            setattr(optimizer, param.name, param.convert_raw_param(value))

        return optimizer


class GradientDescentOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.GradientDescentOptimizer
        super().__init__(hp_space, model_initializer)


class AdadeltaOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('rho', (1e-6, 1.)),#recheck needed
            HyperParameter.float_param('epsilon', (1e-6, 0.1)),#wider range is allowed but not suggested
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.AdadeltaOptimizer
        super().__init__(hp_space, model_initializer)


class AdagradOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('initial_accumulator_value', (1e-6, 2.)),
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.AdagradOptimizer
        super().__init__(hp_space, model_initializer)


class AdagradDAOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.int_param('global_step', (0, 1e9)),
            HyperParameter.float_param('initial_gradient_squared_accumulator_value', (1e-6, 1.)),
            HyperParameter.float_param('l1_regularization_strength', (0., 1.)),
            HyperParameter.float_param('l2_regularization_strength', (0., 1.)),
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.AdagradDAOptimizer
        super().__init__(hp_space, model_initializer)


class MomentumOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('momentum', (0., 1.)),
            HyperParameter.categorical_param('use_locking', (True, False)),
            HyperParameter.categorical_param('use_nesterov', (True, False))
        ]

        model_initializer = tf.train.MomentumOptimizer
        super().__init__(hp_space, model_initializer)


class AdamOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('beta1', (0., 1.-1e-6)),
            HyperParameter.float_param('beta2', (0., 1.-1e-6)),
            HyperParameter.float_param('epsilon', (1e-6, 0.1)),
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.AdamOptimizer
        super().__init__(hp_space, model_initializer)


class FtrlOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('learning_rate_power', (-2.,-1e-6)),
            HyperParameter.float_param('initial_accumulator_value', (0., 2.)),
            HyperParameter.float_param('l1_regularization_strength', (0., 2.)),
            HyperParameter.float_param('l2_regularization_strength', (0., 2.)),
            HyperParameter.categorical_param('use_locking', (True, False)),
            HyperParameter.float_param('l2_shrinkage_regularization_strength', (0., 2.)),
            ]

        model_initializer = tf.train.FtrlOptimizer
        super().__init__(hp_space, model_initializer)


class ProximalGradientDescentOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('l1_regularization_strength', (0., 2.)),
            HyperParameter.float_param('l2_regularization_strength', (0., 2.)),
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.ProximalGradientDescentOptimizer
        super().__init__(hp_space, model_initializer)


class ProximalAdagradOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('initial_accumulator_value', (1e-6, 2.)),
            HyperParameter.float_param('l1_regularization_strength', (0., 2.)),
            HyperParameter.float_param('l2_regularization_strength', (0., 2.)),
            HyperParameter.categorical_param('use_locking', (True, False))
        ]

        model_initializer = tf.train.ProximalAdagradOptimizer
        super().__init__(hp_space, model_initializer)


class RMSPropOptimizer(OptimizerGenerator):
    def __init__(self):
        hp_space = [
            HyperParameter.float_param('learning_rate', (1e-6, 0.15)),
            HyperParameter.float_param('decay', (0., 1.)),
            HyperParameter.float_param('momentum', (0., 1.)),
            HyperParameter.float_param('epsilon', (1e-6, 0.1)),
            HyperParameter.categorical_param('use_locking', (True, False)),
            HyperParameter.categorical_param('centered', (True, False)),
        ]

        model_initializer = tf.trainRMSPropOptimizer
        super().__init__(hp_space, model_initializer)




