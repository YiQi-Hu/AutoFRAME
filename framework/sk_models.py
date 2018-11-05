import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree

from framework.base import ModelGenerator, HyperParameter


class SKLearnModelGenerator(ModelGenerator):

    def __init__(self, hp_space, model_initializer):
        super().__init__(hp_space, model_initializer)

    def generate_model(self, param_values):
        model = self._model_initializer()
        assert len(param_values) == len(self.hp_space)
        # Check and set each parameters for the new model
        for value, param in zip(param_values, self.hp_space):
            if not param.in_range(value):
                raise ValueError('Value of parameter {} is not in range'.format(param.name))

            assert hasattr(model, param.name), 'model is {}, invalid parameter is {}'.format(model, param.name)
            setattr(model, param.name, param.convert_raw_param(value))

        return model


class DecisionTree(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('criterion', ('gini', 'entropy')),
            HyperParameter.int_param('max_depth', (1, 40)),
            HyperParameter.int_param('min_samples_split', (1, 100)),
            HyperParameter.int_param('min_samples_leaf', (1, 100)),
            HyperParameter.categorical_param('max_features', ('sqrt', 'log2', None)),
            HyperParameter.int_param('max_leaf_nodes', (2, 100)),
            HyperParameter.float_param('min_impurity_decrease', (0., 100.))
        ]

        model_initializer = sklearn.tree.DecisionTreeClassifier
        super().__init__(hp_space, model_initializer)


class ExtraTree(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('criterion', ('gini', 'entropy')),
            HyperParameter.int_param('max_depth', (1, 40)),
            HyperParameter.int_param('min_samples_split', (1, 100)),
            HyperParameter.int_param('min_samples_leaf', (1, 100)),
            HyperParameter.categorical_param('max_features', ('sqrt', 'log2', None)),
            HyperParameter.int_param('max_leaf_nodes', (2, 100)),
            HyperParameter.float_param('min_impurity_decrease', (0., 100.))
        ]

        model_initializer = sklearn.tree.ExtraTreeClassifier
        super().__init__(hp_space, model_initializer)


class SVC(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('C', (0.01, 1e6)),
            HyperParameter.categorical_param('kernel', ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')),
            HyperParameter.int_param('degree', (1, 30)),
            HyperParameter.float_param('gamma', (1e-5, 10)),
            HyperParameter.float_param('coef0', (0., 100.)),
            HyperParameter.categorical_param('shrinking', (True, False)),
            HyperParameter.float_param('tol', (1e-5, 1e-1))
        ]

        model_initializer = sklearn.svm.SVC
        super().__init__(hp_space, model_initializer)


class NuSVC(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('nu', (5e-3, 1)),
            HyperParameter.categorical_param('kernel', ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')),
            HyperParameter.int_param('degree', (1, 30)),
            HyperParameter.float_param('gamma', (1e-5, 10)),
            HyperParameter.float_param('coef0', (0., 100.)),
            HyperParameter.categorical_param('shrinking', (True, False)),
            HyperParameter.float_param('tol', (1e-5, 1e-1))
        ]

        model_initializer = sklearn.svm.NuSVC
        super().__init__(hp_space, model_initializer)


class LinearSVC(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('penalty', ('l1', 'l2')),
            HyperParameter.categorical_param('loss', ('hinge', 'squared_hinge')),
            HyperParameter.categorical_param('dual', (True, False)),
            HyperParameter.float_param('tol', (1e-6, 1e-1)),
            HyperParameter.float_param('C', (0.01, 1e6))
        ]

        initializer = sklearn.svm.LinearSVC
        super().__init__(hp_space, initializer)


class KNeighbors(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('n_neighbors', (0, 50)),
            HyperParameter.categorical_param('weights', ('uniform', 'distance')),
            HyperParameter.categorical_param('algorithm', ('ball_tree', 'kd_tree', 'brute')),
            HyperParameter.int_param('leaf_size', (3, 100)),
            HyperParameter.int_param('p', (1, 10))
        ]

        initializer = sklearn.neighbors.KNeighborsClassifier
        super().__init__(hp_space, initializer)


class RadiusNeighbors(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('radius', (1e-2, 1e2)),
            HyperParameter.categorical_param('weights', ('uniform', 'distance')),
            HyperParameter.categorical_param('algorithm', ('ball_tree', 'kd_tree', 'brute')),
            HyperParameter.int_param('leaf_size', (3, 100)),
            HyperParameter.int_param('p', (1, 10))
        ]

        initializer = sklearn.neighbors.RadiusNeighborsClassifier
        super().__init__(hp_space, initializer)


class LogisticRegression(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('penalty', ('l1', 'l2')),
            HyperParameter.categorical_param('dual', (True, False)),
            HyperParameter.float_param('tol', (1e-6, 1e-1)),
            HyperParameter.float_param('C', (1e-2, 1e2)),
            HyperParameter.categorical_param('solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')),
            HyperParameter.int_param('max_iter', (100, 1000)),
            HyperParameter.categorical_param('multi_class', ('ovr', 'multinomial'))
        ]

        initializer = sklearn.linear_model.LogisticRegression
        super().__init__(hp_space, initializer)


class SGD(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('loss', ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron')),
            HyperParameter.categorical_param('penalty', ('none', 'l2', 'l1', 'elasticnet')),
            HyperParameter.float_param('alpha', (1e-5, 1e-3)),
            HyperParameter.float_param('l1_ratio', (0.0, 1.0)),
            HyperParameter.int_param('max_iter', (1000, 10000)),
            HyperParameter.float_param('tol', (1e-4, 1e-2)),
            HyperParameter.categorical_param('learning_rate', ('constant', 'optimal', 'invscaling', 'adaptive')),
            HyperParameter.float_param('eta0', (0.0, 10.0)),
            HyperParameter.float_param('power_t', (0.05, 1))
        ]

        initializer = sklearn.linear_model.SGDClassifier
        super().__init__(hp_space, initializer)


class Ridge(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('alpha', (1e-2, 1e2)),
            HyperParameter.int_param('max_iter', (1000, 10000)),
            HyperParameter.float_param('tol', (1e-4, 1e-2)),
            HyperParameter.categorical_param('solver', ('svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'))
        ]

        initializer = sklearn.linear_model.RidgeClassifier
        super().__init__(hp_space, initializer)


class PassiveAggressive(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('C', (0.1, 10)),
            HyperParameter.int_param('max_iter', (1000, 10000)),
            HyperParameter.float_param('tol', (1e-4, 1e-2)),
            HyperParameter.categorical_param('loss', ('hinge', 'squared_hinge'))
        ]

        initializer = sklearn.linear_model.PassiveAggressiveClassifier
        super().__init__(hp_space, initializer)


class Perceptron(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('penalty', (None, 'l2', 'l1', 'elasticnet')),
            HyperParameter.float_param('alpha', (1e-5, 1e-3)),
            HyperParameter.int_param('max_iter', (1000, 10000)),
            HyperParameter.float_param('tol', (1e-4, 1e-2)),
            HyperParameter.float_param('eta0', (0.1, 10))
        ]

        initializer = sklearn.linear_model.Perceptron
        super().__init__(hp_space, initializer)


class GaussianProcess(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('max_iter_predict', (10, 1000))
        ]

        initializer = sklearn.gaussian_process.GaussianProcessClassifier
        super().__init__(hp_space, initializer)


class AdaBoost(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('n_estimators', (30, 500)),
            HyperParameter.float_param('learning_rate', (0.1, 2.)),
            HyperParameter.categorical_param('algorithm', ('SAMME', 'SAMME.R'))
        ]

        initializer = sklearn.ensemble.AdaBoostClassifier
        super().__init__(hp_space, initializer)


class Bagging(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('n_estimators', (5, 100)),
            HyperParameter.float_param('max_samples', (0.0, 1.0)),
            HyperParameter.float_param('max_features', (0.0, 1.0))
        ]

        initializer = sklearn.ensemble.BaggingClassifier
        super().__init__(hp_space, initializer)


class ExtraTrees(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('n_estimators', (5, 1000)),
            HyperParameter.categorical_param('criterion', ('gini', 'entropy')),
            HyperParameter.int_param('max_depth', (2, 40)),
            HyperParameter.int_param('min_samples_split', (1, 100)),
            HyperParameter.int_param('min_samples_leaf', (1, 100)),
            HyperParameter.categorical_param('max_features', ('sqrt', 'log2', None)),
            HyperParameter.int_param('max_leaf_nodes', (-1, 100)),
            HyperParameter.float_param('min_impurity_decrease', (0.0, 100.0))
        ]

        initializer = sklearn.ensemble.ExtraTreesClassifier
        super().__init__(hp_space, initializer)


class RandomForest(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.int_param('n_estimators', (10, 1000)),
            HyperParameter.int_param('max_depth', (0, 40)),
            HyperParameter.int_param('min_samples_split', (1, 100)),
            HyperParameter.int_param('min_samples_leaf', (1, 100)),
            HyperParameter.categorical_param('max_features', ('sqrt', 'log2', None)),
            HyperParameter.int_param('max_leaf_nodes', (-1, 100)),
            HyperParameter.float_param('min_impurity_decrease', (0.0, 100.0))
        ]

        initializer = sklearn.ensemble.RandomForestClassifier
        super().__init__(hp_space, initializer)


class LinearDiscriminantAnalysis(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.categorical_param('solver', ('svd', 'lsqr', 'eigen')),
            HyperParameter.float_param('shrinkage', (0, 1)),
            HyperParameter.float_param('tol', (1e-5, 1e-3))
        ]

        initializer = sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        super().__init__(hp_space, initializer)


class QuadraticDiscriminantAnalysis(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('reg_param', (0., 1.)),
            HyperParameter.float_param('tol', (1e-5, 1e-3))
        ]

        initializer = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
        super().__init__(hp_space, initializer)


class GaussianNB(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('var_smoothing', (1e-10, 1e-8))
        ]

        initializer = sklearn.naive_bayes.GaussianNB
        super().__init__(hp_space, initializer)


class BernoulliNB(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('alpha', (0.0, 10.0)),
            HyperParameter.float_param('binarize', (0.0, 1.0))
        ]

        initializer = sklearn.naive_bayes.BernoulliNB
        super().__init__(hp_space, initializer)


class MultinomialNB(SKLearnModelGenerator):

    def __init__(self):
        hp_space = [
            HyperParameter.float_param('alpha', (0.0, 10.0))
        ]

        initializer = sklearn.naive_bayes.MultinomialNB
        super().__init__(hp_space, initializer)
