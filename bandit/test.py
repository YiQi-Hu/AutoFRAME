import logging
import time

import framework.sk_models as sk
import utils.loader as data_loader
from bandit.bandit_model_selection import BanditModelSelection, RandomOptimization

logger = logging.getLogger('bandit')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('bandit_test.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def bandit_test():
    # model_generators = [m for m in inspect.getmembers(sk, inspect.isclass) if m[1].__module__ == sk.__name__]
    # filtered_models = filter(lambda p: p[0] != 'SKLearnModelGenerator' and 'SVC' not in p[0], model_generators)

    # get models (models that auto-sklearn has except SVC)
    models = [
        sk.DecisionTree(),
        sk.AdaBoost(),
        sk.QuadraticDiscriminantAnalysis(),
        sk.GaussianNB(),
        sk.LinearSVC(),
        sk.KNeighbors(),
        sk.BernoulliNB(),
        sk.ExtraTree(),
        sk.MultinomialNB(),
        sk.PassiveAggressive(),
        sk.RandomForest(),
        sk.SGD()
    ]

    optimizations = [RandomOptimization(generator, type(generator).__name__) for generator in models]

    # get data sets
    data_sets = [
        ('adult', data_loader.adult_dataset()),
        ('cmc', data_loader.cmc_dataset()),
        ('car', data_loader.car_dataset()),
        ('banknote', data_loader.banknote_dataset())
    ]

    # test with the new function
    logger.info('==================New Method=====================')
    theta = 0.01
    factor = 10

    logger.info("Set theta = {}".format(theta))
    bandit_selection = BanditModelSelection(optimizations, update_func='new', theta=theta, factor=factor)
    _do_model_selection(data_sets, bandit_selection, 'model_new_{}_{}'.format(theta, factor),
                        'selection_new_{}_{}'.format(theta, factor))
    logger.info('==================New Method Done=====================')

    # test with traditional ucb function
    # logger.info('==================Traditional UCB=====================')
    # ucb_bandit_selection = BanditModelSelection(optimizations, update_func='ucb')
    # _do_model_selection(data_sets, ucb_bandit_selection, 'model_ucb', 'selection_ucb')
    # logger.info('==================Traditional UCB Done=====================')


def _do_model_selection(data, strategy, model_file, selection_file):
    assert isinstance(strategy, BanditModelSelection)

    for data_name, (train_x, train_y) in data:

        logger.info('Begin bandit selection on dataset {}'.format(data_name))
        start = time.time()

        result = strategy.fit(train_x, train_y, 1000)
        assert isinstance(result, RandomOptimization)

        elapsed_time = time.time() - start
        logger.info('Bandit selection done, spend {}s\n\n'.format(elapsed_time))

        logger.info('Selection result: \n{}\n\n'.format(result))
        logger.info('All models information:\n{}\n\n'.format(strategy.show_models()))

        strategy.statistics().to_csv('log/{}_{}.csv'.format(model_file, data_name), mode='a')
        with open('log/{}_{}.csv'.format(selection_file, data_name), 'a') as f:
            count = 13  # used to represent selection count
            for record in strategy.param_change_info:
                f.write('t = {}'.format(count))
                record.to_csv(f, mode='a')
                f.write('\n\n')

                count += 1


if __name__ == '__main__':
    bandit_test()
