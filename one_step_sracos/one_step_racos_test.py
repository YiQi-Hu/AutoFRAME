import logging

import framework.sk_models as sk
import matplotlib.pyplot as plt
import time
from one_step_sracos.bandit_model_selection import BanditSelection
from one_step_sracos.framework_adapter import adapt_framework_model
from utils.loader import adult_dataset

logger = logging.getLogger("bandit")
logger.setLevel(logging.DEBUG)

logger_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logger_format)

fh = logging.FileHandler('bandit_test.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logger_format)

logger.addHandler(ch)
logger.addHandler(fh)


def test():
    # get data set
    logger.debug("load data set")
    train_x, train_y = adult_dataset()

    # define models and initialize optimization
    start = time.time()
    logger.debug('initialize models')

    # decision tree, ada boost, quadratic discriminant, perceptron, logistic regression,
    models = [sk.DecisionTree(),
              sk.AdaBoost(),
              sk.QuadraticDiscriminantAnalysis(),
              sk.GaussianNB(),
              # sk.SVC(),
              sk.LinearSVC(),
              sk.KNeighbors(),
              sk.BernoulliNB(),
              sk.ExtraTree(),
              sk.MultinomialNB(),
              sk.PassiveAggressive(),
              sk.RandomForest(),
              sk.SGD()]

    optimizations = [adapt_framework_model(o, train_x, train_y) for o in models]
    logger.info('Racos run_initialization spend: {}s'.format(time.time() - start))

    logger.debug('do bandit selection')

    bandit_selection = BanditSelection(optimizations)
    bandit_selection.fit(100)

    duration = time.time() - start
    logger.info('Total time is {}'.format(duration))

    # draw converge curve
    logger.info('Converge curve is {}'.format(bandit_selection.converge_curve))
    for o in optimizations:
        logger.info('{} information: {}'.format(o.name, o.show_model()))

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(range(1, len(bandit_selection.converge_curve) + 1), bandit_selection.converge_curve)
    plt.savefig('bandit_converge_curve.svg')
    # bandit_selection(optimizations, 100)


if __name__ == '__main__':
    test()
