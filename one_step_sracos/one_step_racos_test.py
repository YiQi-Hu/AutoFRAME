import logging

import framework.sk_models as sk
from one_step_sracos.bandit_model_selection import bandit_selection
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
    logger.debug('initialize models')
    models = [sk.AdaBoost(), sk.QuadraticDiscriminantAnalysis(), sk.Perceptron(), sk.L2PenaltyLogisticRegression(),
              sk.GaussianNB()]
    optimizations = [adapt_framework_model(o, train_x, train_y) for o in models]

    logger.debug('do bandit selection')
    bandit_selection(optimizations, 100)


if __name__ == '__main__':
    test()
