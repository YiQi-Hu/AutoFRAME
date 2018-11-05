import inspect
import logging
import time
import random
import signal
import sys
import os.path

import framework.base as base
import framework.sk_models as sk
from utils.loader import dataset_reader


# --------------------------------------------------------
# define a logger

def _init_logger():
    logger = logging.getLogger('AutoML_Random_Search')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('AutoML.log')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


log = _init_logger()


def timeout_handler(signum, frame):
    raise Exception("Timeout!")


signal.signal(signal.SIGALRM, timeout_handler)


def random_search(model_generator, train_x, train_y, search_times=100):
    evaluator = base.ModelEvaluator(model_generator, train_x, train_y)

    for i in range(search_times):
        # sample a set of parameters and evaluate
        params = random_sample_parameters(model_generator.hp_space)
        pattern = '[{}]: parameters: {}, accuracy: {:.5%}, time: {}s'
        try:
            start = time.time()
            signal.alarm(120)

            accuracy = evaluator.evaluate(params)

            signal.alarm(0)
            elapsed = time.time() - start

            log.info(pattern.format(name, params, float(accuracy), elapsed))
        except Exception as e:
            log.error('[{}]: parameters: {}, error message: {}'.format(name, params, e))
        except RuntimeWarning as w:
            log.warning('[{}]: parameters: {}, warning message: {}'.format(name, params, w))


def random_sample_parameters(hp_space):
    assert len(hp_space) > 0

    result = []

    for param in hp_space:
        assert isinstance(param, base.HyperParameter)

        if param.param_type == base.INTEGER or param.param_type == base.CATEGORICAL:
            result.append(random.randint(*param.param_bound))
        elif param.param_type == base.FLOAT:
            result.append(random.uniform(*param.param_bound))
        else:
            assert False

    assert len(result) == len(hp_space)

    return result


if __name__ == '__main__':

    my_path = os.path.abspath(os.path.dirname(__file__))
    data_file = os.path.join(my_path, '../../temp_dataset/adult/adult_train_data.pkl')
    x, y = dataset_reader(data_file)
    # model = sk.LinearDiscriminantAnalysis()
    model = sk.AdaBoost()
    model.generate_model([100, 0.2, 1])

    model_list = [m for m in inspect.getmembers(sk, inspect.isclass) if m[1].__module__ == sk.__name__]

    for name, classifier in filter(lambda p: p[0] != 'SKLearnModelGenerator', model_list):
        model = classifier()
        random_search(model, x, y, search_times=10)
