from framework.base import ModelEvaluator
from one_step_sracos.Components import Dimension
from one_step_sracos.Racos import RacosOptimization
from one_step_sracos.bandit_model_selection import Optimization
import logging
import time
import signal

logger = logging.getLogger('bandit.racos-initialization')


def adapt_framework_model(model, train_x, train_y):
    dim = Dimension()
    size = len(model.hp_space)
    dim.set_dimension_size(size)
    for i in range(size):
        param = model.hp_space[i]
        dim.set_region(i, param.param_bound, param.param_type)

    evaluator = ModelEvaluator(model, train_x, train_y)
    optimizer = RacosOptimization(dim)

    # hyper-parameter setting
    sample_size = 5
    positive_num = 2
    random_probability = 0.95
    uncertain_bit = 2

    # optimization phase
    model_name = type(model).__name__
    logger.debug('Racos begin to initialize, model is {}'.format(model_name))
    start = time.time()

    optimizer.run_initialization(obj_fct=evaluator.evaluate, ss=sample_size, pn=positive_num, rp=random_probability,
                                 ub=uncertain_bit)

    elapsed = time.time() - start
    logger.debug('Racos complete initialization for model {}, spend {}s'.format(model_name, elapsed))

    return Optimization(optimizer, evaluator.evaluate, model_name)


def evaluator_adapter(evaluator):
    pass
