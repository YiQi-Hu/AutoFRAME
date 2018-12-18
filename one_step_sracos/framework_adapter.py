from framework.base import HyperParameter
from one_step_sracos.Components import Dimension
from one_step_sracos.Racos import RacosOptimization
from framework.base import ModelEvaluator
from one_step_sracos.bandit_model_selection import Optimization


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
    optimizer.run_initialization(obj_fct=evaluator.evaluate, ss=sample_size, pn=positive_num, rp=random_probability,
                                 ub=uncertain_bit)

    return Optimization(optimizer, evaluator.evaluate)


def evaluator_adapter(evaluator):
    pass
