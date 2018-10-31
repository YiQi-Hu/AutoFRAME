import random

import framework.base as base
import framework.sk_models as sk
from utils.loader import dataset_reader


def random_search(model_generator, train_x, train_y, search_times=100):
    evaluator = base.ModelEvaluator(model_generator, train_x, train_y)
    evaluate_result = []

    for i in range(search_times):
        # sample a set of parameters and evaluate
        params = random_sample_parameters(model_generator.hp_space)
        evaluate_result.append(evaluator.evaluate(params))

    return evaluate_result


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
    print(result)

    return result
