from one_step_sracos.framework_adapter import adapt_framework_model
import framework.sk_models as sk
from utils.loader import adult_dataset
from one_step_sracos.bandit_model_selection import bandit_selection


def test():
    # get data set
    train_x, train_y = adult_dataset()

    # define models and initialize optimization
    models = [sk.DecisionTree(), sk.AdaBoost(), sk.Bagging(), sk.KNeighbors()]
    optimizations = [adapt_framework_model(o, train_x, train_y) for o in models]

    result = bandit_selection(optimizations, 100)
    print(result.get_features(), result.get_fitness())


if __name__ == '__main__':
    test()
