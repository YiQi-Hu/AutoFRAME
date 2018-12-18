import numpy as np

from SRacos import SRacos
from framework.base import ModelEvaluator
from framework.gbdt import LightGBM
from framework.gbdt import XGBoost

from framework.sk_models import DecisionTree
from framework.sk_models import ExtraTree
from framework.sk_models import SVC
from framework.sk_models import NuSVC
from framework.sk_models import LinearSVC
from framework.sk_models import KNeighbors
from framework.sk_models import RadiusNeighbors
from framework.sk_models import LogisticRegression
from framework.sk_models import DualLibLinearLogisticRegression
from framework.sk_models import L2PenaltyLogisticRegression
from framework.sk_models import SGD
from framework.sk_models import Ridge
from framework.sk_models import PassiveAggressive
from framework.sk_models import Perceptron
from framework.sk_models import GaussianProcess
from framework.sk_models import AdaBoost
from framework.sk_models import Bagging
from framework.sk_models import ExtraTrees
from framework.sk_models import RandomForest
from framework.sk_models import QuadraticDiscriminantAnalysis
from framework.sk_models import GaussianNB
from framework.sk_models import BernoulliNB
from framework.sk_models import MultinomialNB


def loadDatadet1(infile):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split(',')
        temp2 = list(map(float, temp2))
        dataset.append(temp2)
    return dataset


def loadDatadet2(infile):
    train_y = []
    with open(infile, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip("\n")
            train_y.append(line)
            train_y = list(map(int, train_y))
    return train_y


infile1 = './temp_dataset/Titanic/train_x2.csv'
train_x = loadDatadet1(infile1)
infile2 = './temp_dataset/Titanic/train_y.csv'
train_y = loadDatadet2(infile2)
train_x = np.array(train_x)
train_y = np.array(train_y)

# models = [DecisionTree(), ExtraTree(), SVC(), NuSVC(), LinearSVC(), KNeighbors(), RadiusNeighbors(),
#           LogisticRegression(), DualLibLinearLogisticRegression(), L2PenaltyLogisticRegression(), SGD(),
#           Ridge(), PassiveAggressive(), Perceptron(), GaussianProcess(), AdaBoost(), Bagging(),
#           ExtraTrees(), RandomForest(), QuadraticDiscriminantAnalysis(), GaussianNB(), BernoulliNB(),
#           MultinomialNB()]
models = [GaussianNB(), BernoulliNB(), MultinomialNB()]

for model in models:
    evaluator = ModelEvaluator(model_generator=model, train_x=train_x, train_y=train_y)

    dimension = [param.retrieve_raw_param() for param in model.hp_space]

    sracos = SRacos.Optimizer()
    x, y = sracos.opt(objective=evaluator.evaluate,
                      dimension=dimension, budget=10, k=3, r=5, prob=0.99, max_coordinates=2, print_opt=True)
    print("Solution", x, y)
