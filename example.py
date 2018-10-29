from SRacos import SRacos
from framework.base import ModelEvaluator
from framework.sk_models import DecisionTree
from utils.loader import dataset_reader


data_file = './temp_dataset/adult/adult_train_data.pkl'
train_x, train_y = dataset_reader(data_file)
model = DecisionTree()
evaluator = ModelEvaluator(model_generator=model, train_x=train_x, train_y=train_y)

dimension = [param.retrieve_raw_param() for param in model.hp_space]


sracos = SRacos.Optimizer()
x, y = sracos.opt(objective=evaluator.evaluate,
                  dimension=dimension, budget=10000, k=3, r=5, prob=0.99, max_coordinates=2)
print(x, y)
