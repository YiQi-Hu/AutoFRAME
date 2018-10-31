import framework.sk_models as sk
from utils.loader import dataset_reader
from utils.random_search import random_search

data_file = './temp_dataset/adult/adult_train_data.pkl'
x, y = dataset_reader(data_file)
model = sk.LinearDiscriminantAnalysis()
print(random_search(model, x, y, search_times=10))
