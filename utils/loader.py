import pickle
import numpy as np
import os.path

curr_path = os.path.abspath(os.path.dirname(__file__))


def dataset_reader(train_file):
    f = open(train_file, 'rb')
    train_features = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    return train_features, train_labels


def adult_dataset():
    x1, y1 = dataset_reader(os. path.join(curr_path, "../temp_dataset/adult/adult_train_data.pkl"))
    x2, y2 = dataset_reader(os.path.join(curr_path, "../temp_dataset/adult/adult_test_data.pkl"))
    return np.concatenate([x1, x2]), np.concatenate([y1, y2])


def car_dataset():
    x1, y1 = dataset_reader(os.path.join(curr_path, "../temp_dataset/car/car_train_data.pkl"))
    x2, y2 = dataset_reader(os.path.join(curr_path, "../temp_dataset/car/car_test_data.pkl"))
    return np.concatenate([x1, x2]), np.concatenate([y1, y2])


def cmc_dataset():
    x, y = dataset_reader(os.path.join(curr_path, '../temp_dataset/cmc/cmc_train_data.pkl'))
    x2, y2 = dataset_reader(os.path.join(curr_path, '../temp_dataset/cmc/cmc_test_data.pkl'))

    return np.concatenate([x, x2]), np.concatenate([y, y2])


def banknote_dataset():
    x, y = dataset_reader(os.path.join(curr_path, '../temp_dataset/banknote/banknote_train_data.pkl'))
    x2, y2 = dataset_reader(os.path.join(curr_path, '../temp_dataset/banknote/banknote_test_data.pkl'))

    return np.concatenate([x, x2]), np.concatenate([y, y2])
