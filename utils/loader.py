import pickle
import numpy as np
import os.path


def dataset_reader(train_file):
    f = open(train_file, 'rb')
    train_features = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    return train_features, train_labels


def adult_dataset():
    curr_path = os.path.abspath(os.path.dirname(__file__))
    adult_dataset_path = os.path.join(curr_path, "../temp_dataset/adult/adult_train_data.pkl")
    return dataset_reader(adult_dataset_path)
