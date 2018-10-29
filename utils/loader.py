import pickle
import numpy as np


def dataset_reader(train_file):

    f = open(train_file, 'rb')
    train_features = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    return train_features, train_labels
