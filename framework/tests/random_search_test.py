import inspect
import os.path

import matplotlib.pyplot as plt

import framework.sk_models as sk
from framework.random_search import random_search
from utils.loader import car_dataset


def test_random_search():
    x, y = car_dataset()
    # model = sk.GaussianNB()
    # random_search(model, x, y, search_times=5)

    model_list = [m for m in inspect.getmembers(sk, inspect.isclass) if m[1].__module__ == sk.__name__]
    # filtered_models = filter(lambda p: p[0] != 'SKLearnModelGenerator' and 'NuSVC' not in p[0], model_list)
    filtered_models = [('SVC', sk.SVC)]

    # initialize image folder
    if not os.path.exists('image'):
        os.mkdir('image')
    if not os.path.exists('result_pickle'):
        os.mkdir('result_pickle')

    for name, classifier in filtered_models:
        model = classifier()
        data = random_search(model, x, y, search_times=100)

        # if this type classifiers are all down, continue so that
        # data['Accuracy'] will not raise an error
        if data['Accuracy'].empty:
            continue

        # save image and pickle file
        plt.figure()
        data['Accuracy'].plot.hist(bins=20)
        data.to_pickle('result_pickle/{}.pkl'.format(name))
        plt.savefig('image/{}.svg'.format(name), bbox_inches='tight')


if __name__ == '__main__':
    test_random_search()
