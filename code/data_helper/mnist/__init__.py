"""mnist"""
import numpy as np

DATA_PATH = "./mnist.npz"


def load_data(file_path, use_one_hot=True, use_dev=True):
    """"""
    data_dict = np.load(file_path)
    x_train, y_train = data_dict['x_train'], data_dict['y_train']
    x_test, y_test = data_dict['x_test'], data_dict['y_test']

    # shuffle
    # rand_index = np.arange(len(x_train))
    # np.random.shuffle(rand_index)
    # x_train, y_train = x_train[rand_index], y_train[rand_index]

    if use_one_hot:
        y_train = [one_hot(i) for i in y_train]
        y_test = [one_hot(i) for i in y_test]

    if use_dev:
        x_train, x_dev = x_train[:50000], x_train[50000:]
        y_train, y_dev = y_train[:50000], y_train[50000:]

        return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

    return (x_train, y_train), (x_test, y_test)


def one_hot(i):
    """
    Args:
        n(int):
    """
    e = np.zeros([10, 1])
    e[i] = 1.
    return e
