""""""
from model import Network
import mnist_loader

if __name__ == '__main__':
    """"""
    nn = Network([784, 30, 10])

    n_epoch = 30
    batch_size = 10
    learning_rate = 3.0
    train_data, dev_data, test_data = mnist_loader.load_data_wrapper()
    nn.run(train_data, dev_data, test_data, n_epoch, batch_size, learning_rate)

    # train_data = list(train_data)
    #
    # dev_data = list(dev_data)
    # n_dev = len(dev_data)
    #
    # test_data = list(test_data)
    # n_test = len(test_data)
    #
    # for e in range(n_epoch):
    #     nn.SGD(train_data, n_epoch, 10, 3.0)
    #     print("Epoch {} : {} / {}".format(e + 1, nn.evaluate(dev_data), n_dev))
    #
    # print("Test : {} / {}".format(nn.evaluate(test_data), n_test))
