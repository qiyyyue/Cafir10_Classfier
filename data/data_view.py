import cPickle
import numpy as np


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def init_data(data_dict):
    data = data_dict['data']
    label = data_dict['labels']

    test_data = data[0]

    # np.reshape()
    test_data = test_data.reshape(3, 32, 32)
    print test_data[0][0]



if __name__ == '__main__':

    file_dir = "cifar-10-batches-py/data_batch_1"

    data_dict = unpickle(file_dir)

    init_data(data_dict)