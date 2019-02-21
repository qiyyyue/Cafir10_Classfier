import os
import cPickle as pickle
import numpy as np


def convert_one_hot(m, dim):
    for i in range(len(m)):
        tmp = np.zeros(dim)
        tmp[int(m[i])] = 1
        m[i] = tmp

    return m

if __name__ == '__main__':
    origin_img = []
    adv_img = []
    origin_labels = []
    adv_labels = []


    for i in range(20):
        with open(str(i*500), "rb") as r:
            print i*500
            # pickle.load(r).keys()
            data_dict = pickle.load(r)
            # print data_dict['origin_img'].shape
            # print data_dict['adv_img'].shape
            # print data_dict['origin_lables'].shape
            # print data_dict['adv_labels'].shape
            origin_img += data_dict['origin_img'].tolist()
            adv_img += data_dict['adv_img'].tolist()
            origin_labels += data_dict['origin_lables'].tolist()
            adv_labels += data_dict['adv_labels'].tolist()
            # break

    print np.array(origin_img).shape
    print np.array(adv_img).shape
    print np.array(origin_labels).shape
    print np.array(adv_labels).shape

    np_org_labels = np.array(convert_one_hot(origin_labels, 10))
    np_adv_labels = np.array(convert_one_hot(adv_labels, 10))

    with open("10K_data", "wb") as w:
        pickle.dump({"origin_img": np.array(origin_img).astype(float), "adv_img": np.array(adv_img).astype(float), "origin_labels": np_org_labels.astype(float), "adv_labels": np_adv_labels.astype(float)}, w)
