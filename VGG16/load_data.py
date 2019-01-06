import cPickle as pickle

with open("test_10k_data.pk", "rb") as rb:
    dict = pickle.load(rb)
    print dict