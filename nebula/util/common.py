import pickle


def write_pickle(data, file):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file):
    with open(file, 'rb') as handle:
        b = pickle.load(handle)

    return b

