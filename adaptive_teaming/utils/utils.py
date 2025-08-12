from hydra.utils import to_absolute_path
import pickle as pkl


def pkl_load(filename, absolute=True):
    if absolute:
        filename = to_absolute_path(filename)
    return pkl.load(open(filename, 'rb'))


def pkl_dump(object, filename, absolute=False):
    if absolute:
        filename = to_absolute_path(filename)
    return pkl.dump(object, open(filename, 'wb'))

