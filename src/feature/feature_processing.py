import numpy as np


def transform_numerical_to_label(x: np.ndarray, bins=20):
    """
    Given an array x, containing continuous numerical values, it digitizes the array into
    <<bins>> labels.

    :param x: array of numerical values
    :param bins: number of labels in the output
    :return: array of labelled values
    """
    eps = 10e-6
    norm_x = (x - x.min()) / (x.max() - x.min() + eps) * 100
    bins_list = [i * (norm_x.max()/bins) for i in range(bins)]
    labelled_x = np.digitize(norm_x, bins_list, right=True)
    return labelled_x
