import numpy as np
import os

SPLIT_SEED = 69420
TOTAL_USERS = 30911
TOTAL_ITEMS = 18495


def get_total_number_of_users():
    return TOTAL_USERS


def get_total_number_of_items():
    return TOTAL_ITEMS


def get_split_seed():
    return SPLIT_SEED


def get_seed_lists(size, seed):
    np.random.seed(seed)
    return np.random.randint(low=0, high=2**16-1, size=size, dtype=np.int)


def get_project_root_path():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def block_print():
    import sys
    import os
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    import sys
    sys.stdout = sys.__stdout__


def get_root_data_path():
    root_path = get_project_root_path()
    return os.path.join(root_path, "data")


def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def from_string_to_dict(s):
    """
    Convert a string to a dictionary.
    This function takes care of True/
    False values, integer, float and string, as possible values.
    It does not handle nasted dictionary or more complex structure such as tuple, list elements and so on.

    :param s: string represeting a dictionary. { } are assumed to be already removed
    :return: dictionary of the string
    """
    my_dict = {}

    split_dict = s.split(",")

    for elem in split_dict:
        split_elem = elem.split(":")
        key = split_elem[0]
        key = key.replace("'", "")
        key = key.replace(" ", "")
        value = split_elem[1]

        value = value.replace(" ", "")

        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        elif represents_int(value):
            value = int(value)
        elif represents_float(value):
            value = float(value)
        else:
            value = value.replace("'", "")
        my_dict[key] = value

    return my_dict


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def represents_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def n_ranges(start, end, return_flat=True):
    """
    Returns n ranges, n being the length of start (or end, they must be the same length) where each value in
    start represents the start of a range, and a value in end at the same index the end of it

    :param start: 1D np.ndarray representing the start of a range. Each value in start must be <=
                  than that of stop in the same index
    :param end: 1D np.ndarray representing the end of a range
    :param return_flat:
    :return All ranges flattened in a 1darray if return_flat is True otherwise an array of arrays with a range in each
    """
    # lengths of the ranges
    lens = end - start
    # repeats starts as many times as lens
    start_rep = np.repeat(start, lens)
    # helper mask with as many True in each row
    # as value in same index in lens
    arr = np.arange(lens.max())
    m =  arr < lens[:,None]
    # ranges in a flattened 1d array
    # right term is a cumcount up to each len
    ranges = start_rep + (arr * m)[m]
    # returns if True otherwise in split arrays
    if return_flat:
        return ranges
    else:
        return np.split(ranges, np.cumsum(lens)[:-1])