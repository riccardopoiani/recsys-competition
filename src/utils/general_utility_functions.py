


SPLIT_SEED = 69420


def get_split_seed():
    return SPLIT_SEED


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


def from_string_to_dict(s):
    '''
    Convert a string to a dictionary.
    This function takes care of True/
    False values, integer, float and string, as possible values.
    It does not handle nasted dictionary or more complex structure such as tuple, list elements and so on.

    :param s: string represeting a dictionary. { } are assumed to be already removed
    :return: dictionary of the string
    '''
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
