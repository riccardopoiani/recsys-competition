import numpy as np

def get_popular_items(URM, popular_threshold=100):
    '''
    Get the items above a certain threshold

    :param URM: URM on which items will be extracted
    :param popular_threshold: popularity threshold
    :return:
    '''
    return __get_popular__(URM, popular_threshold, axis=0)


def get_active_users(URM, popular_threshold=100):
    '''
    Get the users with activity above a certain threshold

    :param URM: URM on which users will be extracted
    :param popular_threshold: popularty threshold
    :return:
    '''
    return __get_popular__(URM, popular_threshold, axis=1)

def get_unpopular_items(URM, popular_t):
    return __get_unpopular__(URM, popular_t, axis=0)

def get_unactive_users(URM, popular_t):
    return __get_unpopular__(URM, popular_t, axis=1)

def __get_unpopular__(URM, popular_threshold, axis):
    items = (URM > 0).sum(axis=axis)
    items_unsorted = np.array(items).squeeze()

    items_above_t = np.where(items_unsorted <= popular_threshold, 1, 0)

    index_list = []
    for i in range(0, items_above_t.size):
        if items_above_t[i] == 1:
            index_list.append(i)
    index_arr = np.array(index_list)

    return index_arr

def __get_popular__(URM, popular_t, axis):
    items = (URM > 0).sum(axis=axis)
    items_unsorted = np.array(items).squeeze()

    items_above_t = np.where(items_unsorted > popular_t, 1, 0)

    index_list = []
    for i in range(0, items_above_t.size):
        if items_above_t[i] == 1:
            index_list.append(i)
    index_arr = np.array(index_list)

    return index_arr
