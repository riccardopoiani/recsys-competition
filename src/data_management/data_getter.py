import numpy as np


def get_popular_items(URM, popular_threshold=100):
    """
    Get the items above a certain threshold

    :param URM: URM on which items will be extracted
    :param popular_threshold: popularity threshold
    :return:
    """
    return _get_popular(URM, popular_threshold, axis=0)


def get_active_users(URM, popular_threshold=100):
    """
    Get the users with activity above a certain threshold

    :param URM: URM on which users will be extracted
    :param popular_threshold: popularty threshold
    :return:
    """
    return _get_popular(URM, popular_threshold, axis=1)


def get_unpopular_items(URM, popular_t):
    return _get_unpopular(URM, popular_t, axis=0)


def get_unactive_users(URM, popular_t):
    return _get_unpopular(URM, popular_t, axis=1)


def get_warmer_UCM(UCM, URM_all, threshold_users):
    """
    Return the UCM with only users that has profile length more than threshold_users

    :param UCM: any UCM
    :param URM_all: URM containing all users (warm users), basically, it is the one directly from the reader
    :param threshold_users: threshold for warm users
    :return: the UCM with only users that has profile length more than threshold_users
    """
    warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > threshold_users
    warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

    return UCM.copy()[warm_users, :]


def _get_unpopular(URM, popular_threshold, axis):
    items = (URM > 0).sum(axis=axis)
    items_unsorted = np.array(items).squeeze()

    items_above_t = np.where(items_unsorted <= popular_threshold, 1, 0)

    index_list = []
    for i in range(0, items_above_t.size):
        if items_above_t[i] == 1:
            index_list.append(i)
    index_arr = np.array(index_list)

    return index_arr


def _get_popular(URM, popular_t, axis):
    items = (URM > 0).sum(axis=axis)
    items_unsorted = np.array(items).squeeze()

    items_above_t = np.where(items_unsorted > popular_t, 1, 0)

    index_list = []
    for i in range(0, items_above_t.size):
        if items_above_t[i] == 1:
            index_list.append(i)
    index_arr = np.array(index_list)

    return index_arr


def get_warm_URM(URM):
    """
    :param URM: user rating matrix
    :return: warm version of the user rating matrix
    """
    warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
    warm_items = np.arange(URM.shape[1])[warm_items_mask]

    URM = URM[:, warm_items]

    warm_users_mask = np.ediff1d(URM.tocsr().indptr) > 0
    warm_users = np.arange(URM.shape[0])[warm_users_mask]

    URM = URM[warm_users, :]

    return URM