import numpy as np

from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_preprocessing import apply_feature_engineering_UCM, apply_transformation_UCM, \
    apply_discretization_UCM, build_UCM_all_from_dict


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


def get_UCM_train(reader: New_DataSplitter_leave_k_out, root_data_path: str):
    """
    It returns all the UCM_all after applying feature engineering

    :param reader: data splitter
    :param root_data_path: the root path of the data folder
    :return: return UCM_all
    """
    URM_train, _ = reader.get_holdout_split()
    UCM_all_dict = reader.get_loaded_UCM_dict()
    data_reader = RecSys2019Reader(root_data_path)
    data_reader.load_data()
    ICM_dict = data_reader.get_loaded_ICM_dict()
    UCM_all_dict = apply_feature_engineering_UCM(UCM_all_dict, URM_train, ICM_dict,
                                                 ICM_names_to_UCM=["ICM_sub_class", "ICM_price"])

    # These are useful feature weighting for UserCBF_CF_Warm
    UCM_all_dict = apply_transformation_UCM(UCM_all_dict,
                                            UCM_name_to_transform_mapper={"UCM_sub_class": lambda x: x / 2,
                                                                          "UCM_user_act": np.log1p})
    UCM_all_dict = apply_discretization_UCM(UCM_all_dict, UCM_name_to_bins_mapper={"UCM_user_act": 50})
    UCM_all = build_UCM_all_from_dict(UCM_all_dict)
    return UCM_all


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
