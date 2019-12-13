from course_lib.Data_manager.DataReader import DataReader
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_preprocessing import apply_feature_engineering_ICM, apply_filtering_ICM, \
    apply_transformation_ICM, apply_discretization_ICM, build_ICM_all_from_dict, apply_feature_engineering_UCM, \
    apply_transformation_UCM, apply_discretization_UCM, build_UCM_all_from_dict

import numpy as np


def read_target_users(path="../data/data_target_users_test.csv"):
    """
    :return: list of user to recommend in the target playlist
    """
    target_file = open(path, 'r')

    target_file.seek(0)
    target_tuple = []

    for line in target_file:
        if line != "user_id\n":
            target_tuple.append(row_split_target(line))

    return target_tuple


def read_URM_cold_all(path="../data/data_train.csv"):
    """
    :return: all the user rating matrix, in csr format
    """
    import scipy.sparse as sps
    import numpy as np
    import pandas as pd

    # Reading data
    df_original = pd.read_csv(path)

    user_id_list = df_original['row'].values
    item_id_list = df_original['col'].values
    rating_list = np.ones(len(user_id_list))

    # Creating URM
    URM_all = sps.coo_matrix((rating_list, (user_id_list, item_id_list)))
    URM_all = URM_all.tocsr()

    return URM_all


def read_UCM_cold_all(num_users, root_path="../data/"):
    """
    :return: all the UCM in csr format
    """
    import scipy.sparse as sps
    import numpy as np
    import pandas as pd
    import os

    # Reading age data
    df_age = pd.read_csv(os.path.join(root_path, "data_UCM_age.csv"))

    user_id_list = df_age['row'].values
    age_id_list = df_age['col'].values
    UCM_age = sps.coo_matrix((np.ones(len(user_id_list)), (user_id_list, age_id_list)),
                             shape=(num_users, np.max(age_id_list) + 1))

    # Reading region data
    df_region = pd.read_csv(os.path.join(root_path, "data_UCM_region.csv"))
    user_id_list = df_region['row'].values
    region_id_list = df_region['col'].values
    UCM_region = sps.coo_matrix((np.ones(len(user_id_list)), (user_id_list, region_id_list)),
                                shape=(num_users, np.max(region_id_list) + 1))

    # Merge UCMs
    UCM_all, _ = merge_UCM(UCM_age, UCM_region, {}, {})
    UCM_all = UCM_all.tocsr()
    return UCM_all


def get_ICM_all(reader: RecSys2019Reader):
    """
    It returns all the ICM_all after applying feature engineering

    :param reader: data splitter
    :return: return ICM_all
    """
    URM_all = reader.get_URM_all()
    UCM_all_dict = reader.get_loaded_UCM_dict()
    ICM_all_dict = reader.get_loaded_ICM_dict()
    ICM_all_dict.pop("ICM_all")
    ICM_all_dict = apply_feature_engineering_ICM(ICM_all_dict, URM_all, UCM_all_dict,
                                                 ICM_names_to_count=["ICM_sub_class"], UCM_names_to_list=["UCM_age"])
    ICM_all_dict = apply_filtering_ICM(ICM_all_dict,
                                       ICM_name_to_filter_mapper={"ICM_asset": lambda x: x < np.quantile(x, q=0.75) +
                                                                                         0.72 * (np.quantile(x,
                                                                                                             q=0.75) -
                                                                                                 np.quantile(x,
                                                                                                             q=0.25)),
                                                                  "ICM_price": lambda x: x < np.quantile(x, q=0.75) +
                                                                                         4 * (np.quantile(x, q=0.75) -
                                                                                              np.quantile(x, q=0.25))})
    ICM_all_dict = apply_transformation_ICM(ICM_all_dict,
                                            ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                          "ICM_price": lambda x: np.log1p(1 / x),
                                                                          "ICM_item_pop": np.log1p,
                                                                          "ICM_sub_class_count": np.log1p,
                                                                          "ICM_age": lambda x: x ** (1 / 2.5)})
    ICM_all_dict = apply_discretization_ICM(ICM_all_dict,
                                            ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                     "ICM_price": 200,
                                                                     "ICM_item_pop": 50,
                                                                     "ICM_sub_class_count": 50})
    ICM_all = build_ICM_all_from_dict(ICM_all_dict)
    return ICM_all


def get_ICM_train(reader: New_DataSplitter_leave_k_out):
    """
    It returns all the ICM_train_all after applying feature engineering

    :param reader: data splitter
    :return: return ICM_train_all
    """
    URM_train, _ = reader.get_holdout_split()
    UCM_all_dict = reader.get_loaded_UCM_dict()
    ICM_all_dict = reader.get_loaded_ICM_dict()
    ICM_all_dict.pop("ICM_all")
    ICM_all_dict = apply_feature_engineering_ICM(ICM_all_dict, URM_train, UCM_all_dict,
                                                 ICM_names_to_count=["ICM_sub_class"], UCM_names_to_list=["UCM_age"])
    ICM_all_dict = apply_filtering_ICM(ICM_all_dict,
                                       ICM_name_to_filter_mapper={"ICM_asset": lambda x: x < np.quantile(x, q=0.75) +
                                                                                         0.72 * (np.quantile(x,
                                                                                                             q=0.75) -
                                                                                                 np.quantile(x,
                                                                                                             q=0.25)),
                                                                  "ICM_price": lambda x: x < np.quantile(x, q=0.75) +
                                                                                         4 * (np.quantile(x, q=0.75) -
                                                                                              np.quantile(x, q=0.25))})
    ICM_all_dict = apply_transformation_ICM(ICM_all_dict,
                                            ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                          "ICM_price": lambda x: np.log1p(1 / x),
                                                                          "ICM_item_pop": np.log1p,
                                                                          "ICM_sub_class_count": np.log1p,
                                                                          "ICM_age": lambda x: x ** (1 / 2.5)})
    ICM_all_dict = apply_discretization_ICM(ICM_all_dict,
                                            ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                     "ICM_price": 200,
                                                                     "ICM_item_pop": 50,
                                                                     "ICM_sub_class_count": 50})
    ICM_all = build_ICM_all_from_dict(ICM_all_dict)
    return ICM_all


def get_UCM_all(reader: RecSys2019Reader):
    URM_all = reader.get_URM_all()
    UCM_all_dict = reader.get_loaded_UCM_dict()
    ICM_dict = reader.get_loaded_ICM_dict()
    UCM_all_dict = apply_feature_engineering_UCM(UCM_all_dict, URM_all, ICM_dict,
                                                 ICM_names_to_UCM=["ICM_sub_class"])

    # These are useful feature weighting for UserCBF_CF_Warm
    UCM_all_dict = apply_transformation_UCM(UCM_all_dict,
                                            UCM_name_to_transform_mapper={"UCM_sub_class": lambda x: x / 2,
                                                                          "UCM_user_act": np.log1p})
    UCM_all_dict = apply_discretization_UCM(UCM_all_dict, UCM_name_to_bins_mapper={"UCM_user_act": 50})
    UCM_all = build_UCM_all_from_dict(UCM_all_dict)
    return UCM_all


def get_UCM_train(reader: New_DataSplitter_leave_k_out):
    """
    It returns all the UCM_all after applying feature engineering

    :param reader: data splitter
    :return: return UCM_all
    """
    URM_train, _ = reader.get_holdout_split()
    UCM_all_dict = reader.get_loaded_UCM_dict()
    ICM_dict = reader.get_loaded_ICM_dict()
    UCM_all_dict = apply_feature_engineering_UCM(UCM_all_dict, URM_train, ICM_dict,
                                                 ICM_names_to_UCM=["ICM_sub_class"])

    # These are useful feature weighting for UserCBF_CF_Warm
    UCM_all_dict = apply_transformation_UCM(UCM_all_dict,
                                            UCM_name_to_transform_mapper={"UCM_sub_class": lambda x: x / 2,
                                                                          "UCM_user_act": np.log1p})
    UCM_all_dict = apply_discretization_UCM(UCM_all_dict, UCM_name_to_bins_mapper={"UCM_user_act": 50})
    UCM_all = build_UCM_all_from_dict(UCM_all_dict)
    return UCM_all


def row_split(row_string):
    """
    Helper for splitting the URM
    :param row_string: line of the URM train
    :return: splitted row
    """
    row_string = row_string.replace("\n", "")
    split = row_string.split(",")

    split[0] = int(split[0])
    split[1] = int(split[1])

    result = tuple(split)

    return result


def row_split_target(row_string):
    """
    Function helper to read the target playlist
    :param row_string:
    :return:
    """
    return int(row_string.replace("\n", ""))


def get_warm_user_rating_matrix(user_rating_matrix):
    """
    :param user_rating_matrix: user rating matrix
    :return: warm version of the user rating matrix
    """
    import numpy as np
    warm_items_mask = np.ediff1d(user_rating_matrix.tocsc().indptr) > 0
    warm_items = np.arange(user_rating_matrix.shape[1])[warm_items_mask]

    user_rating_matrix = user_rating_matrix[:, warm_items]

    warm_users_mask = np.ediff1d(user_rating_matrix.tocsr().indptr) > 0
    warm_users = np.arange(user_rating_matrix.shape[0])[warm_users_mask]

    user_rating_matrix = user_rating_matrix[warm_users, :]

    return user_rating_matrix
