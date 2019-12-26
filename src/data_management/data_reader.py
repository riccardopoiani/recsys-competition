import os

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_preprocessing import apply_feature_engineering_ICM, apply_filtering_ICM, \
    apply_transformation_ICM, apply_discretization_ICM, build_ICM_all_from_dict, apply_feature_engineering_UCM, \
    apply_transformation_UCM, apply_discretization_UCM, build_UCM_all_from_dict, apply_imputation_ICM, \
    apply_imputation_UCM, apply_feature_entropy_UCM

import scipy.sparse as sps
import numpy as np

from src.utils.general_utility_functions import get_project_root_path


# -------- COLD DATA MATRICES ALL --------
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


# -------- GET ITEM CONTENT MATRIX --------
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
    It returns all the ICM_train_all after applying feature engineering. This preprocessing is used on new_best_models
    file

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


def get_ICM_train_new(reader: New_DataSplitter_leave_k_out):
    """
    It returns all the ICM_train_all after applying feature engineering.

    :param reader: data splitter
    :return: return ICM_train_all
    """
    ICM_all_dict = reader.get_loaded_ICM_dict()
    ICM_all_dict.pop("ICM_all")
    ICM_all_dict = apply_filtering_ICM(ICM_all_dict,
                                       ICM_name_to_filter_mapper={"ICM_asset": lambda x: x < np.quantile(x, q=0.75) +
                                                                                         0.72 * (np.quantile(x,
                                                                                                             q=0.75) -
                                                                                                 np.quantile(x,
                                                                                                             q=0.25)),
                                                                  "ICM_price": lambda x: x < np.quantile(x, q=0.75) +
                                                                                         4 * (np.quantile(x, q=0.75) -
                                                                                              np.quantile(x, q=0.25))})
    # Apply useful transformation
    ICM_all_dict = apply_transformation_ICM(ICM_all_dict,
                                            ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                          "ICM_price": lambda x: np.log1p(1 / x),
                                                                          "ICM_item_pop": np.log1p})
    ICM_all_dict = apply_discretization_ICM(ICM_all_dict,
                                            ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                     "ICM_price": 200,
                                                                     "ICM_item_pop": 50})
    # Apply feature weighting
    ICM_all_dict = apply_transformation_ICM(ICM_all_dict,
                                            ICM_name_to_transform_mapper={"ICM_price": lambda x: x * 1.8474248499810804,
                                                                          "ICM_asset": lambda x: x * 1.2232716972721878,
                                                                          "ICM_sub_class": lambda
                                                                              x: x * 1.662671860026709,
                                                                          "ICM_item_pop": lambda
                                                                              x: x * 0.886528360392298})

    ICM_all = None
    item_feature_to_range_mapper = {}
    last_range = 0
    for idx, ICM_key_value in enumerate(ICM_all_dict.items()):
        ICM_name, ICM_object = ICM_key_value
        if idx == 0:
            ICM_all = ICM_object
        else:
            ICM_all = sps.hstack([ICM_all, ICM_object], format="csr")
        item_feature_to_range_mapper[ICM_name] = (last_range, last_range + ICM_object.shape[1])
        last_range = last_range + ICM_object.shape[1]
    return ICM_all, item_feature_to_range_mapper


# -------- GET USER CONTENT MATRIX --------
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
    It returns all the UCM_all after applying feature engineering. This preprocessing is used on new_best_models file

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


def get_UCM_train_new(reader: New_DataSplitter_leave_k_out):
    URM_train, _ = reader.get_holdout_split()
    UCM_all_dict = reader.get_loaded_UCM_dict()
    ICM_dict = reader.get_loaded_ICM_dict()

    # Preprocess ICM
    ICM_dict.pop("ICM_all")
    ICM_dict = apply_feature_engineering_ICM(ICM_dict, URM_train, UCM_all_dict,
                                             ICM_names_to_count=["ICM_sub_class"], UCM_names_to_list=["UCM_age"])
    ICM_dict = apply_filtering_ICM(ICM_dict,
                                   ICM_name_to_filter_mapper={"ICM_asset": lambda x: x < np.quantile(x, q=0.75) +
                                                                                     0.72 * (np.quantile(x,
                                                                                                         q=0.75) -
                                                                                             np.quantile(x,
                                                                                                         q=0.25)),
                                                              "ICM_price": lambda x: x < np.quantile(x, q=0.75) +
                                                                                     4 * (np.quantile(x, q=0.75) -
                                                                                          np.quantile(x, q=0.25))})
    ICM_dict = apply_transformation_ICM(ICM_dict,
                                        ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                      "ICM_price": lambda x: np.log1p(1 / x),
                                                                      "ICM_item_pop": np.log1p,
                                                                      "ICM_sub_class_count": np.log1p,
                                                                      "ICM_age": lambda x: x ** (1 / 2.5)})
    ICM_dict = apply_discretization_ICM(ICM_dict,
                                        ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                 "ICM_price": 200,
                                                                 "ICM_item_pop": 50,
                                                                 "ICM_sub_class_count": 50})

    # Preprocess UCM
    UCM_all_dict = apply_feature_engineering_UCM(UCM_all_dict, URM_train, ICM_dict,
                                                 ICM_names_to_UCM=["ICM_sub_class", "ICM_item_pop"])
    UCM_all_dict = apply_feature_entropy_UCM(UCM_all_dict, UCM_names_to_entropy=["UCM_sub_class"])
    # Apply useful transformation
    UCM_all_dict = apply_transformation_UCM(UCM_all_dict,
                                            UCM_name_to_transform_mapper={"UCM_user_act": np.log1p})

    UCM_all_dict = apply_discretization_UCM(UCM_all_dict, UCM_name_to_bins_mapper={"UCM_user_act": 50,
                                                                                   "UCM_sub_class_entropy": 20})

    # Apply feature weighting TODO

    UCM_all = None
    user_feature_to_range_mapper = {}
    last_range = 0
    for idx, UCM_key_value in enumerate(UCM_all_dict.items()):
        UCM_name, UCM_object = UCM_key_value
        if idx == 0:
            UCM_all = UCM_object
        else:
            UCM_all = sps.hstack([UCM_all, UCM_object], format="csr")
        user_feature_to_range_mapper[UCM_name] = (last_range, last_range + UCM_object.shape[1])
        last_range = last_range + UCM_object.shape[1]
    return UCM_all, user_feature_to_range_mapper


# -------- GET SPECIFIC USERS --------
def read_target_users(path="../data/data_target_users_test.csv"):
    """
    :return: list of user to recommend in the target users
    """

    def row_split_target(row_string):
        """
        Function helper to read the target users
        :param row_string:
        :return:
        """
        return int(row_string.replace("\n", ""))

    target_file = open(path, 'r')

    target_file.seek(0)
    target_tuple = []

    for line in target_file:
        if line != "user_id\n":
            target_tuple.append(row_split_target(line))

    return target_tuple


def get_index_target_users(original_target_users, original_user_id_to_index_mapper):
    """
    Retrieve the target user inside the URM using its original_user_id_to_index_mapper

    :param original_target_users: target users of Kaggle test set
    :param original_user_id_to_index_mapper: mapper referred to URM
    :return: target users of URM
    """
    original_ids = np.array(list(original_user_id_to_index_mapper.keys()), dtype=np.int)
    index_ids = np.array(list(original_user_id_to_index_mapper.values()), dtype=np.int)
    original_ids_mask = np.in1d(original_ids, original_target_users, assume_unique=True)
    target_users = index_ids[original_ids_mask]
    return target_users


def get_users_outside_profile_len(URM_train, lower_threshold, upper_threshold):
    n_interactions_per_user = np.ediff1d(URM_train.tocsr().indptr)
    ignore_users_mask = np.logical_or(n_interactions_per_user <= lower_threshold,
                                      n_interactions_per_user >= upper_threshold)
    return np.arange(URM_train.shape[0])[ignore_users_mask]


def get_ignore_users(URM_train, original_user_id_to_index_mapper, lower_threshold, upper_threshold,
                     ignore_non_target_users=True):
    data_path = os.path.join(get_project_root_path(), "data/")
    ignore_users = []
    users_outside = get_users_outside_profile_len(URM_train, lower_threshold=lower_threshold,
                                                  upper_threshold=upper_threshold)
    if len(users_outside) > 0:
        print("Excluding users with profile length outside ({}, {})".format(lower_threshold, upper_threshold))
        ignore_users = np.concatenate([ignore_users, users_outside])
    if ignore_non_target_users:
        print("Excluding non-target users...")
        original_target_users = read_target_users(os.path.join(data_path, "data_target_users_test.csv"))
        target_users = get_index_target_users(original_target_users,
                                              original_user_id_to_index_mapper)
        non_target_users = np.setdiff1d(np.arange(URM_train.shape[0]), target_users, assume_unique=True)
        ignore_users = np.concatenate([ignore_users, non_target_users])
    return np.unique(ignore_users)


def get_users_of_age(age_demographic, age_list):
    """
    Get users of a certain age

    :param age_demographic: retrieved with get_demographic
    :param age_list: users of these ages will be kept
    :return: np array containing users with age in age_list
    """
    if len(age_list) == 0:
        return np.array([])

    age_demographic_describer_list = np.array([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Finding ages that needs to be excluded (the ones that are not in age_list
    age_to_exclude = np.in1d(age_demographic_describer_list, np.array(age_list), invert=False)
    age_to_exclude = age_demographic_describer_list[age_to_exclude]
    users_to_ignore = []

    for a in age_to_exclude:
        # find_index
        pos = np.argwhere(age_demographic_describer_list == a)[0][0]
        users_to_ignore.extend(age_demographic[pos])

    return users_to_ignore


def get_ignore_users_age(age_demographic, age_list):
    """
    Collect all the users with age different from the one in age_list

    :param age_demographic: retrieved with get_demographic
    :param age_list: list of age of users that will be kept
    :return: np.array containing users with age not in age_list
    """
    if len(age_list) == 0:
        return np.array([])

    age_demographic_describer_list = np.array([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Finding ages that needs to be excluded (the ones that are not in age_list
    age_to_exclude = np.in1d(age_demographic_describer_list, np.array(age_list), invert=True)
    age_to_exclude = age_demographic_describer_list[age_to_exclude]
    users_to_ignore = []

    for a in age_to_exclude:
        # find_index
        pos = np.argwhere(age_demographic_describer_list == a)[0][0]
        users_to_ignore.extend(age_demographic[pos])

    return users_to_ignore
