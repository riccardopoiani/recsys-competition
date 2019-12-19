import os

import numpy as np
import scipy.sparse as sps
from sklearn.model_selection import train_test_split
from xlearn import write_data_to_xlearn_format

from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_preprocessing import apply_feature_engineering_ICM, apply_filtering_ICM, \
    apply_transformation_ICM, apply_discretization_ICM, apply_feature_engineering_UCM, apply_transformation_UCM, \
    apply_discretization_UCM
from src.data_management.data_preprocessing_fm import add_ICM_info, add_UCM_info, \
    convert_URM_to_FM, sample_negative_interactions_uniformly
from src.utils.general_utility_functions import get_split_seed, get_project_root_path


def get_ICM_with_fields(reader: New_DataSplitter_leave_k_out):
    """
    It returns all the ICM_train_all after applying feature engineering

    :param reader: data splitter
    :return: return ICM_train_all and feature fields
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

    ICM_all = None
    item_feature_fields = None
    for idx, ICM_key_value in enumerate(ICM_all_dict.items()):
        ICM_name, ICM_object = ICM_key_value
        if idx == 0:
            ICM_all = ICM_object
            item_feature_fields = np.full(shape=ICM_object.shape[1], fill_value=idx)
        else:
            ICM_all = sps.hstack([ICM_all, ICM_object], format="csr")
            item_feature_fields = np.concatenate([item_feature_fields,
                                                  np.full(shape=ICM_object.shape[1], fill_value=idx)])
    return ICM_all, item_feature_fields


def get_UCM_with_fields(reader: New_DataSplitter_leave_k_out):
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
    UCM_all = None
    user_feature_fields = None
    for idx, UCM_key_value in enumerate(UCM_all_dict.items()):
        UCM_name, UCM_object = UCM_key_value
        if idx == 0:
            UCM_all = UCM_object
            user_feature_fields = np.full(shape=UCM_object.shape[1], fill_value=idx)
        else:
            UCM_all = sps.hstack([UCM_all, UCM_object], format="csr")
            user_feature_fields = np.concatenate([user_feature_fields,
                                                  np.full(shape=UCM_object.shape[1], fill_value=idx)])
    return UCM_all, user_feature_fields


if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_items=-1, threshold_users=50)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True,
                                               seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all, item_feature_fields = get_ICM_with_fields(data_reader)
    UCM_all, user_feature_fields = get_UCM_with_fields(data_reader)

    user_fields = np.full(shape=URM_train.shape[0], fill_value=0)
    item_fields = np.full(shape=URM_train.shape[1], fill_value=1)
    item_feature_fields = item_feature_fields + 2
    user_feature_fields = user_feature_fields + np.max(item_feature_fields) + 1
    fields = np.concatenate([user_fields, item_fields, item_feature_fields, user_feature_fields])

    positive_URM = URM_train
    negative_URM = sample_negative_interactions_uniformly(negative_sample_size=len(positive_URM.data),
                                                          URM=positive_URM)

    URM_positive_FM_matrix = convert_URM_to_FM(positive_URM)
    URM_negative_FM_matrix = convert_URM_to_FM(negative_URM)

    URM_FM_matrix = sps.vstack([URM_positive_FM_matrix, URM_negative_FM_matrix], format='csr')
    URM_FM_matrix = add_ICM_info(URM_FM_matrix, ICM_all, URM_train.shape[0])
    URM_FM_matrix = add_UCM_info(URM_FM_matrix, UCM_all, 0)

    root_path = get_project_root_path()
    fm_data_path = os.path.join(root_path, "resources", "ffm_data")

    # Prepare train sparse matrix and labels for dumping to file
    FM_sps_matrix = URM_FM_matrix.copy()
    labels = np.concatenate([np.ones(shape=URM_positive_FM_matrix.shape[0], dtype=np.int).tolist(),
                             np.zeros(shape=URM_negative_FM_matrix.shape[0], dtype=np.int).tolist()])

    random_state = 69420
    x_train, x_valid, y_train, y_valid = train_test_split(FM_sps_matrix, labels, shuffle=True,
                                                          test_size=0.2, random_state=random_state)

    # Dump libffm file for train set
    print("Writing train and valid dataset in libffm format...")
    train_file_path = os.path.join(fm_data_path, "warm_50_train_uncompressed.txt")
    valid_file_path = os.path.join(fm_data_path, "warm_50_valid_uncompressed.txt")
    write_data_to_xlearn_format(X=x_train, y=y_train, fields=fields, filepath=train_file_path)
    write_data_to_xlearn_format(X=x_valid, y=y_valid, fields=fields, filepath=valid_file_path)
    print("...Writing is over.")
