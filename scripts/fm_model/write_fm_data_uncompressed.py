import os

import numpy as np
import scipy.sparse as sps
from sklearn.model_selection import train_test_split
from xlearn import write_data_to_xlearn_format

from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_preprocessing_fm import add_ICM_info, add_UCM_info, \
    sample_negative_interactions_uniformly, convert_URM_to_FM
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.utils.general_utility_functions import get_split_seed, get_project_root_path

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_items=-1, threshold_users=25)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True,
                                               seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all = get_ICM_train(data_reader)
    UCM_all = get_UCM_train(data_reader)

    positive_URM = URM_train
    negative_URM = sample_negative_interactions_uniformly(negative_sample_size=len(positive_URM.data), URM=positive_URM)

    URM_positive_FM_matrix = convert_URM_to_FM(positive_URM)
    URM_negative_FM_matrix = convert_URM_to_FM(negative_URM)

    URM_FM_matrix = sps.vstack([URM_positive_FM_matrix, URM_negative_FM_matrix], format='csr')
    URM_FM_matrix = add_ICM_info(URM_FM_matrix, ICM_all, URM_train.shape[0])
    URM_FM_matrix = add_UCM_info(URM_FM_matrix, UCM_all, 0)

    root_path = get_project_root_path()
    fm_data_path = os.path.join(root_path, "resources", "fm_data")

    # Prepare train sparse matrix and labels for dumping to file
    FM_sps_matrix = URM_FM_matrix.copy()
    labels = np.concatenate([np.ones(shape=URM_positive_FM_matrix.shape[0], dtype=np.int).tolist(),
                             np.zeros(shape=URM_negative_FM_matrix.shape[0], dtype=np.int).tolist()])

    random_state = 69420
    x_train, x_valid, y_train, y_valid = train_test_split(FM_sps_matrix, labels, shuffle=True,
                                                          test_size=0.2, random_state=random_state)

    # Dump libffm file for train set
    print("Writing train and valid dataset in libsvm format...")
    train_file_path = os.path.join(fm_data_path, "warm_25_train_uncompressed.txt")
    valid_file_path = os.path.join(fm_data_path, "warm_25_valid_uncompressed.txt")
    write_data_to_xlearn_format(X=x_train, y=y_train, fields=None, filepath=train_file_path)
    write_data_to_xlearn_format(X=x_valid, y=y_valid, fields=None, filepath=valid_file_path)
    print("...Writing is over.")
