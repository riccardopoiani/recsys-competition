import os

import numpy as np
import xlearn as xl

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_preprocessing_fm import format_URM_positive_non_compressed, \
    format_URM_negative_sampling_non_compressed, uniform_sampling_strategy, mix_URM, add_ICM_info, add_UCM_info
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.utils.general_utility_functions import get_split_seed, get_project_root_path

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True,
                                               seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all = get_ICM_train(data_reader)
    UCM_all = get_UCM_train(data_reader)

    URM_positive_FM_matrix = format_URM_positive_non_compressed(URM_train)
    URM_negative_FM_matrix = format_URM_negative_sampling_non_compressed(URM_train, negative_rate=1,
                                                                         sampling_function=uniform_sampling_strategy,
                                                                         check_replacement=True)
    URM_FM_matrix = mix_URM(URM_positive_FM_matrix, URM_negative_FM_matrix)[:, :-1]  # remove rating list
    URM_FM_matrix = add_ICM_info(URM_FM_matrix, ICM_all, URM_train.shape[0])
    URM_FM_matrix = add_UCM_info(URM_FM_matrix, UCM_all, 0)

    root_path = get_project_root_path()
    fm_data_path = os.path.join(root_path, "resources", "fm_data")

    # Prepare train sparse matrix and labels for dumping to file
    FM_sps_matrix = URM_FM_matrix.copy()
    FM_sps_matrix[FM_sps_matrix == -1] = 1
    labels = np.concatenate([np.ones(shape=URM_positive_FM_matrix.shape[0], dtype=np.int).tolist(),
                             np.zeros(shape=URM_negative_FM_matrix.shape[0], dtype=np.int).tolist()])

    # Shuffle data
    index = np.arange(FM_sps_matrix.shape[0])
    np.random.shuffle(index)
    FM_sps_matrix = FM_sps_matrix[index, :]
    labels = labels[index]

    # Dump libsvm file for train set
    train_file_path = os.path.join(fm_data_path, "ICM_UCM_uncompressed.txt")
    xl.dump_svmlight_file(X=FM_sps_matrix, y=labels, f=train_file_path)
