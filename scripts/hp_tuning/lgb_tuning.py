from datetime import datetime

import numpy as np
import pandas as pd

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ignore_users
from src.model.Ensemble.Boosting.boosting_preprocessing import get_label_array, preprocess_dataframe_after_reading
from src.tuning.holdout_validation.run_parameter_search_lightgbm import run_parameter_search_lightgbm
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Reading the dataframe
    dataframe_path = "../../resources/boosting_dataframe/"
    train_df = pd.read_csv(dataframe_path + "train_df_80_advanced_lt_20.csv")
    valid_df = pd.read_csv(dataframe_path + "valid_df_10_advanced_lt_20.csv")

    train_df = preprocess_dataframe_after_reading(train_df)
    #y_train = train_df['label'].values
    train_df = train_df.drop(columns=["label"], inplace=False)
    valid_df = preprocess_dataframe_after_reading(valid_df)

    y_train, non_zero_count, total = get_label_array(data_frame=train_df, URM_train=URM_train)
    y_valid, _, _ = get_label_array(data_frame=valid_df, URM_train=URM_test)

    # Setting evaluator
    mapper = data_reader.get_original_user_id_to_index_mapper()
    ignore_users = get_ignore_users(URM_train, mapper, lower_threshold=20, upper_threshold=2 ** 16 - 1,
                                    ignore_non_target_users=True)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=ignore_users)
    total_users = np.arange(URM_train.shape[0])
    mask = np.in1d(total_users, ignore_users, invert=True)
    users_to_validate = total_users[mask]

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/light_gbm/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    run_parameter_search_lightgbm(URM_train, train_df, y_train, valid_df, y_valid, cutoff_test=10,
                                  categorical_features=["subclass"],
                                  num_iteration=10000, early_stopping_iteration=150,
                                  objective="lambdarank", verbose=True,
                                  output_folder_path=version_path, evaluator_validation=evaluator,
                                  n_cases=100, n_random_starts=40, metric_to_optimize="MAP")

    print("...tuning ended")
