from datetime import datetime

import numpy as np
import pandas as pd

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model.Ensemble.Boosting.boosting_preprocessing import add_label
from src.tuning.run_xgboost_tuning import run_xgb_tuning
from src.utils.general_utility_functions import get_split_seed


def _preprocess_dataframe(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=["index"], inplace=False)
    df = df.sort_values(by="user_id", ascending=True)
    df = df.reset_index()
    df = df.drop(columns=["index"], inplace=False)
    return df


if __name__ == '__main__':
    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all = get_ICM_train(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    UCM_all = get_UCM_train(data_reader)

    # Reading the dataframe
    dataframe_path = "../../boosting_dataframe/"
    train_df = pd.read_csv(dataframe_path + "train_df_20.csv")
    valid_df = pd.read_csv(dataframe_path + "valid_df_20.csv")

    train_df = _preprocess_dataframe(train_df)
    valid_df = _preprocess_dataframe(valid_df)

    print("Retrieving training labels...", end="")
    y_train, non_zero_count, total = add_label(data_frame=train_df, URM_train=URM_train)
    print("Done")

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_user = np.arange(URM_train.shape[0])[cold_users_mask]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=cold_user)
    total_users = np.arange(URM_train.shape[0])
    mask = np.in1d(total_users, cold_user, invert=True)
    users_to_validate = total_users[mask]

    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/boosting/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    run_xgb_tuning(train_df=train_df,
                   valid_df=valid_df,
                   y_train=y_train,
                   non_zero_count=non_zero_count,
                   total=total,
                   URM_train=URM_train,
                   evaluator=evaluator,
                   n_trials=40,
                   max_iter_per_trial=30000, n_early_stopping=500,
                   objective="binary:logistic", parameters=None,
                   cutoff=20,
                   valid_size=0.2,
                   best_model_folder=version_path)

    print("...tuning ended")
