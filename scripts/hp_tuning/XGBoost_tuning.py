from datetime import datetime

import numpy as np

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models
from src.utils.general_utility_functions import get_split_seed

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
    UCM_all = get_UCM_train(data_reader, root_data_path)

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_user = np.arange(URM_train.shape[0])[cold_users_mask]
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_user)

    # XGB SETUP
    main_rec = new_best_models.ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)
    #other_rec = [new_best_models.UserCBF_CF_Warm.get_model(URM_train=URM_train, UCM_train=UCM_all)]

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

    # run_xgb_tuning(user_to_validate=users_to_validate, URM_train=URM_train, main_recommender=main_rec,
    #               recommender_list=[], mapper=mapper, evaluator=evaluator, n_trials=40,
    #               max_iter_per_trial=5000, n_early_stopping=15, output_folder_file="temp.txt")

    print("...tuning ended")
