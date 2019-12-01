from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.tuning.run_parameter_search_mf import run_parameter_search_mf_collaborative
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    import os

    os.environ["MKL_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    version_path = "../../report/hp_tuning/mf_bpr/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_mf_collaborative(URM_train=URM_train,
                                          recommender_class=MF_BPR_Recommender,
                                          evaluator_validation=evaluator,
                                          metric_to_optimize="MAP",
                                          output_folder_path=version_path,
                                          n_cases=35, n_random_starts=5, save_model="no")

    print("...tuning ended")
