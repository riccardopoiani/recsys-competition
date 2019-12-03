from datetime import datetime

from numpy.random import seed

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.IR_feature_weighting import TF_IDF
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.tuning.run_parameter_search_mf import run_parameter_search_mf_collaborative
import os

from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Set environment variables to avoid warnings
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    # Setting evaluator
    h = 30
    if h != 0:
        print("Excluding users with less than {} interactions".format(h))
        ignore_users_mask = np.ediff1d(URM_train.tocsc().indptr) < h
        ignore_users = np.arange(URM_train.shape[1])[ignore_users_mask]
    else:
        ignore_users = None

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    version_path = "../../report/hp_tuning/ials/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_foh_30/"
    version_path = version_path + "/" + now

    URM_train = URM_train.astype(np.float32)
    URM_train = TF_IDF(URM_train.T).T

    run_parameter_search_mf_collaborative(URM_train=URM_train,
                                          recommender_class=ImplicitALSRecommender,
                                          evaluator_validation=evaluator,
                                          metric_to_optimize="MAP",
                                          output_folder_path=version_path,
                                          n_cases=30, n_random_starts=5, save_model="no")
    print("...tuning ended")
