import os

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.IR_feature_weighting import TF_IDF, okapi_BM_25
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical
from src.data_management.data_getter import get_warmer_UCM
from src.feature.feature_weighting import weight_matrix_by_demographic_popularity, weight_matrix_by_user_profile
from src.model import best_models
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    """
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users = np.arange(URM_train.shape[0])[~cold_users_mask]

    cold_items_mask = np.ediff1d(URM_train.tocsc().indptr) == 0
    cold_items = np.arange(URM_train.shape[1])[cold_items_mask]
    """

    # Setting evaluator
    ignore_users_mask = np.ediff1d(URM_train.tocsr().indptr) < 30
    ignore_users = np.arange(URM_train.shape[0])[ignore_users_mask]
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    UCM_age = get_warmer_UCM(UCM_age, URM_all, threshold_users=3)
    UCM_region = get_warmer_UCM(UCM_region, URM_all, threshold_users=3)
    URM_train = URM_train.astype(np.float32)
    URM_train = weight_matrix_by_user_profile(URM_train, URM_train, "log")

    model = best_models.ItemCBF_CF.get_model(URM_train, ICM_all, load_model=False)
    print(evaluator.evaluateRecommender(model))
