from datetime import datetime

from numpy.random import seed

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.KNN.UserSimilarityRecommender import UserSimilarityRecommender
from src.tuning.run_parameter_search_user_similarity_rs import run_parameter_search_user_similarity_rs
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_all, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_all = get_warmer_UCM(UCM_all, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_all, URM_train, {}, {})

    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]

    best_model = best_models.ItemCBF_CF.get_model(URM_train, ICM_train=ICM_all)

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=warm_users)

    version_path = "../../report/hp_tuning/user_similarity_rs/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_user_similarity_rs(UserSimilarityRecommender, best_model, URM_train, UCM_all, UCM_name="UCM_all",
                                            metric_to_optimize="MAP",
                                            evaluator_validation=evaluator,
                                            output_folder_path=version_path,
                                            n_cases=60, n_random_starts=20)
    print("...tuning ended")
