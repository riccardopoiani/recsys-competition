from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train_new
from src.model import new_best_models
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.tuning.holdout_validation.run_parameter_search_field_weight import run_parameter_search_field_ICM_weight
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build UCMs
    ICM_all, item_feature_to_range_mapper = get_ICM_train_new(data_reader)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    version_path = "../../report/hp_tuning/search_weight_icm/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_field_ICM_weight(URM_train, ICM_all, ItemKNNCBFCFRecommender,
                                          new_best_models.ItemCBF_CF.get_best_parameters(),
                                          item_feature_to_range_mapper,
                                          metric_to_optimize="MAP",
                                          evaluator_validation=evaluator,
                                          output_folder_path=version_path,
                                          n_cases=100, n_random_starts=40)

    print("...tuning ended")
