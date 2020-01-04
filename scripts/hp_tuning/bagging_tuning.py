from datetime import datetime

from skopt.space import Categorical, Integer, Real

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train, get_ignore_users, get_ICM_train_new
from src.model import new_best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeUserSimilarityRecommender, \
    BaggingMergeItemSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.tuning.holdout_validation.run_parameter_search_bagging import run_parameter_search_bagging
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    ICM_all, _ = get_ICM_train_new(data_reader)

    UCM_all = get_UCM_train(data_reader)

    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=-1, upper_threshold=22, ignore_non_target_users=True)

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    version_path = "../../report/hp_tuning/bagging/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    hyper_parameters_range = {}
    for par, value in new_best_models.ItemCBF_CF.get_best_parameters().items():
        hyper_parameters_range[par] = Categorical([value])
    hyper_parameters_range['topK'] = Integer(low=3, high=30)
    hyper_parameters_range['shrink'] = Integer(low=1000, high=2000)

    constructor_kwargs = {}
    constructor_kwargs['recommender_class'] = ItemKNNCBFCFRecommender
    constructor_kwargs['ICM_train'] = ICM_all
    constructor_kwargs['do_bootstrap'] = False

    fit_kwargs = {}
    fit_kwargs['hyper_parameters_range'] = hyper_parameters_range

    run_parameter_search_bagging(BaggingMergeItemSimilarityRecommender, URM_train,
                                 constructor_kwargs, fit_kwargs,
                                 metric_to_optimize="MAP",
                                 evaluator_validation=evaluator,
                                 output_folder_path=version_path,
                                 n_cases=60, n_random_starts=30)

    print("...tuning ended")
