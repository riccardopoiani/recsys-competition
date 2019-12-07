from datetime import datetime

from skopt.space import Categorical, Integer, Real

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_UCM_all
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.Ensemble.BaggingMergeSimilarityRecommender import BaggingMergeItemSimilarityRecommender, \
    BaggingMergeUserSimilarityRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.tuning.run_parameter_search_bagging import run_parameter_search_bagging
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    URM_all = data_reader.dataReader_object.get_URM_all()

    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_subclass_URM, _ = merge_ICM(ICM_subclass, URM_train.transpose(), {}, {})

    UCM_all = get_UCM_all(data_reader.dataReader_object, discretize_user_act_bins=20)
    UCM_train = get_warmer_UCM(UCM_all, URM_all, threshold_users=3)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    version_path = "../../report/hp_tuning/bagging/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    hyperparameters_range = {}
    hyperparameters_range['similarity'] = Categorical(['asymmetric'])
    hyperparameters_range['normalize'] = Categorical([True])
    hyperparameters_range['feature_weighting'] = Categorical(["TF-IDF"])
    hyperparameters_range['topK'] = Integer(low=1000, high=2000)
    hyperparameters_range['shrink'] = Integer(low=0, high=200)
    hyperparameters_range['asymmetric_alpha'] = Real(low=0, high=1e-8)

    constructor_kwargs = {}
    constructor_kwargs['recommender_class'] = UserKNNCBFRecommender
    constructor_kwargs['UCM_train'] = UCM_train

    fit_kwargs = {}
    fit_kwargs['hyper_parameters_range'] = hyperparameters_range

    run_parameter_search_bagging(BaggingMergeUserSimilarityRecommender, URM_train,
                                 constructor_kwargs, fit_kwargs,
                                 metric_to_optimize="MAP",
                                 evaluator_validation=evaluator,
                                 output_folder_path=version_path,
                                 n_cases=35)

    print("...tuning ended")
