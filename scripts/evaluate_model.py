import os

from skopt.space import Categorical, Integer, Real

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.DataPreprocessing import DataPreprocessingDigitizeICMs,\
    DataPreprocessingImputationNumericalICMs, DataPreprocessingAddICMsGroupByCounting, DataPreprocessingTransformICMs
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, get_UCM_all
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models, new_best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender

from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = DataPreprocessingAddICMsGroupByCounting(data_reader, ICM_names=["ICM_sub_class"])
    data_reader = DataPreprocessingImputationNumericalICMs(data_reader,
                                                           ICM_name_to_agg_mapper={"ICM_asset": np.median,
                                                                                   "ICM_price": np.median})
    data_reader = DataPreprocessingTransformICMs(data_reader,
                                                 ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1/x),
                                                                               "ICM_price": lambda x: np.log1p(1/x),
                                                                               "ICM_item_pop": np.log1p,
                                                                               "ICM_sub_class_count": np.log1p})
    data_reader = DataPreprocessingDigitizeICMs(data_reader, ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                                      "ICM_price": 200,
                                                                                      "ICM_item_pop": 50,
                                                                                      "ICM_sub_class_count": 20})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM_all = data_reader.get_ICM_from_name("ICM_all")

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age_region = get_UCM_all(data_reader.dataReader_object, with_user_act=False, discretize_user_act_bins=20)
    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    model = BaggingMergeItemSimilarityRecommender(URM_train, SLIM_BPR_Cython, do_bootstrap=False)
    hyperparameters_range = {}
    for par, value in best_models.SLIM_BPR.get_best_parameters().items():
        hyperparameters_range[par] = Categorical([value])
    hyperparameters_range['topK'] = Integer(low=5, high=30)
    hyperparameters_range['lambda_i'] = Real(low=1e-1, high=1e0, prior="log_uniform")
    hyperparameters_range['lambda_j'] = Real(low=1e-6, high=1e-5, prior="log_uniform")
    hyperparameters_range['learning_rate'] = Real(low=1e-6, high=1e-5)
    hyperparameters_range['epochs'] = Integer(low=200, high=500)

    model.fit(num_models=50, hyper_parameters_range=hyperparameters_range)

    print(evaluator.evaluateRecommender(model))
