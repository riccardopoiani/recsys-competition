import os

from skopt.space import Categorical, Integer, Real

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.DataPreprocessing import DataPreprocessingDigitizeICMs
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, get_UCM_all
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = DataPreprocessingDigitizeICMs(data_reader, ICM_name_to_bins_mapper={"ICM_asset": 50, "ICM_price": 50,
                                                                                      "ICM_item_pop": 20})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_all")
    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_all = get_UCM_all(data_reader.dataReader_object.reader, discretize_user_act_bins=20)
    UCM_train = get_warmer_UCM(UCM_all, URM_all, threshold_users=3)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                  ICM_train=ICM_subclass)
    hyperparameters_range = {}
    for par, value in best_models.ItemCBF_CF.get_best_parameters().items():
        hyperparameters_range[par] = Categorical([value])
    hyperparameters_range['topK'] = Integer(low=3, high=50)
    hyperparameters_range['shrink'] = Integer(low=0, high=2000)
    hyperparameters_range['asymmetric_alpha'] = Real(low=1e-2, high=1e-1, prior="log-uniform")

    model.fit(num_models=100, hyper_parameters_range=hyperparameters_range)

    print(evaluator.evaluateRecommender(model))
