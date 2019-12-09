import os

from skopt.space import Categorical, Integer, Real

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from src.data_management.DataPreprocessing import DataPreprocessingDiscretization, \
    DataPreprocessingImputation, DataPreprocessingFeatureEngineering, DataPreprocessingTransform
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, get_UCM_all
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models, new_best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender, \
    BaggingMergeUserSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender

from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = DataPreprocessingFeatureEngineering(data_reader,
                                                      ICM_names_to_count=["ICM_sub_class"],
                                                      ICM_names_to_UCM=["ICM_sub_class", "ICM_price", "ICM_asset"],
                                                      UCM_names_to_ICM=[])
    data_reader = DataPreprocessingImputation(data_reader,
                                              ICM_name_to_agg_mapper={"ICM_asset": np.median,
                                                                      "ICM_price": np.median})
    data_reader = DataPreprocessingTransform(data_reader,
                                             ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                           "ICM_price": lambda x: np.log1p(1 / x),
                                                                           "ICM_item_pop": np.log1p,
                                                                           "ICM_sub_class_count": np.log1p},
                                             UCM_name_to_transform_mapper={"UCM_price": lambda x: np.log1p(1 / x),
                                                                           "UCM_asset": lambda x: np.log1p(1 / x)})
    data_reader = DataPreprocessingDiscretization(data_reader,
                                                  ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                           "ICM_price": 200,
                                                                           "ICM_item_pop": 50,
                                                                           "ICM_sub_class_count": 50},
                                                  UCM_name_to_bins_mapper={"UCM_price": 200,
                                                                           "UCM_asset": 200,
                                                                           "UCM_user_act": 50})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM_all = data_reader.get_ICM_from_name("ICM_all")

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_all = data_reader.get_UCM_from_name("UCM_all")
    UCM_all = get_warmer_UCM(UCM_all, URM_all, threshold_users=3)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    par = {'user_similarity_type': 'cosine', 'item_similarity_type': 'asymmetric', 'user_feature_weighting': 'BM25',
           'item_feature_weighting': 'TF-IDF', 'user_normalize': True,
           'item_normalize': True, 'item_asymmetric_alpha': 0.1539884061705812,
           'user_topK': 16, 'user_shrink': 1000, 'item_topK': 12, 'item_shrink': 1374}
    model = UserItemCBFCFDemographicRecommender(URM_train, UCM_all, ICM_all)
    model.fit(**par)

    print(evaluator.evaluateRecommender(model))
