from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.DataPreprocessing import DataPreprocessingFeatureEngineering, \
    DataPreprocessingImputation, DataPreprocessingTransform, DataPreprocessingDiscretization
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.model import best_models, new_best_models
from src.model.HybridRecommender.HybridMixedSimilarityRecommender import ItemHybridModelRecommender
from src.utils.general_utility_functions import get_split_seed
from src.tuning.run_parameter_search_hybrid_mixed_similarity import run_parameter_search_mixed_similarity_item
import numpy as np

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = DataPreprocessingFeatureEngineering(data_reader,
                                                      ICM_names_to_count=["ICM_sub_class"],
                                                      ICM_names_to_UCM=["ICM_sub_class", "ICM_price", "ICM_asset"],
                                                      UCM_names_to_ICM=["UCM_age", "UCM_region"])
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
                                                                           "UCM_asset": 200})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())

    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all = data_reader.get_ICM_from_name("ICM_all")

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    very_warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 3
    very_warm_users = np.arange(URM_train.shape[0])[very_warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, very_warm_users)))

    cutoff_list = [10]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=cold_users)

    # Path setting
    print("Start tuning...")
    version_path = "../../report/hp_tuning/mixed_item/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    item_cf = best_models.ItemCF.get_model(URM_train, load_model=False)
    item_cbf_cf = new_best_models.FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)
    item_cbf_all = new_best_models.ItemCBF_all.get_model(URM_train=URM_train,
                                                         ICM_train=ICM_all, load_model=False)

    hybrid = ItemHybridModelRecommender(URM_train)
    hybrid.add_similarity_matrix(item_cf.W_sparse)
    hybrid.add_similarity_matrix(item_cbf_cf.W_sparse)
    hybrid.add_similarity_matrix(item_cbf_all.W_sparse)

    run_parameter_search_mixed_similarity_item(hybrid, URM_train=URM_train, output_folder_path=version_path,
                                               evaluator_validation=evaluator_test, evaluator_test=None,
                                               n_cases=50, n_random_starts=20, metric_to_optimize="MAP")

    print("...tuning ended")
