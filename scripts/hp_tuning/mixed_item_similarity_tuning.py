from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader, merge_ICM, get_ICM_numerical
from src.model import best_models
from src.model.HybridRecommender.HybridMixedSimilarityRecommender import ItemHybridModelRecommender
from src.utils.general_utility_functions import get_split_seed
from src.tuning.run_parameter_search_hybrid_mixed_similarity import run_parameter_search_mixed_similarity_item
import numpy as np

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_all")
    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})
    ICM_subclass_all, _ = merge_ICM(ICM_subclass, URM_train.transpose(), {}, {})

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    cutoff_list = [10]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=cold_users)

    # Path setting
    print("Start tuning...")
    version_path = "../../report/hp_tuning/mixed_item/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    item_cf = best_models.ItemCF.get_model(URM_train, load_model=False)
    item_cbf_cf = best_models.ItemCBF_CF.get_model(URM_train=URM_train, load_model=False, ICM_train=ICM_subclass_all)
    item_cbf_all = best_models.ItemCBF_all.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)

    hybrid = ItemHybridModelRecommender(URM_train)
    hybrid.add_similarity_matrix(item_cf.W_sparse)
    hybrid.add_similarity_matrix(item_cbf_cf.W_sparse)
    hybrid.add_similarity_matrix(item_cbf_all.W_sparse)

    run_parameter_search_mixed_similarity_item(hybrid, URM_train=URM_train, output_folder_path=version_path,
                                               evaluator_validation=evaluator_test, evaluator_test=None,
                                               n_cases=50, n_random_starts=10, metric_to_optimize="MAP")

    print("...tuning ended")
