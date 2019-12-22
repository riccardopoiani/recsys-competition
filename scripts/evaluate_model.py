import os

import pandas as pd
from skopt.space import Integer, Categorical, Real

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.IR_feature_weighting import TF_IDF
from course_lib.Base.NonPersonalizedRecommender import TopPop
from course_lib.Base.Recommender_utils import similarityMatrixTopK
from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_reader import get_ICM_train, get_UCM_train, get_UCM_train_new, get_ICM_train_new, \
    read_target_users, get_index_target_users, read_URM_cold_all
from src.model import new_best_models, best_models
from src.model.Ensemble.BaggingAverageRecommender import BaggingAverageRecommender
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender, \
    BaggingMergeUserSimilarityRecommender
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.model.GraphBased.P3alphaDemographicRecommender import P3alphaDemographicRecommender
from src.model.GraphBased.RP3betaDemographicRecommender import RP3betaDemographicRecommender
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.NewUserKNNCFRecommender import NewUserKNNCFRecommender
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.plots.recommender_plots import plot_hexbin
from src.utils.general_utility_functions import get_split_seed


def _get_all_models(URM_train, ICM_all, UCM_all):
    all_models = {}

    all_models['MIXED'] = new_best_models.MixedItem.get_model(URM_train, ICM_all)

    all_models['S_SLIM_BPR'] = new_best_models.SSLIM_BPR.get_model(sps.vstack([URM_train, ICM_all.T]))
    all_models['S_PURE_SVD'] = new_best_models.PureSVDSideInfo.get_model(URM_train, ICM_all)
    all_models['S_IALS'] = new_best_models.IALSSideInfo.get_model(URM_train, ICM_all)
    all_models['USER_CBF_CF'] = new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all)

    return all_models


if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    root_data_path = "../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all, _ = get_ICM_train_new(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    #UCM_all = get_UCM_train(data_reader)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0  # < 26 for k1out, # 22 for k3out
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    ignore_users = cold_users

    cold_items_mask = np.ediff1d(URM_train.tocsc().indptr) == 0
    cold_items = np.arange(URM_train.shape[1])[cold_items_mask]

    exclude_non_target_users = False
    if exclude_non_target_users:
        original_target_users = read_target_users("../data/data_target_users_test.csv")
        target_users = get_index_target_users(original_target_users,
                                              data_reader.get_original_user_id_to_index_mapper())
        non_target_users = np.setdiff1d(np.arange(URM_train.shape[0]), target_users, assume_unique=True)
        ignore_users = np.concatenate([ignore_users, non_target_users])

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    """hyper_parameters_range = {}
    par = {'topK': 998, 'shrink': 968, 'similarity': 'cosine', 'normalize': False,
           'feature_weighting': 'BM25', 'interactions_feature_weighting': "TF-IDF"}
    for key, value in par.items():
        hyper_parameters_range[key] = Categorical([value])
    model = BaggingMergeUserSimilarityRecommender(URM_train, UserKNNCBFCFRecommender,
                                                  do_bootstrap=True, UCM_train=UCM_all)
    model.fit(num_models=1, hyper_parameters_range=hyper_parameters_range)"""

    best_par = new_best_models.ItemCBF_CF.get_best_parameters()
    model = ItemKNNCBFCFRecommender(URM_train, ICM_all)
    model.fit(**best_par)

    print(evaluator.evaluateRecommender(model))
