import os

from skopt.space import Categorical, Integer

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
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
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all = get_ICM_train(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    UCM_all = get_UCM_train(data_reader)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 3
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    hyper_parameters_range = {}
    for par, value in new_best_models.ItemCBF_CF.get_best_parameters().items():
        hyper_parameters_range[par] = Categorical([value])
    hyper_parameters_range['topK'] = Integer(low=3, high=30)
    hyper_parameters_range['shrink'] = Integer(low=1000, high=2000)
    model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                  ICM_train=ICM_all)
    model.fit(topK=100, num_models=100, hyper_parameters_range=hyper_parameters_range)

    print(evaluator.evaluateRecommender(model))
