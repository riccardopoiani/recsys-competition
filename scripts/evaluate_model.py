import os

from skopt.space import Integer, Real, Categorical

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.IR_feature_weighting import TF_IDF
from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models, best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeUserSimilarityRecommender, \
    BaggingMergeItemSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.NewUserKNNCFRecommender import NewUserKNNCFRecommender
from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical

from src.model import new_best_models
from src.utils.general_utility_functions import get_split_seed

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
    UCM_all = get_UCM_train(data_reader, root_data_path)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    model = new_best_models.FusionMergeItem_CBF_CF.get_model(URM_train, ICM_all)

    print(evaluator.evaluateRecommender(model))
