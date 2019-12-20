import argparse
import os
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython
from course_lib.MatrixFactorization.NMFRecommender import NMFRecommender
from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model.MatrixFactorization.FunkSVDRecommender import FunkSVDRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.tuning.cross_validation.run_cv_parameter_search_collaborative import run_cv_parameter_search_collaborative
from src.utils.general_utility_functions import get_split_seed, get_project_root_path, get_seed_lists

N_CASES = 60
N_RANDOM_STARTS = 20
N_FOLDS = 5
K_OUT = 1
CUTOFF = 10
RECOMMENDER_CLASS_DICT = {
    # KNN
    "item_cf": ItemKNNCFRecommender,
    "user_cf": UserKNNCFRecommender,

    # ML SIMILARITY BASED
    "slim_bpr": SLIM_BPR_Cython,
    "slim_elastic": SLIMElasticNetRecommender,
    "p3alpha": P3alphaRecommender,
    "rp3beta": RP3betaRecommender,

    # Matrix Factorization
    "pure_svd": PureSVDRecommender,
    "light_fm": LightFMRecommender,
    "ials": ImplicitALSRecommender,
    "logistic_mf": LogisticMFRecommender,
    "mf_bpr": MF_BPR_Recommender,
    "funk_svd": FunkSVDRecommender,
    "asy_svd": MatrixFactorization_AsySVD_Cython,
    "nmf": NMFRecommender
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--reader_path", default="../../data/", help="path to the root of data files")
    parser.add_argument("-r", "--recommender_name", required=True,
                        help="recommender names should be one of: {}".format(list(RECOMMENDER_CLASS_DICT.keys())))
    parser.add_argument("-n", "--n_cases", default=N_CASES, help="number of cases for hyperparameter tuning")
    parser.add_argument("-f", "--n_folds", default=N_FOLDS, help="number of folds for cross validation")
    parser.add_argument("-nr", "--n_random_starts", default=N_RANDOM_STARTS,
                        help="number of random starts for hyperparameter tuning")
    parser.add_argument("-d", "--discretize", default=False, help="if true, it will discretize the ICMs")
    parser.add_argument("--seed", default=get_split_seed(), help="seed used in splitting the dataset")
    parser.add_argument("-foh", "--focus_on_high", default=0, help="focus the tuning only on users with profile"
                                                                   "lengths larger than the one specified here")
    parser.add_argument("-eu", "--exclude_users", default=False, help="1 to exclude cold users, 0 otherwise")

    return parser.parse_args()


def set_env_variables():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def main():
    set_env_variables()
    args = get_arguments()
    seeds = get_seed_lists(N_FOLDS, get_split_seed())

    # --------- DATA LOADING --------- #
    data_reader = RecSys2019Reader(args.reader_path)
    URM_train_list = []
    evaluator_list = []
    for fold_idx in range(args.n_folds):
        data_splitter = New_DataSplitter_leave_k_out(data_reader, k_out_value=K_OUT, use_validation_set=False,
                                                     force_new_split=True, seed=seeds[fold_idx])
        data_splitter.load_data()
        URM_train, URM_test = data_splitter.get_holdout_split()

        # Setting evaluator with ignore users
        exclude_cold_users = args.exclude_users
        h = int(args.focus_on_high)
        if h != 0:
            print("Excluding users with less than {} interactions".format(h))
            ignore_users_mask = np.ediff1d(URM_train.tocsr().indptr) < h
            ignore_users = np.arange(URM_train.shape[1])[ignore_users_mask]
        elif exclude_cold_users:
            print("Excluding cold users...")
            cold_user_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
            ignore_users = np.arange(URM_train.shape[0])[cold_user_mask]
        else:
            ignore_users = None

        cutoff_list = [CUTOFF]
        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

        URM_train_list.append(URM_train)
        evaluator_list.append(evaluator)

    # --------- HYPER PARAMETERS TUNING SECTION --------- #
    print("Start tuning...")

    hp_tuning_path = os.path.join(get_project_root_path(), "report", "hp_tuning", "{}".format(args.recommender_name))
    date_string = datetime.now().strftime('%b%d_%H-%M-%S_keep1out/')
    output_folder_path = os.path.join(hp_tuning_path, date_string)

    run_cv_parameter_search_collaborative(URM_train_list=URM_train_list,
                                          recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                          evaluator_validation_list=evaluator_list,
                                          metric_to_optimize="MAP",
                                          output_folder_path=output_folder_path,
                                          n_cases=int(args.n_cases), n_random_starts=int(args.n_random_starts))
    print("...tuning ended")


if __name__ == '__main__':
    main()
