import argparse
import os
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import best_models
from src.model.FactorizationMachine.FactorizationMachineRecommender import FactorizationMachineRecommender
from src.model.MatrixFactorization.FunkSVDRecommender import FunkSVDRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.tuning.run_parameter_search_mf import run_parameter_search_mf_collaborative
from src.utils.general_utility_functions import get_split_seed

N_CASES = 60
N_RANDOM_STARTS = 20
RECOMMENDER_CLASS_DICT = {
    "light_fm": LightFMRecommender,
    "ials": ImplicitALSRecommender,
    "logistic_mf": LogisticMFRecommender,
    "mf_bpr": MF_BPR_Recommender,
    "fm": FactorizationMachineRecommender,
    "funk_svd": FunkSVDRecommender
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--reader_path", default="../../data/", help="path to the root of data files")
    parser.add_argument("-r", "--recommender_name", required=True,
                        help="recommender names should be one of: {}".format(list(RECOMMENDER_CLASS_DICT.keys())))
    parser.add_argument("-n", "--n_cases", default=N_CASES, help="number of cases for hyperparameter tuning")
    parser.add_argument("-nr", "--n_random_starts", default=N_RANDOM_STARTS,
                        help="number of random starts for hyperparameter tuning")
    parser.add_argument("-d", "--discretize", default=False, help="if true, it will discretize the ICMs")
    parser.add_argument("--seed", default=get_split_seed(), help="seed used in splitting the dataset")
    parser.add_argument("-foh", "--focus_on_high", default=0, help="focus the tuning only on users with profile"
                                                                   "lengths larger than the one specified here")
    parser.add_argument("-eu", "--exclude_users", default=False, help="1 to exclude cold users, 0 otherwise")

    return parser.parse_args()


def main():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    args = get_arguments()

    # Data loading
    data_reader = RecSys2019Reader(args.reader_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=args.seed)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all = get_ICM_train(data_reader)

    # Build UCMs
    UCM_all = get_UCM_train(data_reader)

    UCM = None
    ICM = None
    UCM_name = ""
    ICM_name = ""
    approximate_recommender = None
    if args.recommender_name == "light_fm":
        UCM = UCM_all
        ICM = ICM_all
        ICM_name = "ICM_all"
        UCM_name = "UCM_all"

    if args.recommender_name == "fm":
        if not args.discretize:
            raise ValueError("Cannot use FM without discretizing!")
        sub = best_models.UserCF.get_model(URM_train)
        approximate_recommender = sub

        UCM = UCM_all
        ICM = ICM_all
        ICM_name = "ICM_all"
        UCM_name = "UCM_all"

    # Setting evaluator
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

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)
    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/{}/".format(args.recommender_name)
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_mf_collaborative(URM_train=URM_train, ICM_train=ICM, ICM_name=ICM_name,
                                          UCM_train=UCM, UCM_name=UCM_name,
                                          recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                          approximate_recommender=approximate_recommender,
                                          evaluator_validation=evaluator,
                                          metric_to_optimize="MAP",
                                          output_folder_path=version_path,
                                          n_cases=int(args.n_cases), n_random_starts=int(args.n_random_starts),
                                          save_model="no")
    print("...tuning ended")


if __name__ == '__main__':
    main()
