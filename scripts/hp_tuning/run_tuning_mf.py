import argparse
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.NonPersonalizedRecommender import TopPop
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.DataPreprocessing import DataPreprocessingDigitizeICMs
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, merge_UCM, get_UCM_all
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.FactorizationMachine.FactorizationMachineRecommender import FactorizationMachineRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.tuning.run_parameter_search_mf import run_parameter_search_mf_collaborative
from src.utils.general_utility_functions import get_split_seed

import os

N_CASES = 35
N_RANDOM_STARTS = 5
RECOMMENDER_CLASS_DICT = {
    "light_fm": LightFMRecommender,
    "ials": ImplicitALSRecommender,
    "logistic_mf": LogisticMFRecommender,
    "mf_bpr": MF_BPR_Recommender,
    "fm": FactorizationMachineRecommender
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
    if args.discretize:
        data_reader = DataPreprocessingDigitizeICMs(data_reader, ICM_name_to_bins_mapper={"ICM_asset": 50, "ICM_price": 50,
                                                                                      "ICM_item_pop": 20})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=args.seed)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.reader.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.reader.get_UCM_from_name("UCM_region")

    UCM = None
    ICM = None
    UCM_name = ""
    ICM_name = ""
    approximate_recommender = None
    if args.recommender_name == "light_fm":
        UCM, _ = merge_UCM(UCM_age, UCM_region, {}, {})
        UCM = get_warmer_UCM(UCM, URM_all, threshold_users=3)
        ICM = ICM_categorical
        ICM_name = "ICM_categorical"
        UCM_name = "UCM_age_region"

    if args.recommender_name == "fm":
        if not args.discretize:
            raise ValueError("Cannot use FM without discretizing!")
        #ICM_all, _ = merge_ICM(ICM_categorical, URM_train.T, {}, {})
        sub = best_models.UserCF.get_model(URM_train)
        approximate_recommender = sub

        UCM = get_UCM_all(data_reader.dataReader_object.reader, discretize_user_act_bins=20)
        UCM = get_warmer_UCM(UCM, URM_all, threshold_users=3)
        ICM = data_reader.get_ICM_from_name("ICM_all")
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
