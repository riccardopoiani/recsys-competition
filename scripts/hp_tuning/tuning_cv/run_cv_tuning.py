import argparse
import multiprocessing
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.IR_feature_weighting import TF_IDF
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython
from course_lib.MatrixFactorization.NMFRecommender import NMFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from scripts.scripts_utils import set_env_variables, read_split_load_data
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.data_reader import get_ICM_train_new, get_UCM_train_new, get_ignore_users
from src.model.KNN.ItemKNNDotCFRecommender import ItemKNNDotCFRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.NewItemKNNCBFRecommender import NewItemKNNCBFRecommender
from src.model.KNN.NewUserKNNCFRecommender import NewUserKNNCFRecommender
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.model.KNN.UserKNNDotCFRecommender import UserKNNDotCFRecommender
from src.model.MatrixFactorization.FunkSVDRecommender import FunkSVDRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.model.MatrixFactorization.NewPureSVDRecommender import NewPureSVDRecommender
from src.tuning.cross_validation.run_cv_parameter_search import run_cv_parameter_search
from src.utils.general_utility_functions import get_split_seed, get_seed_lists, \
    get_root_data_path, str2bool

N_CASES = 100
N_RANDOM_STARTS = 40
N_FOLDS = 5
K_OUT = 1
CUTOFF = 10
MAX_UPPER_THRESHOLD = 2 ** 32 - 1
MIN_LOWER_THRESHOLD = -1

AGE_TO_KEEP = []  # Default []

SIDE_INFO_CLASS_DICT = {
    # Graph-Based
    "rp3beta_side": RP3betaRecommender,

    # ML-Based
    "pure_svd_side": NewPureSVDRecommender,
    "slim_side": SLIM_BPR_Cython
}

COLLABORATIVE_RECOMMENDER_CLASS_DICT = {
    # KNN
    "item_cf": ItemKNNCFRecommender,
    "user_cf": UserKNNCFRecommender,
    "new_user_cf": NewUserKNNCFRecommender,
    "user_dot_cf": UserKNNDotCFRecommender,
    "item_dot_cf": ItemKNNDotCFRecommender,

    # ML Item-Similarity Based
    "slim_bpr": SLIM_BPR_Cython,
    "slim_elastic": SLIMElasticNetRecommender,

    # Graph-based
    "p3alpha": P3alphaRecommender,
    "rp3beta": RP3betaRecommender,

    # Matrix Factorization
    "pure_svd": NewPureSVDRecommender,
    "light_fm": LightFMRecommender,
    "ials": ImplicitALSRecommender,
    "logistic_mf": LogisticMFRecommender,
    "mf_bpr": MF_BPR_Recommender,
    "funk_svd": FunkSVDRecommender,
    "asy_svd": MatrixFactorization_AsySVD_Cython,
    "nmf": NMFRecommender
}
CONTENT_RECOMMENDER_CLASS_DICT = {
    # Pure CBF KNN
    "new_item_cbf": NewItemKNNCBFRecommender,
    "item_cbf_cf": ItemKNNCBFCFRecommender,
    "item_cbf_all": ItemKNNCBFRecommender
}

DEMOGRAPHIC_RECOMMENDER_CLASS_DICT = {
    # Pure Demographic KNN
    "user_cbf": UserKNNCBFRecommender,
    "user_cbf_cf": UserKNNCBFCFRecommender
}

RECOMMENDER_CLASS_DICT = dict(**COLLABORATIVE_RECOMMENDER_CLASS_DICT, **CONTENT_RECOMMENDER_CLASS_DICT,
                              **DEMOGRAPHIC_RECOMMENDER_CLASS_DICT, **SIDE_INFO_CLASS_DICT)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--reader_path", default=get_root_data_path(), help="path to the root of data files")
    parser.add_argument("-r", "--recommender_name", required=True,
                        help="recommender names should be one of: {}".format(list(RECOMMENDER_CLASS_DICT.keys())))
    parser.add_argument("-n", "--n_cases", default=N_CASES, type=int, help="number of cases for hyper parameter tuning")
    parser.add_argument("-f", "--n_folds", default=N_FOLDS, type=int, help="number of folds for cross validation")
    parser.add_argument("-nr", "--n_random_starts", default=N_RANDOM_STARTS, type=int,
                        help="number of random starts for hyper parameter tuning")
    parser.add_argument("-p", "--parallelize", default=1, type=str2bool,
                        help="1 to parallelize the search, 0 otherwise")
    parser.add_argument("-ut", "--upper_threshold", default=MAX_UPPER_THRESHOLD, type=int,
                        help="Upper threshold (included) of user profile length to validate")
    parser.add_argument("-lt", "--lower_threshold", default=MIN_LOWER_THRESHOLD, type=int,
                        help="Lower threshold (included) of user profile length to validate")
    parser.add_argument("-acu", "--allow_cold_users", default=0, type=str2bool, help="1 to allow cold users,"
                                                                                     " 0 otherwise")
    parser.add_argument("-ent", "--exclude_non_target", default=1, type=str2bool,
                        help="1 to exclude non-target users, 0 otherwise")
    parser.add_argument("-nj", "--n_jobs", default=multiprocessing.cpu_count(), help="Number of workers", type=int)
    parser.add_argument("--seed", default=get_split_seed(), help="seed for the experiment", type=int)
    # parser.add_argument("-a", "--age", default=-69420, help="Validate only on the users of this region", type=int)

    return parser.parse_args()


def main():
    set_env_variables()
    args = get_arguments()
    seeds = get_seed_lists(args.n_folds, get_split_seed())

    # --------- DATA LOADING SECTION --------- #
    URM_train_list = []
    ICM_train_list = []
    UCM_train_list = []
    evaluator_list = []
    for fold_idx in range(args.n_folds):
        # Read and split data
        data_reader = read_split_load_data(K_OUT, args.allow_cold_users, seeds[fold_idx])
        URM_train, URM_test = data_reader.get_holdout_split()
        ICM_train, item_feature2range = get_ICM_train_new(data_reader)
        UCM_train, user_feature2range = get_UCM_train_new(data_reader)

        # Ignore users and setting evaluator
        ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                        args.lower_threshold, args.upper_threshold,
                                        ignore_non_target_users=args.exclude_non_target)

        # Ignore users by age
        # UCM_age = data_reader.get_UCM_from_name("UCM_age")
        # age_feature_to_id_mapper = data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
        # age_demographic = get_user_demographic(UCM_age, age_feature_to_id_mapper, binned=True)
        # ignore_users = np.unique(np.concatenate((ignore_users, get_ignore_users_age(age_demographic, AGE_TO_KEEP))))

        URM_train_list.append(URM_train)
        ICM_train_list.append(ICM_train)
        UCM_train_list.append(UCM_train)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=np.unique(ignore_users))
        evaluator_list.append(evaluator)

    # --------- HYPER PARAMETERS TUNING SECTION --------- #
    print("Start tuning...")

    hp_tuning_path = "../../../report/hp_tuning/" + args.recommender_name + "/"
    date_string = datetime.now().strftime('%b%d_%H-%M-%S_k1_lt_{}/'.format(args.lower_threshold))
    output_folder_path = hp_tuning_path + date_string

    if args.recommender_name in COLLABORATIVE_RECOMMENDER_CLASS_DICT.keys():
        run_cv_parameter_search(URM_train_list=URM_train_list,
                                recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                evaluator_validation_list=evaluator_list,
                                metric_to_optimize="MAP", output_folder_path=output_folder_path,
                                parallelize_search=args.parallelize, n_jobs=args.n_jobs,
                                n_cases=args.n_cases, n_random_starts=args.n_random_starts)
    elif args.recommender_name in CONTENT_RECOMMENDER_CLASS_DICT.keys():
        run_cv_parameter_search(URM_train_list=URM_train_list, ICM_train_list=ICM_train_list, ICM_name="ICM_all",
                                recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                evaluator_validation_list=evaluator_list,
                                metric_to_optimize="MAP", output_folder_path=output_folder_path,
                                parallelize_search=args.parallelize, n_jobs=args.n_jobs,
                                n_cases=args.n_cases, n_random_starts=args.n_random_starts)
    elif args.recommender_name in DEMOGRAPHIC_RECOMMENDER_CLASS_DICT.keys():
        run_cv_parameter_search(URM_train_list=URM_train_list, UCM_train_list=UCM_train_list, UCM_name="UCM_all",
                                recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                evaluator_validation_list=evaluator_list,
                                metric_to_optimize="MAP", output_folder_path=output_folder_path,
                                parallelize_search=args.parallelize, n_jobs=args.n_jobs,
                                n_cases=args.n_cases, n_random_starts=args.n_random_starts)
    elif args.recommender_name in SIDE_INFO_CLASS_DICT:
        temp_list = []
        for i, URM in enumerate(URM_train_list):
            temp = sps.vstack([URM, ICM_train_list[i].T], format="csr")
            temp = TF_IDF(temp).tocsr()
            temp_list.append(temp)

        run_cv_parameter_search(URM_train_list=temp_list,
                                recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                evaluator_validation_list=evaluator_list, metric_to_optimize="MAP",
                                output_folder_path=output_folder_path, parallelize_search=args.parallelize,
                                n_jobs=args.n_jobs, n_cases=args.n_cases, n_random_starts=args.n_random_starts)

    print("...tuning ended")


if __name__ == '__main__':
    main()
