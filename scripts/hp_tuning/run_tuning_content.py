import argparse
import multiprocessing
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from scripts.scripts_utils import read_split_load_data
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.data_management.data_reader import get_ICM_train, get_ICM_train_new, get_ignore_users
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.SLIM.SSLIM_BPR import SSLIM_BPR
from src.tuning.holdout_validation.run_parameter_search_item_content import run_parameter_search_item_content
from src.utils.general_utility_functions import get_split_seed, get_root_data_path, str2bool

N_RANDOM_STARTS = 30
N_CASES = 70
MAX_UPPER_THRESHOLD = 2**16-1
MIN_LOWER_THRESHOLD = -1

RECOMMENDER_CLASS_DICT = {
    "item_cbf_numerical": ItemKNNCBFRecommender,
    "item_cbf_categorical": ItemKNNCBFRecommender,
    "item_cbf_all": ItemKNNCBFRecommender,
    "item_cbf_cf_all": ItemKNNCBFCFRecommender,
    "sslim_bpr": SSLIM_BPR
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--reader_path", default=get_root_data_path(), help="path to the root of data files")
    parser.add_argument("-r", "--recommender_name", required=True,
                        help="recommender names should be one of: {}".format(list(RECOMMENDER_CLASS_DICT.keys())))
    parser.add_argument("-n", "--n_cases", default=N_CASES, type=int, help="number of cases for hyper parameter tuning")
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
    parser.add_argument("--seed", default=get_split_seed(), help="seed for the experiment", type=int)

    return parser.parse_args()


def main():
    args = get_arguments()

    # Data loading
    data_reader = read_split_load_data(3, args.allow_cold_users, args.seed)
    URM_train, URM_test = data_reader.get_holdout_split()

    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM_all, _ = get_ICM_train_new(data_reader)

    similarity_type_list = None
    if args.recommender_name == "item_cbf_numerical":
        ICM = ICM_numerical
        ICM_name = "ICM_numerical"
    elif args.recommender_name == "item_cbf_categorical":
        ICM = ICM_categorical
        ICM_name = "ICM_categorical"
    else:
        ICM = ICM_all
        ICM_name = "ICM_all"

    # Setting evaluator
    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=args.lower_threshold, upper_threshold=args.upper_threshold,
                                    ignore_non_target_users=args.exclude_non_target)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=ignore_users)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/{}/".format(args.recommender_name)
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_item_content(URM_train=URM_train, ICM_object=ICM, ICM_name=ICM_name,
                                      recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                      evaluator_validation=evaluator,
                                      metric_to_optimize="MAP",
                                      output_folder_path=version_path,
                                      similarity_type_list=similarity_type_list,
                                      parallelizeKNN=True,
                                      n_cases=args.n_cases,
                                      n_random_starts=args.n_random_starts)
    print("...tuning ended")


if __name__ == '__main__':
    main()
