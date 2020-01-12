import argparse
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_UCM_train, get_ignore_users, get_UCM_train_cold
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.tuning.holdout_validation.run_parameter_search_user_content import run_parameter_search_user_demographic
from src.utils.general_utility_functions import get_split_seed, str2bool, get_root_data_path

N_RANDOM_STARTS = 30
N_CASES = 70
MAX_UPPER_THRESHOLD = 2**16-1
MIN_LOWER_THRESHOLD = -1
K_OUT = 3
ALLOW_COLD_USERS = True

RECOMMENDER_CLASS_DICT = {
    "user_cbf_all": UserKNNCBFRecommender,
    "user_cbf_cf_all": UserKNNCBFCFRecommender
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
    root_data_path = args.reader_path
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=K_OUT, allow_cold_users=ALLOW_COLD_USERS,
                                               use_validation_set=False, force_new_split=True, seed=args.seed)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Remove interactions to users that has len == 1 to URM_train
    len_1_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 1
    len_1_users = np.arange(URM_train.shape[0])[len_1_users_mask]

    URM_train = URM_train.tolil()
    URM_train[len_1_users, :] = 0
    URM_train = URM_train.tocsr()

    # Remove interactions to users that has len == 1 to URM_test
    len_1_users_mask = np.ediff1d(URM_test.tocsr().indptr) == 1
    len_1_users = np.arange(URM_test.shape[0])[len_1_users_mask]

    URM_test = URM_test.tolil()
    URM_test[len_1_users, :] = 0
    URM_test = URM_test.tocsr()

    UCM_all = get_UCM_train_cold(data_reader)

    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=args.lower_threshold, upper_threshold=args.upper_threshold,
                                    ignore_non_target_users=args.exclude_non_target)
    ignore_users = np.concatenate([ignore_users, len_1_users])

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/{}/".format(args.recommender_name)
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_{}/".format(K_OUT)
    version_path = version_path + "/" + now

    run_parameter_search_user_demographic(URM_train=URM_train, UCM_object=UCM_all, UCM_name="UCM_all",
                                          recommender_class=RECOMMENDER_CLASS_DICT[args.recommender_name],
                                          evaluator_validation=evaluator,
                                          metric_to_optimize="MAP",
                                          output_folder_path=version_path,
                                          parallelizeKNN=True,
                                          n_cases=int(args.n_cases),
                                          n_random_starts=int(args.n_random_starts))

    print("...tuning ended")


if __name__ == '__main__':
    main()
