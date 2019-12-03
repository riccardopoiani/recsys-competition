import argparse
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.tuning.run_parameter_search_item_content import run_parameter_search_item_content
from src.utils.general_utility_functions import get_split_seed

N_CASES = 35
RECOMMENDER_CLASS_DICT = {
    "item_cbf_numerical": ItemKNNCBFRecommender,
    "item_cbf_categorical": ItemKNNCBFRecommender,
    "item_cbf_all": ItemKNNCBFRecommender,
    "item_cbf_all_and_URM": ItemKNNCBFRecommender,
    "item_cbf_categorical_and_URM": ItemKNNCBFRecommender
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--reader_path", default="../../data/", help="path to the root of data files")
    parser.add_argument("-r", "--recommender_name", required=True,
                        help="recommender names should be one of: {}".format(list(RECOMMENDER_CLASS_DICT.keys())))
    parser.add_argument("-n", "--n_cases", default=N_CASES, help="number of cases for hyperparameter tuning")
    parser.add_argument("--seed", default=get_split_seed(), help="seed used in splitting the dataset")
    parser.add_argument("-foh", "--focus_on_high", default=0, help="focus the tuning only on users with profile"
                                                                   "lengths larger than the one specified here")
    parser.add_argument("-eu", "--exclude_users", default=False, help="1 to exclude cold users, 0 otherwise")

    return parser.parse_args()


def main():
    args = get_arguments()

    # Data loading
    data_reader = RecSys2019Reader(args.reader_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=args.seed)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)

    if args.recommender_name == "item_cbf_numerical":
        ICM = ICM_numerical
        ICM_name = "ICM_numerical"
        similarity_type_list = ['euclidean']
    elif args.recommender_name == "item_cbf_categorical":
        ICM = ICM_categorical
        ICM_name = "ICM_categorical"
        similarity_type_list = None
    elif args.recommender_name == "item_cbf_all":
        ICM, _ = merge_ICM(ICM_numerical, ICM_categorical, {}, {})
        ICM_name = "ICM_all"
        similarity_type_list = ['euclidean']
    elif args.recommender_name == "item_cbf_all_and_URM":
        ICM, _ = merge_ICM(ICM_numerical, ICM_categorical, {}, {})
        ICM, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})
        ICM_name = "ICM_all_and_URM"
        similarity_type_list = ['euclidean']
    else:
        ICM, _ = merge_ICM(ICM_categorical, URM_train.transpose(), {}, {})
        ICM_name = "ICM_categorical_and_URM"
        similarity_type_list = None

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

    run_parameter_search_item_content(URM_train=URM_train, ICM_object=ICM, ICM_name=ICM_name,
                                      recommender_class=ItemKNNCBFRecommender,
                                      evaluator_validation=evaluator,
                                      metric_to_optimize="MAP",
                                      output_folder_path=version_path,
                                      similarity_type_list=similarity_type_list,
                                      parallelizeKNN=True,
                                      n_cases=int(args.n_cases))
    print("...tuning ended")


if __name__ == '__main__':
    main()
