import argparse
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.DataPreprocessing import DataPreprocessingFeatureEngineering, DataPreprocessingImputation, \
    DataPreprocessingTransform, DataPreprocessingDiscretization
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_getter import get_UCM_train
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.tuning.run_parameter_search_user_content import run_parameter_search_user_demographic
from src.utils.general_utility_functions import get_split_seed

N_CASES = 60
N_RANDOM_STARTS = 20
RECOMMENDER_CLASS_DICT = {
    "user_cbf_all": UserKNNCBFRecommender,
    "user_cbf_cf_all": UserKNNCBFCFRecommender
}

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--reader_path", default="../../data/", help="path to the root of data files")
    parser.add_argument("-r", "--recommender_name", required=True,
                        help="recommender names should be one of: {}".format(list(RECOMMENDER_CLASS_DICT.keys())))
    parser.add_argument("-n", "--n_cases", default=N_CASES, help="number of cases for hyperparameter tuning")
    parser.add_argument("-nr", "--n_random_starts", default=N_RANDOM_STARTS, help="If true, the search is random")
    parser.add_argument("--seed", default=get_split_seed(), help="seed used in splitting the dataset")
    parser.add_argument("-foh", "--focus_on_high", default=-1, help="focus the tuning only on users with profile"
                                                                   "lengths larger than the one specified here")
    parser.add_argument("-eu", "--exclude_users", default=False, help="1 to exclude cold users, 0 otherwise")
    parser.add_argument("-fol", "--focus_on_low", default=-1, help="focus the tuning only on users with profile"
                                                                  "lengths smaller than the one specified here")

    return parser.parse_args()

def main():
    args = get_arguments()

    # Data loading
    root_data_path = args.reader_path
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = DataPreprocessingFeatureEngineering(data_reader,
                                                      ICM_names_to_count=["ICM_sub_class"])
    data_reader = DataPreprocessingImputation(data_reader,
                                              ICM_name_to_agg_mapper={"ICM_asset": np.median,
                                                                      "ICM_price": np.median})
    data_reader = DataPreprocessingTransform(data_reader,
                                             ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                           "ICM_price": lambda x: np.log1p(1 / x),
                                                                           "ICM_item_pop": np.log1p,
                                                                           "ICM_sub_class_count": np.log1p})
    data_reader = DataPreprocessingDiscretization(data_reader,
                                                  ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                           "ICM_price": 200,
                                                                           "ICM_item_pop": 50,
                                                                           "ICM_sub_class_count": 50})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=args.seed)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    UCM_all = get_UCM_train(data_reader, root_data_path)

    exclude_cold_users = args.exclude_users
    h = int(args.focus_on_high)
    fol = int(args.focus_on_low)
    if h >= 0:
        print("Excluding users with less than {} interactions".format(h))
        ignore_users_mask = np.ediff1d(URM_train.tocsr().indptr) < h
        ignore_users = np.arange(URM_train.shape[0])[ignore_users_mask]
    elif fol >= 0:
        print("Excluding users with more than {} interactions".format(fol))
        warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > fol
        ignore_users = np.arange(URM_train.shape[0])[warm_users_mask]
        if exclude_cold_users:
            cold_user_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
            cold_users = np.arange(URM_train.shape[0])[cold_user_mask]
            ignore_users = np.unique(np.concatenate((cold_users, ignore_users)))
    elif exclude_cold_users:
        print("Excluding cold users...")
        cold_user_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        ignore_users = np.arange(URM_train.shape[0])[cold_user_mask]
    else:
        ignore_users = None

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/{}/".format(args.recommender_name)
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
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