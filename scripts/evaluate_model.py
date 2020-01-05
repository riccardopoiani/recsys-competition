import os

from skopt.space import Categorical, Integer

from course_lib.Base.Evaluation.Evaluator import *
from scripts.scripts_utils import set_env_variables
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_UCM_train, get_ICM_train_new, \
    get_ignore_users
from src.model import new_best_models
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.ItemKNNDotCFRecommender import ItemKNNDotCFRecommender
from src.model.KNN.UserKNNDotCFRecommender import UserKNNDotCFRecommender
from src.utils.general_utility_functions import get_split_seed, get_project_root_path

K_OUT = 1
CUTOFF = 10
ALLOW_COLD_USERS = False
LOWER_THRESHOLD = -1  # Remove users below or equal this threshold (default value: -1)
UPPER_THRESHOLD = 22  # Remove users above or equal this threshold (default value: 2**16-1)
IGNORE_NON_TARGET_USERS = True


def get_model(URM_train, ICM_train, UCM_train):
    # Write the model that you want to evaluate here. Possibly, do not modify the code if unnecessary in the main
    model = UserKNNDotCFRecommender(URM_train)
    model.fit(**{'topK': 989, 'shrink': 1228, 'normalize': True, 'feature_weighting': 'BM25'})
    return model


def main():
    # Data loading
    root_data_path = os.path.join(get_project_root_path(), "data/")
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=K_OUT, use_validation_set=False,
                                               allow_cold_users=ALLOW_COLD_USERS,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all, _ = get_ICM_train_new(data_reader)
    UCM_all = get_UCM_train(data_reader)

    # Ignoring users
    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD,
                                    ignore_non_target_users=IGNORE_NON_TARGET_USERS)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=ignore_users)

    # Model evaluation
    model = get_model(URM_train, ICM_all, UCM_all)
    print(evaluator.evaluateRecommender(model))


if __name__ == '__main__':
    set_env_variables()
    main()
