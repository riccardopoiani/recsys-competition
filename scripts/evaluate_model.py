from course_lib.Base.Evaluation.Evaluator import *
from scripts.scripts_utils import set_env_variables, read_split_load_data
from src.data_management.data_reader import get_UCM_train, get_ICM_train_new, \
    get_ignore_users
from src.model import new_best_models
from src.model.HybridRecommender.HybridRerankingRecommender import HybridRerankingRecommender
from src.utils.general_utility_functions import get_split_seed

K_OUT = 3
CUTOFF = 10
ALLOW_COLD_USERS = False
LOWER_THRESHOLD = -1  # Remove users below or equal this threshold (default value: -1)
UPPER_THRESHOLD = 2**16-1  # Remove users above or equal this threshold (default value: 2**16-1)
IGNORE_NON_TARGET_USERS = True


def get_model(URM_train, ICM_train, UCM_train):
    # Write the model that you want to evaluate here. Possibly, do not modify the code if unnecessary in the main
    model = new_best_models.ItemCBF_all.get_model(URM_train, ICM_train)
    return model


def main():
    # Data loading
    data_reader = read_split_load_data(K_OUT, ALLOW_COLD_USERS, get_split_seed())
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
