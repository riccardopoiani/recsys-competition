from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from scripts.scripts_utils import read_split_load_data
from src.data_management.data_reader import get_UCM_train, get_ignore_users, get_ICM_train_new
from src.model import new_best_models
from src.model.GraphBased.P3alphaWSparseRecommender import P3alphaWSparseRecommender
from src.tuning.holdout_validation.run_parameter_search_p3alpha_wsparse import run_parameter_search_p3alpha_wsparse
from src.utils.general_utility_functions import get_split_seed

K_OUT = 3
CUTOFF = 10
ALLOW_COLD_USERS = False
LOWER_THRESHOLD = -1  # Remove users below or equal this threshold (default value: -1)
UPPER_THRESHOLD = 2 ** 16 - 1  # Remove users above or equal this threshold (default value: 2**16-1)
IGNORE_NON_TARGET_USERS = True

if __name__ == '__main__':
    # Data loading
    data_reader = read_split_load_data(K_OUT, ALLOW_COLD_USERS, get_split_seed())
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all, _ = get_ICM_train_new(data_reader)

    # Build UCMs
    UCM_all = get_UCM_train(data_reader)

    item_cbf = new_best_models.ItemCBF_all_FW.get_model(URM_train, ICM_all)
    item_w_sparse = item_cbf.W_sparse

    user_cbf = new_best_models.UserCBF.get_model(URM_train, UCM_all)
    user_w_sparse = user_cbf.W_sparse

    # Setting evaluator
    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD,
                                    ignore_non_target_users=True)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=ignore_users)

    version_path = "../../report/hp_tuning/p3alpha_wsparse"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_p3alpha_wsparse(P3alphaWSparseRecommender, URM_train, item_w_sparse, user_w_sparse,
                                         metric_to_optimize="MAP",
                                         evaluator_validation=evaluator,
                                         output_folder_path=version_path,
                                         n_cases=60, n_random_starts=20)

    print("...tuning ended")
