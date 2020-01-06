from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train_new, get_ignore_users
from src.model import best_models
from src.tuning.holdout_validation.run_parameter_search_cfw_linalg import run_parameter_search
from src.utils.general_utility_functions import get_split_seed

K_OUT = 3
CUTOFF = 10
ALLOW_COLD_USERS = False
LOWER_THRESHOLD = -1  # Remove users below or equal this threshold (default value: -1)
UPPER_THRESHOLD = 22 # Remove users above or equal this threshold (default value: 2**16-1)
IGNORE_NON_TARGET_USERS = True

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=K_OUT, use_validation_set=True,
                                               force_new_split=True, seed=get_split_seed())

    data_reader.load_data()
    URM_train, URM_valid, URM_test = data_reader.get_holdout_split()
    ICM_all, _ = get_ICM_train_new(data_reader)

    # Setting evaluator
    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD,
                                    ignore_non_target_users=IGNORE_NON_TARGET_USERS)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=ignore_users)

    # Path setting
    print("Start tuning...")
    version_path = "../../report/hp_tuning/cfw_lin_alg_lt_{}/".format(LOWER_THRESHOLD)
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_{}_eval/".format(K_OUT)
    version_path = version_path + now

    item_cf = best_models.ItemCF.get_model(URM_train)

    # Fit ItemKNN best model and get the sparse matrix of the weights
    run_parameter_search(URM_train=URM_train, output_folder_path=version_path,
                         evaluator_test=evaluator,
                         W_sparse_CF=item_cf.W_sparse, ICM_all=ICM_all,
                         n_cases=70, n_random_starts=30)
    print("...tuning ended")
