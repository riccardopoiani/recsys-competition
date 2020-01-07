import os
from datetime import datetime

from course_lib.Base.BaseRecommender import BaseRecommender
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from scripts.model_selection.cross_validate_utils import get_seed_list, write_results_on_file
from scripts.scripts_utils import read_split_load_data
from src.data_management.data_reader import get_UCM_train, get_ICM_train_new, get_ignore_users
from src.model import best_models_lower_threshold_23, best_models_upper_threshold_22
from src.model import new_best_models
from src.tuning.cross_validation.CrossSearchAbstractClass import compute_mean_std_result_dict, get_result_string
from src.utils.general_utility_functions import get_project_root_path

# CONSTANTS TO MODIFY
K_OUT = 1
CUTOFF = 10
ALLOW_COLD_USERS = False
LOWER_THRESHOLD = 23  # Remove users below or equal this threshold (default value: -1)
UPPER_THRESHOLD = 2**16-1 # Remove users above or equal this threshold (default value: 2**16-1)
IGNORE_NON_TARGET_USERS = True

AGE_TO_KEEP = []  # Default []

# VARIABLES TO MODIFY
model_name = "WeightedAvgLT23_FUSIONFIXED"


def _get_all_models(URM_train, ICM_all, UCM_all):
    all_models = {}

    all_models['WEIGHTED_AVG_ITEM'] = new_best_models.WeightedAverageItemBased.get_model(URM_train, ICM_all)

    all_models['S_PURE_SVD'] = new_best_models.PureSVDSideInfo.get_model(URM_train, ICM_all)
    all_models['S_IALS'] = new_best_models.IALSSideInfo.get_model(URM_train, ICM_all)
    all_models['USER_CBF_CF'] = new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all)
    all_models['USER_CF'] = new_best_models.UserCF.get_model(URM_train)

    return all_models


def get_model(URM_train, ICM_train, UCM_train):
    model = best_models_lower_threshold_23.WeightedAverageItemBasedWithRP3.\
        get_model(URM_train=URM_train, ICM_all=ICM_train)
    return model


def main():
    seed_list = get_seed_list()

    results_list = []
    for i in range(0, len(seed_list)):
        data_reader = read_split_load_data(K_OUT, ALLOW_COLD_USERS, seed_list[i])

        URM_train, URM_test = data_reader.get_holdout_split()
        ICM_all, _ = get_ICM_train_new(data_reader)
        UCM_all = get_UCM_train(data_reader)

        # Setting evaluator
        ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                        lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD,
                                        ignore_non_target_users=IGNORE_NON_TARGET_USERS)

        single_evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=ignore_users)

        # Get model
        model: BaseRecommender = get_model(URM_train, ICM_all, UCM_all)

        result_dict = single_evaluator.evaluateRecommender(model)[0][CUTOFF]
        print("FOLD-{} RESULT: {} \n".format(i + 1, result_dict))
        results_list.append(result_dict)

    mean_result_dict, std_result_dict = compute_mean_std_result_dict(results_list)
    results = get_result_string(mean_result_dict, std_result_dict)

    # Store results on file
    date_string = datetime.now().strftime('%b%d_%H-%M-%S')
    cross_valid_path = os.path.join(get_project_root_path(), "report/cross_validation/")
    file_path = os.path.join(cross_valid_path, "cross_valid_model_{}.txt".format(date_string))
    write_results_on_file(file_path, model_name, {}, len(seed_list), seed_list, results)


if __name__ == '__main__':
    main()
