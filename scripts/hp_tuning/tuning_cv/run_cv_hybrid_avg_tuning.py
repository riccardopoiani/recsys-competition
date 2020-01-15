from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from scripts.scripts_utils import set_env_variables, read_split_load_data
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.data_reader import get_ICM_train_new, get_UCM_train_new, get_ignore_users
from src.model import new_best_models, best_models_lower_threshold_23, best_models_upper_threshold_22
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.tuning.cross_validation.run_cv_parameter_search_hybrid_avg import run_cv_parameter_search_hybrid_avg
from src.utils.general_utility_functions import get_split_seed, get_seed_lists

# Parameters to modify
N_CASES = 120
N_RANDOM_STARTS = 70
N_FOLDS = 5
K_OUT = 1
CUTOFF = 10
UPPER_THRESHOLD = 2 ** 16 - 1  # default 2**16-1
LOWER_THRESHOLD = 23  # default -1
ALLOW_COLD_USERS = False
IGNORE_NON_TARGET_USERS = True
NORMALIZE = True

AGE_TO_KEEP = []  # Default []


def get_map_max():
    return 24


def get_mapping():
    mapping = {0: 0, 1: 0.1, 2: 0.13, 3: 0.17, 4: 0.2, 5: 0.25, 6: 0.29, 7: 0.31, 8: 0.34, 9: 0.39,
               10: 0.43, 11: 0.46, 12: 0.5, 13: 0.551, 14: 0.605, 15: 0.62, 16: 0.66, 17: 0.71, 18: 0.75, 19: 0.8,
               20: 0.85,
               21: 0.88, 22: 0.91, 23: 0.965, 24: 1}
    return mapping


def _get_all_models(URM_train, ICM_train, UCM_train):
    # Method to modify
    all_models = {}
    all_models['ItemAvgLT23'] = best_models_lower_threshold_23.WeightedAverageItemBasedWithRP3.get_model(
        URM_train=URM_train,
        ICM_all=ICM_train)

    all_models['ItemAvgOld'] = best_models_upper_threshold_22.WeightedAverageItemBased.get_model(URM_train=URM_train,
                                                                                                 ICM_all=ICM_train)

    return all_models


def get_model(URM_train, ICM_train, UCM_train):
    model = HybridWeightedAverageRecommender(URM_train, normalize=NORMALIZE, mapping=None,
                                             global_normalization=False)

    all_models = _get_all_models(URM_train=URM_train, ICM_train=ICM_train, UCM_train=UCM_train)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    return model


if __name__ == '__main__':
    set_env_variables()
    seeds = get_seed_lists(N_FOLDS, get_split_seed())

    # --------- DATA LOADING SECTION --------- #
    URM_train_list = []
    ICM_train_list = []
    UCM_train_list = []
    evaluator_list = []
    model_list = []
    for fold_idx in range(N_FOLDS):
        # Read and split data
        data_reader = read_split_load_data(K_OUT, ALLOW_COLD_USERS, seeds[fold_idx])
        URM_train, URM_test = data_reader.get_holdout_split()
        ICM_train, item_feature2range = get_ICM_train_new(data_reader)
        UCM_train, user_feature2range = get_UCM_train_new(data_reader)

        # Ignore users and setting evaluator
        ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                        LOWER_THRESHOLD, UPPER_THRESHOLD,
                                        ignore_non_target_users=IGNORE_NON_TARGET_USERS)

        # Ignore users by age
        # UCM_age = data_reader.get_UCM_from_name("UCM_age")
        # age_feature_to_id_mapper = data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
        # age_demographic = get_user_demographic(UCM_age, age_feature_to_id_mapper, binned=True)
        # ignore_users = np.unique(np.concatenate((ignore_users, get_ignore_users_age(age_demographic, AGE_TO_KEEP))))

        URM_train_list.append(URM_train)
        ICM_train_list.append(ICM_train)
        UCM_train_list.append(UCM_train)
        model_list.append(get_model(URM_train, ICM_train, UCM_train))

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=np.unique(ignore_users))
        evaluator_list.append(evaluator)

    # --------- HYPER PARAMETERS TUNING SECTION --------- #
    print("Start tuning...")

    hp_tuning_path = "../../../report/hp_tuning/" + "hybrid_avg" + "/"
    date_string = datetime.now().strftime('%b%d_%H-%M-%S_k1_lt_{}/'.format(LOWER_THRESHOLD))
    output_folder_path = hp_tuning_path + date_string

    run_cv_parameter_search_hybrid_avg(model_list, URM_train_list, metric_to_optimize="MAP",
                                       evaluator_validation_list=evaluator_list, output_folder_path=output_folder_path,
                                       parallelize_search=True, n_cases=N_CASES, n_random_starts=N_RANDOM_STARTS,
                                       map_max=get_map_max())
    print("...tuning ended")
