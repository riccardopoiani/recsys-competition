from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from scripts.scripts_utils import read_split_load_data
from src.data_management.data_reader import get_UCM_train, get_ignore_users, get_ICM_train_new
from src.model import new_best_models
from src.model.HybridRecommender.HybridRerankingRecommender import HybridRerankingRecommender
from src.tuning.holdout_validation.run_parameter_search_hybrid import run_parameter_search_hybrid
from src.utils.general_utility_functions import get_split_seed

K_OUT = 1
CUTOFF = 10
ALLOW_COLD_USERS = False
LOWER_THRESHOLD = -1  # Remove users below or equal this threshold (default value: -1)
UPPER_THRESHOLD = 2**16-1  # Remove users above or equal this threshold (default value: 2**16-1)
IGNORE_NON_TARGET_USERS = True

def _get_all_models(URM_train, ICM_all, UCM_all):
    all_models = {}

    all_models['USER_CBF_CF'] = new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all)
    all_models['ITEM_CBF_CF'] = new_best_models.ItemCBF_CF.get_model(URM_train, ICM_all)
    all_models['RP3BETA_SIDE'] = new_best_models.RP3BetaSideInfo.get_model(URM_train, ICM_all)
    all_models['ITEM_CBF_FW'] = new_best_models.ItemCBF_all_FW.get_model(URM_train, ICM_all)
    all_models['PURE_SVD_SIDE'] = new_best_models.PureSVDSideInfo.get_model(URM_train, ICM_all)
    all_models['IALS_SIDE'] = new_best_models.IALSSideInfo.get_model(URM_train, ICM_all)
    #all_models['SSLIM_BPR'] = new_best_models.SSLIM_BPR.get_model(URM_train, ICM_all)

    return all_models


if __name__ == '__main__':
    # Data loading
    data_reader = read_split_load_data(K_OUT, ALLOW_COLD_USERS, get_split_seed())
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all, _ = get_ICM_train_new(data_reader)

    # Build UCMs
    UCM_all = get_UCM_train(data_reader)

    main_recommender = new_best_models.FusionMergeItem_CBF_CF.get_model(URM_train, ICM_all)
    model = HybridRerankingRecommender(URM_train, main_recommender)

    all_models = _get_all_models(URM_train=URM_train, UCM_all=UCM_all, ICM_all=ICM_all)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    # Setting evaluator
    ignore_users = get_ignore_users(URM_train, data_reader.get_original_user_id_to_index_mapper(),
                                    lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD,
                                    ignore_non_target_users=True)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[CUTOFF], ignore_users=ignore_users)

    version_path = "../../report/hp_tuning/hybrid_reranking"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_hybrid(model, metric_to_optimize="MAP",
                                evaluator_validation=evaluator,
                                output_folder_path=version_path,
                                n_cases=200, n_random_starts=70)

    print("...tuning ended")
