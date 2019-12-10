from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import best_models
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.tuning.run_parameter_search_hybrid import run_parameter_search_hybrid
from src.utils.general_utility_functions import get_split_seed


def _get_all_models(URM_train, ICM_all, UCM_all):
    all_models = {}

    all_models['ITEMCBFALLFOL'] = best_models.ItemCBF_CF_all_EUC1_FOL3.get_model(URM_train=URM_train,
                                                                                 ICM_train=ICM_all,
                                                                                 load_model=False)
    all_models['ITEMCBFCFFOL'] = best_models.ItemCBF_CF_FOL_3_ECU_1.get_model(URM_train=URM_train,
                                                                              ICM_train=ICM_all,
                                                                              load_model=False)
    all_models['USERCBFCF'] = best_models.UserCBF_CF_Warm.get_model(URM_train=URM_train, UCM_train=UCM_all)

    return all_models


if __name__ == '__main__':
    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all = get_ICM_train(data_reader)

    # Build UCMs
    UCM_all = get_UCM_train(data_reader, root_data_path)

    model = HybridWeightedAverageRecommender(URM_train, normalize=False)

    all_models = _get_all_models(URM_train=URM_train, UCM_all=UCM_all, ICM_all=ICM_all)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    very_warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 3
    very_warm_users = np.arange(URM_train.shape[0])[very_warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, very_warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

    version_path = "../../report/hp_tuning/hybrid_weighted_avg"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_hybrid(model, metric_to_optimize="MAP",
                                evaluator_validation=evaluator,
                                output_folder_path=version_path,
                                n_cases=60, n_random_starts=20)

    print("...tuning ended")
