from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_UCM_train, get_ICM_train, get_ICM_train_new
from src.model import new_best_models
from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender
from src.tuning.holdout_validation.run_parameter_search_hybrid import run_parameter_search_hybrid
from src.utils.general_utility_functions import get_split_seed


def _get_all_models(URM_train, ICM_all, UCM_all):
    all_models = {}

    all_models['WEIGHTED_AVG_ITEM'] = new_best_models.WeightedAverageItemBased.get_model(URM_train, ICM_all)

    #all_models['S_SLIM_BPR'] = new_best_models.SSLIM_BPR.get_model(sps.vstack([URM_train, ICM_all.T]))
    all_models['S_PURE_SVD'] = new_best_models.PureSVDSideInfo.get_model(URM_train, ICM_all)
    all_models['S_IALS'] = new_best_models.IALSSideInfo.get_model(URM_train, ICM_all)
    all_models['USER_CBF_CF'] = new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all)
    all_models['USER_CF'] = new_best_models.UserCF.get_model(URM_train)

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
    ICM_all, _ = get_ICM_train_new(data_reader)

    # Build UCMs
    UCM_all = get_UCM_train(data_reader)

    model = HybridRankBasedRecommender(URM_train)

    all_models = _get_all_models(URM_train=URM_train, ICM_all=ICM_all, UCM_all=UCM_all)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    version_path = "../../report/hp_tuning/hybrid_weighted_rank/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_hybrid(model,
                                metric_to_optimize="MAP",
                                evaluator_validation=evaluator,
                                output_folder_path=version_path,
                                n_cases=200,
                                n_random_starts=70,
                                parallelizeKNN=True)

    print("...tuning ended")
