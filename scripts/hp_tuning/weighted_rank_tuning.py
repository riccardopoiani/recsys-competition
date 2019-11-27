from datetime import datetime

from numpy.random import seed

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, merge_UCM
from src.data_management.data_getter import get_warmer_UCM
from src.data_management.dataframe_preprocesser import get_preprocessed_dataframe
from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender
from src.tuning.run_parameter_search_hybrid import run_parameter_search_hybrid
from src.model import best_models

SEED = 69420


def _get_all_models(URM_train, UCM_train, demographic_df, mapper):
    all_models = {}

    topPop = TopPop(URM_train)
    topPop.fit()
    all_models["TOP_POP"] = topPop

    all_models["ADV_TOP_POP"] = best_models.AdvancedTopPop.get_model(URM_train, demographic_df, mapper)
    all_models["USER_CBF"] = best_models.UserCBF.get_model(URM_train, UCM_train)

    return all_models


if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed(SEED)

    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']

    df = get_preprocessed_dataframe("../../data/", keep_warm_only=True)

    # Reset seed for hyper-parameter tuning
    seed()

    model = HybridRankBasedRecommender(URM_train)

    all_models = _get_all_models(URM_train, UCM_all, df, mapper)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    # Setting evaluator
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=warm_users)

    version_path = "../../report/hp_tuning/hybrid_weighted_rank/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_hybrid(model, metric_to_optimize="MAP",
                                evaluator_validation=evaluator,
                                output_folder_path=version_path,
                                n_cases=35, parallelizeKNN=True)

    print("...tuning ended")
