from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader, get_ICM_numerical
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.tuning.run_parameter_search_hybrid import run_parameter_search_hybrid
from src.utils.general_utility_functions import get_split_seed


def _get_all_models(URM_train, ICM_all, UCM_all, ICM_subclass_all, ICM_numerical, ICM_categorical):
    all_models = {}

    all_models['ITEM_CBF_CF'] = best_models.ItemCBF_CF.get_model(URM_train, ICM_subclass_all, load_model=False)
    all_models['ITEM_CF'] = best_models.ItemCF.get_model(URM_train=URM_train, load_model=False, save_model=False)
    all_models['ITEM_CBF_ALL'] = best_models.ItemCBF_all.get_model(URM_train=URM_train,
                                                                   load_model=False, save_model=False,
                                                                   ICM_train=ICM_all)
    # all_models['USER_CF'] = best_models.UserCF.get_model(URM_train, load_model=False)
    # all_models['SLIM_BPR'] = best_models.SLIM_BPR.get_model(URM_train)
    # all_models['P3ALPHA'] = best_models.P3Alpha.get_model(URM_train, load_model=False)
    # all_models['RP3BETA'] = best_models.RP3Beta.get_model(URM_train, load_model=False)
    # all_models['IALS'] = best_models.IALS.get_model(URM_train, load_model=False)
    # all_models['USER_ITEM_ALL'] = best_models.UserItemKNNCBFCFDemographic.get_model(URM_train, ICM_subclass_all, UCM_all)
    # all_models['Mixed'] = best_models.MixedItem.get_model(URM_train=URM_train, ICM_all=ICM_all,
    #                                                      ICM_subclass_all=ICM_subclass_all,
    #                                                      load_model=False)
    # all_models['Hybrid1'] = best_models.HybridWeightedAvgSubmission1.get_model(URM_train=URM_train,
    #                                                                           ICM_numerical=ICM_numerical,
    #                                                                           ICM_categorical=ICM_categorical,
    #                                                                           load_model=False)
    # all_models['Hybrid12'] = best_models.HybridWeightedAvgSubmission2.get_model(URM_train=URM_train,
    #                                                                           ICM_train=ICM_subclass_all,
    #                                                                            UCM_train=UCM_all,
    #                                                                           load_model=False)

    return all_models


if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_all")
    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})
    ICM_subclass_all, _ = merge_ICM(ICM_subclass, URM_train.transpose(), {}, {})

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    model = HybridWeightedAverageRecommender(URM_train)

    all_models = _get_all_models(URM_train=URM_train, ICM_subclass_all=ICM_subclass_all, UCM_all=UCM_all,
                                 ICM_all=ICM_all, ICM_numerical=ICM_numerical, ICM_categorical=ICM_subclass)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    # Setting evaluator
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    version_path = "../../report/hp_tuning/hybrid_weighted_avg/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_hybrid(model, metric_to_optimize="MAP",
                                evaluator_validation=evaluator,
                                output_folder_path=version_path,
                                n_cases=60, n_random_starts=10)

    print("...tuning ended")
