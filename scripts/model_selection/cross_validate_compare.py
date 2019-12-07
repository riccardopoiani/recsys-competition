from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender


def _get_all_models_weighted_average(URM_train, ICM_all, UCM_all, ICM_subclass_all, ICM_numerical, ICM_categorical):
    all_models = {}

    all_models['ITEMCBFALLFOL'] = best_models.ItemCBF_all_EUC1_FOL3.get_model(URM_train=URM_train,
                                                                              ICM_train=ICM_all,
                                                                              load_model=False)
    all_models['ITEMCBFCFFOL'] = best_models.ItemCBF_CF_FOL_3_ECU_1.get_model(URM_train=URM_train,
                                                                              ICM_train=ICM_subclass_all,
                                                                              load_model=False)
    all_models['ITEMCFFOL'] = best_models.ItemCF_EUC_1_FOL_3.get_model(URM_train=URM_train)
    return all_models


if __name__ == '__main__':
    # Weighted models best parameter - tuning results
    weighted_best_param = {'ITEMCBFALLFOL': 0.5301487737487677, 'ITEMCBFCFFOL': 0.3878808597351461,
                           'ITEMCFFOL': 0.14971067748668312}
    weighted_best_param_tuning_map = 0.0303

    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467]

    weighted_score = 0
    # mixed_score = 0
    # temp_score = 0

    for i in range(0, len(seed_list)):
        # Data loading
        data_reader = RecSys2019Reader("../../data/")
        data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                                   force_new_split=True, seed=seed_list[i])
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

        # Weighted average recommender
        model = HybridWeightedAverageRecommender(URM_train, normalize=False)
        all_models = _get_all_models_weighted_average(URM_train=URM_train, ICM_subclass_all=ICM_subclass_all,
                                                      UCM_all=UCM_all,
                                                      ICM_all=ICM_all, ICM_numerical=ICM_numerical,
                                                      ICM_categorical=ICM_subclass)
        for model_name, model_object in all_models.items():
            model.add_fitted_model(model_name, model_object)

        model.fit(**weighted_best_param)

        # Mixed similarity building
        # hybrid = best_models.MixedUser.get_model(URM_train=URM_train, UCM_all=UCM_all)

        # Setting evaluator
        cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
        warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 3
        warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
        ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

        cutoff_list = [10]
        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

        # temp_model = best_models.UserCBF_CF_Warm.get_model(URM_train=URM_train,
        #                                                   UCM_train=UCM_all)

        # temp_model = ItemKNNCFRecommender(URM_train=URM_train)
        # temp_model_args = {'topK': 30, 'shrink': 2, 'similarity': 'tversky', 'normalize': True,
        #                    'tversky_alpha': 0.07389665789291368,'tversky_beta': 0.2013116625076397}
        # temp_model.fit(**temp_model_args)

        weighted_current_map = evaluator.evaluateRecommender(model)[0][10]['MAP']
        # mixed_current_map = evaluator.evaluateRecommender(hybrid)[0][10]['MAP']
        # curr_map = evaluator.evaluateRecommender(temp_model)[0][10]['MAP']

        print("SEED: {} \n ".format(seed_list[i]))
        # print("TempModelCurrMap {} \n".format(curr_map))
        print("WeightedCurrMap {} \n".format(weighted_current_map))
        # print("MixedCurrMap {} \n ".format(mixed_current_map))

        # temp_score += curr_map
        weighted_score += weighted_current_map
        # mixed_score += mixed_current_map

    weighted_score /= len(seed_list)
    # mixed_score /= len(seed_list)
    # temp_score /= len(seed_list)

    # Store results on file
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_comparison" + now + ".txt"
    num_folds = len(seed_list)

    f = open(destination_path, "w")
    f.write("X-validation WeightedAvgFOL3 of users with [1,3] interactions \n\n")
    # f.write("MixedSimilarity tuning map {} --- MixedSimilarity X-val map {} \n\n".format(mixed_best_param_map,
    #                                                                                     mixed_score))
    f.write("X-val map {} \n\n".format(weighted_score))

    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n\n")

    # f.write("MixedBestParam:")
    # f.write(str(mixed_best_param))
    # f.write("\n")

    # f.write("WeightedBestParam:")
    # f.write(str(weighted_best_param))
    # f.write("\n")

    f.close()
