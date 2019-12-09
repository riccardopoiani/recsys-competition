from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.DataPreprocessing import DataPreprocessingFeatureEngineering, DataPreprocessingImputation, \
    DataPreprocessingTransform, DataPreprocessingDiscretization
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender


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


def _get_all_models_ranked(URM_train, ICM_all, UCM_all, ICM_subclass_all):
    all_models = {}

    all_models['MIXED'] = best_models.WeightedAverageMixed.get_model(URM_train=URM_train,
                                                                     ICM_all=ICM_all,
                                                                     ICM_subclass_all=ICM_subclass_all,
                                                                     UCM_age_region=UCM_all)

    all_models['SLIM_BPR'] = best_models.SLIM_BPR.get_model(URM_train)
    all_models['P3ALPHA'] = best_models.P3Alpha.get_model(URM_train)
    all_models['RP3BETA'] = best_models.RP3Beta.get_model(URM_train)
    all_models['IALS'] = best_models.IALS.get_model(URM_train)
    all_models['USER_ITEM_ALL'] = best_models.UserItemKNNCBFCFDemographic.get_model(URM_train, ICM_all, UCM_all)

    return all_models


if __name__ == '__main__':
    # Weighted models best parameter - tuning results
    # weighted_best_param = {'ITEMCBFALLFOL': 0.5301487737487677, 'ITEMCBFCFFOL': 0.3878808597351461,
    #                       'ITEMCFFOL': 0.14971067748668312}
    # rank_best_param = {'strategy': 'norm_weighted_avg', 'multiplier_cutoff': 10, 'MIXED': 1.0, 'SLIM_BPR': 0.0,
    #                   'P3ALPHA': 1.0,
    #                   'RP3BETA': 0.0, 'IALS': 1.0, 'USER_ITEM_ALL': 1.0}

    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467]

    # ranked_score = 0
    # weighted_score = 0
    # mixed_score = 0
    temp_score = 0

    for i in range(0, len(seed_list)):
        # Data loading
        data_reader = RecSys2019Reader("../../data/")
        data_reader = DataPreprocessingFeatureEngineering(data_reader,
                                                          ICM_names_to_count=["ICM_sub_class"],
                                                          ICM_names_to_UCM=["ICM_sub_class", "ICM_price", "ICM_asset"],
                                                          UCM_names_to_ICM=[])
        data_reader = DataPreprocessingImputation(data_reader,
                                                  ICM_name_to_agg_mapper={"ICM_asset": np.median,
                                                                          "ICM_price": np.median})
        data_reader = DataPreprocessingTransform(data_reader,
                                                 ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                               "ICM_price": lambda x: np.log1p(1 / x),
                                                                               "ICM_item_pop": np.log1p,
                                                                               "ICM_sub_class_count": np.log1p},
                                                 UCM_name_to_transform_mapper={"UCM_price": lambda x: np.log1p(1 / x),
                                                                               "UCM_asset": lambda x: np.log1p(1 / x)})
        data_reader = DataPreprocessingDiscretization(data_reader,
                                                      ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                               "ICM_price": 200,
                                                                               "ICM_item_pop": 50,
                                                                               "ICM_sub_class_count": 50},
                                                      UCM_name_to_bins_mapper={"UCM_price": 200,
                                                                               "UCM_asset": 200,
                                                                               "UCM_user_act": 50})
        data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False,
                                                   force_new_split=True, seed=seed_list[i])
        data_reader.load_data()
        URM_train, URM_test = data_reader.get_holdout_split()

        # Build ICMs
        ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
        ICM_all = data_reader.get_ICM_from_name("ICM_all")

        # Build UCMs
        URM_all = data_reader.dataReader_object.get_URM_all()
        UCM_all = data_reader.get_UCM_from_name("UCM_all")
        UCM_all = get_warmer_UCM(UCM_all, URM_all, threshold_users=1)

        # Ranked
        # model = HybridRankBasedRecommender(URM_train)

        # all_models = _get_all_models_ranked(URM_train=URM_train, ICM_all=ICM_all,
        #                                    UCM_all=UCM_all, ICM_subclass_all=ICM_subclass_all)
        # for model_name, model_object in all_models.items():
        #    model.add_fitted_model(model_name, model_object)
        # print("The models added in the hybrid are: {}".format(list(all_models.keys())))
        # model.fit(**rank_best_param)

        # Weighted average recommender
        # model = HybridWeightedAverageRecommender(URM_train, normalize=False)
        # all_models = _get_all_models_weighted_average(URM_train=URM_train, ICM_subclass_all=ICM_subclass_all,
        #                                              UCM_all=UCM_all,
        #                                              ICM_all=ICM_all, ICM_numerical=ICM_numerical,
        #                                              ICM_categorical=ICM_subclass)
        # for model_name, model_object in all_models.items():
        #    model.add_fitted_model(model_name, model_object)

        # model.fit(**weighted_best_param)

        # Mixed similarity building
        # hybrid = best_models.MixedUser.get_model(URM_train=URM_train, UCM_all=UCM_all)

        # Setting evaluator
        cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

        cutoff_list = [10]
        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

        par = {'user_similarity_type': 'cosine', 'item_similarity_type': 'asymmetric', 'user_feature_weighting': 'BM25',
               'item_feature_weighting': 'TF-IDF', 'user_normalize': True,
               'item_normalize': True, 'item_asymmetric_alpha': 0.1539884061705812,
               'user_topK': 16, 'user_shrink': 1000, 'item_topK': 12, 'item_shrink': 1374}
        temp_model = UserItemCBFCFDemographicRecommender(URM_train, UCM_all, ICM_all)
        temp_model.fit(**par)

        # current_ranked_score = evaluator.evaluateRecommender(model)[0][10]['MAP']
        # weighted_current_map = evaluator.evaluateRecommender(model)[0][10]['MAP']
        # mixed_current_map = evaluator.evaluateRecommender(hybrid)[0][10]['MAP']
        curr_map = evaluator.evaluateRecommender(temp_model)[0][10]['MAP']

        print("SEED: {} \n ".format(seed_list[i]))
        print("TempModelCurrMap {} \n".format(curr_map))
        # print("WeightedCurrMap {} \n".format(current_ranked_score))
        # print("MixedCurrMap {} \n ".format(mixed_current_map))

        # ranked_score += current_ranked_score
        temp_score += curr_map
        # weighted_score += weighted_current_map
        # mixed_score += mixed_current_map

    # ranked_score /= len(seed_list)
    # weighted_score /= len(seed_list)
    # mixed_score /= len(seed_list)
    temp_score /= len(seed_list)

    # Store results on file
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_comparison" + now + ".txt"
    num_folds = len(seed_list)

    f = open(destination_path, "w")
    f.write("X-validation HybridSubmission2 of warm users \n\n")
    # f.write("MixedSimilarity tuning map {} --- MixedSimilarity X-val map {} \n\n".format(mixed_best_param_map,
    #                                                                                     mixed_score))
    f.write("X-val map {} \n\n".format(temp_score))

    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n\n")

    # f.write("MixedBestParam:")
    # f.write(str(mixed_best_param))
    # f.write("\n")

    # f.write("WeightedBestParam:")
    # f.write(str(weighted_best_param))
    # f.write("\n")

    f.close()
