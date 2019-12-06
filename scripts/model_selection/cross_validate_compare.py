from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader, get_ICM_numerical
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.HybridRecommender.HybridMixedSimilarityRecommender import ItemHybridModelRecommender
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from datetime import datetime


def _get_all_models_weighted_average(URM_train, ICM_all, UCM_all, ICM_subclass_all, ICM_numerical, ICM_categorical):
    all_models = {}
    all_models['ITEM_CBF_CF'] = best_models.ItemCBF_CF.get_model(URM_train, ICM_subclass_all, load_model=False)
    all_models['ITEM_CF'] = best_models.ItemCF.get_model(URM_train=URM_train, load_model=False, save_model=False)
    all_models['ITEM_CBF_ALL'] = best_models.ItemCBF_all.get_model(URM_train=URM_train,
                                                                   load_model=False, save_model=False,
                                                                   ICM_train=ICM_all)
    return all_models


if __name__ == '__main__':
    # Weighted models best parameter - tuning results
    weighted_best_param = {'ITEM_CBF_CF': 0.09095731575438316, 'ITEM_CF': 0.012943479714417009,
                           'ITEM_CBF_ALL': 0.0024173339166811973}
    weighted_best_param_tuning_map = 0.0350374

    # Mixed similarity best parameters - tuning results
    mixed_best_param = {'topK': 1056, 'alpha1': 0.15663912433996233, 'alpha2': 0.8823526667242283,
                        'alpha3': 0.28586869725984193}
    mixed_best_param_map = 0.0350856

    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467]

    weighted_score = 0
    mixed_score = 0

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
        model = HybridWeightedAverageRecommender(URM_train)
        all_models = _get_all_models_weighted_average(URM_train=URM_train, ICM_subclass_all=ICM_subclass_all,
                                                      UCM_all=UCM_all,
                                                      ICM_all=ICM_all, ICM_numerical=ICM_numerical,
                                                      ICM_categorical=ICM_subclass)
        for model_name, model_object in all_models.items():
            model.add_fitted_model(model_name, model_object)

        model.fit(**weighted_best_param)

        # Mixed similarity building
        item_cf = best_models.ItemCF.get_model(URM_train, load_model=False)
        item_cbf_cf = best_models.ItemCBF_CF.get_model(URM_train=URM_train, load_model=False,
                                                       ICM_train=ICM_subclass_all)
        item_cbf_all = best_models.ItemCBF_all.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)

        hybrid = ItemHybridModelRecommender(URM_train)
        hybrid.add_similarity_matrix(item_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_all.W_sparse)

        hybrid.fit(**mixed_best_param)

        # Setting evaluator
        cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
        cutoff_list = [10]
        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

        weighted_current_map = evaluator.evaluateRecommender(model)[0][10]['MAP']
        mixed_current_map = evaluator.evaluateRecommender(hybrid)[0][10]['MAP']

        print("SEED: {} \n ".format(seed_list[i]))
        print("WeightedCurrMap {} \n".format(weighted_current_map))
        print("MixedCurrMap {} \n ".format(mixed_current_map))

        weighted_score += weighted_current_map
        mixed_score += mixed_current_map

    weighted_score /= len(seed_list)
    mixed_score /= len(seed_list)

    # Store results on file
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_comparison" + now + ".txt"
    num_folds = len(seed_list)

    f = open(destination_path, "w")
    f.write("Comparison between MixedItemSimilarity and WeightedAverage on the same models, using best models"
            "find with hp tuning\n\n")
    f.write("MixedSimilarity tuning map {} --- MixedSimilarity X-val map {} \n\n".format(mixed_best_param_map,
                                                                                         mixed_score))
    f.write("Weighted tuning map {} --- Weighted X-val map {} \n\n".format(weighted_best_param_tuning_map,
                                                                           weighted_score))

    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n\n")

    f.write("MixedBestParam:")
    f.write(str(mixed_best_param))
    f.write("\n")

    f.write("WeightedBestParam:")
    f.write(str(weighted_best_param))
    f.write("\n")

    f.close()
