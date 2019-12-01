from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, merge_UCM
from src.data_management.data_getter import get_warmer_UCM
from src.feature.demographics_content import get_user_demographic
from src.feature.demographics_content import get_profile_demographic_wrapper
from src.model import best_models
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.plots.recommender_plots import *
from src.data_management.dataframe_preprocesser import get_preprocessed_dataframe


def _get_all_models(URM_train, ICM_numerical, ICM_categorical):
    all_models = {}

    item_cf_keywargs = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train)
    item_cf.fit(**item_cf_keywargs)
    all_models['ITEM_CF'] = item_cf

    user_cf_keywargs = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    user_cf = UserKNNCFRecommender(URM_train)
    user_cf.fit(**user_cf_keywargs)
    all_models['USER_CF'] = user_cf

    item_cbf_numerical_kwargs = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                                 'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                                 'topK': 1000}
    item_cbf_numerical = ItemKNNCBFRecommender(URM_train, ICM_numerical)
    item_cbf_numerical.fit(**item_cbf_numerical_kwargs)
    item_cbf_numerical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderNumerical"
    all_models['ITEM_CBF_NUM'] = item_cbf_numerical

    item_cbf_categorical_kwargs = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                   'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    item_cbf_categorical = ItemKNNCBFRecommender(URM_train, ICM_categorical)
    item_cbf_categorical.fit(**item_cbf_categorical_kwargs)
    item_cbf_categorical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderCategorical"
    all_models['ITEM_CBF_CAT'] = item_cbf_categorical

    slim_bpr_kwargs = {'topK': 5, 'epochs': 1499, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
    slim_bpr = SLIM_BPR_Cython(URM_train)
    slim_bpr.fit(**slim_bpr_kwargs)
    all_models['SLIM_BPR'] = slim_bpr

    p3alpha_kwargs = {'topK': 84, 'alpha': 0.6033770403001427, 'normalize_similarity': True}
    p3alpha = P3alphaRecommender(URM_train)
    p3alpha.fit(**p3alpha_kwargs)
    all_models['P3ALPHA'] = p3alpha

    rp3beta_kwargs = {'topK': 5, 'alpha': 0.37829128706576887, 'beta': 0.0, 'normalize_similarity': False}
    rp3beta = RP3betaRecommender(URM_train)
    rp3beta.fit(**rp3beta_kwargs)
    all_models['RP3BETA'] = rp3beta

    return all_models


if __name__ == '__main__':
    # Data reading
    data_reader = RecSys2019Reader()
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']
    df = get_preprocessed_dataframe("../../data/", keep_warm_only=True)

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM_categorical, URM_train.T, {}, {})

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    # Build recommender list
    item_cbf_numerical_args = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                               'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                               'topK': 1000}
    item_cbf_numerical = ItemKNNCBFRecommender(URM_train, ICM_numerical)
    item_cbf_numerical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderNumerical"
    item_cbf_numerical.fit(**item_cbf_numerical_args)

    best_parameters_categorical = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                   'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    item_cbf_categorical = ItemKNNCBFRecommender(URM_train, ICM_categorical)
    item_cbf_categorical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderCategorical"
    item_cbf_categorical.fit(**best_parameters_categorical)

    item_cf_args = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                    'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train)
    item_cf.fit(**item_cf_args)

    user_cf_best_parameters = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                               'feature_weighting': 'TF-IDF'}
    user_cf = UserKNNCFRecommender(URM_train)
    user_cf.fit(**user_cf_best_parameters)

    best_parameters_alpha = {'topK': 84, 'alpha': 0.6033770403001427, 'normalize_similarity': True}
    p3alpha = P3alphaRecommender(URM_train)
    p3alpha.fit(**best_parameters_alpha)

    best_parameters_p3beta = {'topK': 5, 'alpha': 0.37829128706576887, 'beta': 0.0, 'normalize_similarity': False}
    p3beta = RP3betaRecommender(URM_train)
    p3beta.fit(**best_parameters_p3beta)

    best_parameters = {'topK': 5, 'epochs': 1499, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
    slim = SLIM_BPR_Cython(URM_train)
    slim.fit(**best_parameters)

    best_parameters = {'num_factors': 376}
    pure_svd = PureSVDRecommender(URM_train)
    pure_svd.fit(**best_parameters)

    # Building the hybrid
    hybrid = HybridWeightedAverageRecommender(URM_train)
    all_models = _get_all_models(URM_train=URM_train, ICM_numerical=ICM_numerical,
                                 ICM_categorical=ICM_categorical)
    for model_name, model_object in all_models.items():
        hybrid.add_fitted_model(model_name, model_object)

    hybrid_param = {'ITEM_CF': 0.969586046573504, 'USER_CF': 0.943330450168123,
                    'ITEM_CBF_NUM': 0.03250599212747674,
                    'ITEM_CBF_CAT': 0.018678076600871066,
                    'SLIM_BPR': 0.03591603993769955,
                    'P3ALPHA': 0.7474845972085382, 'RP3BETA': 0.1234024366177027}
    hybrid.fit(**hybrid_param)

    user_cf_cbf_demographic = best_models.UserItemKNNCBFCFDemographic.get_model(URM_train, ICM_train=ICM_all, UCM_train=UCM_all)

    recommender_list = []
    #recommender_list.append(item_cf)
    #recommender_list.append(user_cf)
    #recommender_list.append(slim)
    #recommender_list.append(pure_svd)
    #recommender_list.append(p3alpha)
    #recommender_list.append(p3beta)
    #recommender_list.append(item_cbf_numerical)
    #recommender_list.append(item_cbf_categorical)
    recommender_list.append(hybrid)
    recommender_list.append(user_cf_cbf_demographic)

    # Building path
    version_path = "../../report/graphics/comparison/"

    # Plot the comparison on item popularity
    """item_popularity, item_popularity_descriptor = get_profile_demographic_wrapper(URM_train, bins=10, users=False)
    content_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
                 URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
                 output_folder_path=version_path + "item_popularity/", content_name="Item popularity",
                 content=item_popularity, content_describer_list=item_popularity_descriptor,
                 exclude_cold_items=False)"""

    # Plotting the comparison on age
    #region_demographic = get_user_demographic(UCM_region, URM_all, 3, binned=True)
    #region_demographic_describer_list = [-1, 0, 2, 3, 4, 5, 6, 7]
    #demographic_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
    #                URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
    #                 output_folder_path=version_path + "region/", demographic_name="Region",
    #                 demographic=region_demographic, demographic_describer_list=region_demographic_describer_list,
    #                 exclude_cold_users=True)

    # Plotting the comparison on region
    #age_demographic = get_user_demographic(UCM_age, URM_all, 3, binned=True)
    #age_demographic_describer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #demographic_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
    #                 URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
    #                 output_folder_path=version_path + "age/", demographic_name="Age",
    #                 demographic=age_demographic, demographic_describer_list=age_demographic_describer_list,
    #                 exclude_cold_users=True)

    plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True,
                                               output_folder_path=version_path + "user_activity/")

    # Plotting the comparison based on clustering
    #dataframe = get_preprocessed_dataframe(path="../../data/", keep_warm_only=True)
    #plot_clustering_demographics(recommender_list, URM_train, URM_test, dataframe,
    #                             metric="MAP", cutoff=10, save_on_file=True,
    #                             output_folder_path=version_path + "clustering/",
    #                             exclude_cold_users=True)
