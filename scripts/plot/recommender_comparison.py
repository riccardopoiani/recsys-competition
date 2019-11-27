from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.model.FallbackRecommender.AdvancedTopPopular import AdvancedTopPopular
from src.plots.recommender_plots import *
from src.data_management.dataframe_preprocesser import get_preprocessed_dataframe

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

    # Building the recommenders
    """
    item_cf_keywargs = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train)
    item_cf.fit(**item_cf_keywargs)

    user_cf_keywargs = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    user_cf = UserKNNCFRecommender(URM_train)
    user_cf.fit(**user_cf_keywargs)

    item_cbf_numerical_kwargs = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                       'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                       'topK': 1000}
    item_cbf_numerical = ItemKNNCBFRecommender(ICM_numerical, URM_train)
    item_cbf_numerical.fit(**item_cbf_numerical_kwargs)
    item_cbf_numerical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderNumerical"

    item_cbf_categorical_kwargs = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                   'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    item_cbf_categorical = ItemKNNCBFRecommender(ICM_categorical, URM_train)
    item_cbf_categorical.fit(**item_cbf_categorical_kwargs)
    item_cbf_categorical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderCategorical"
    

    p3alpha_kwargs = {'topK': 84, 'alpha': 0.6033770403001427, 'normalize_similarity': True}
    p3alpha = P3alphaRecommender(URM_train)
    p3alpha.fit(**p3alpha_kwargs)

    rp3beta_kwargs = {'topK': 5, 'alpha': 0.37829128706576887, 'beta': 0.0, 'normalize_similarity': False}
    rp3beta = RP3betaRecommender(URM_train)
    rp3beta.fit(**rp3beta_kwargs)

    slim_bpr_kwargs = {'topK': 5, 'epochs': 1499, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
    slim_bpr = SLIM_BPR_Cython(URM_train)
    slim_bpr.fit(**slim_bpr_kwargs)

    pure_svd_kwargs = {'num_factors': 350}
    pure_svd = PureSVDRecommender(URM_train)
    pure_svd.fit(**pure_svd_kwargs)
    """

    top_pop = TopPop(URM_train)
    top_pop.fit()

    advanced_top_pop_keywargs = advanced_keywargs =  {'clustering_method': 'kmodes', 'n_clusters': 7, 'init_method': 'Cao'}
    advanced_top_pop = AdvancedTopPopular(URM_train, df, mapper)
    advanced_top_pop.fit(**advanced_top_pop_keywargs)



    recommender_list = []
    #recommender_list.append(item_cf)
    #recommender_list.append(user_cf)
    recommender_list.append(top_pop)
    recommender_list.append(advanced_top_pop)
    #recommender_list.append(item_cbf_numerical)
    #recommender_list.append(item_cbf_categorical)
    #recommender_list.append(p3alpha)
    #recommender_list.append(rp3beta)
    #recommender_list.append(pure_svd_kwargs)
    #recommender_list.append(slim_bpr)


    # Plotting the comparison based on user activity
    #plot_compare_recommender_user_group()
    plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True)