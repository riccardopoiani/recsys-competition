from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.plots.recommender_plots import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.DataPreprocessing import *
from course_lib.KNN.ItemKNNCFRecommender import *
from course_lib.KNN.UserKNNCFRecommender import *
from course_lib.Base.NonPersonalizedRecommender import TopPop

if __name__ == '__main__':
    # Data reading
    data_reader = RecSys2019Reader()
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICM numerical
    ICM_price = data_reader.get_ICM_from_name("ICM_price")
    ICM_asset = data_reader.get_ICM_from_name("ICM_asset")
    ICM_item_pop = data_reader.get_ICM_from_name("ICM_item_pop")
    ICM_numerical, _ = merge_ICM(ICM_price, ICM_asset, {}, {})
    ICM_numerical, _ = merge_ICM(ICM_numerical, ICM_item_pop, {}, {})

    # Build ICM categorical
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")

    # Building the recommenders
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

    item_cbf_categorical_kwargs = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                   'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    item_cbf_categorical = ItemKNNCBFRecommender(ICM_categorical, URM_train)
    item_cbf_categorical.fit(**item_cbf_numerical_kwargs)
    item_cbf_categorical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderCategorical"

    top_pop = TopPop(URM_train)
    top_pop.fit()

    recommender_list = []
    recommender_list.append(item_cf)
    recommender_list.append(user_cf)
    recommender_list.append(top_pop)
    recommender_list.append(item_cbf_numerical)
    recommender_list.append(item_cbf_categorical)

    # Plotting the comparison based on user activity
    plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True)