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
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_users=3)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Building the recommenders
    item_cf_keywargs = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train)
    item_cf.fit(**item_cf_keywargs)

    user_cf_keywargs = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    user_cf = UserKNNCFRecommender(URM_train)
    user_cf.fit(**user_cf_keywargs)

    top_pop = TopPop(URM_train)
    top_pop.fit()

    recommender_list = []
    recommender_list.append(item_cf)
    recommender_list.append(user_cf)
    recommender_list.append(top_pop)

    # Plotting the comparison based on user activity
    plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True)