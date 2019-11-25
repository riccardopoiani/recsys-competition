from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.feature.demographics import get_user_profile_demographic
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
from src.plots.recommender_plots import basic_plots_recommender
from course_lib.Base.NonPersonalizedRecommender import TopPop

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Building the single blocks
    item_cf_keywargs = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train)
    item_cf.fit(**item_cf_keywargs)

    top_pop = TopPop(URM_train)
    top_pop.fit()

    # Getting the profile
    bins = 10
    block_size, profile_length, sorted_users, group_mean_len = get_user_profile_demographic(URM_train, 10)

    hybrid = HybridDemographicRecommender(URM_train)
    for group_id in range(0, bins):
        start_pos = group_id * block_size
        if group_id < bins - 1:
            end_pos = min((group_id + 1) * block_size, len(profile_length))
        else:
            end_pos = len(profile_length)
        users_in_group = sorted_users[start_pos:end_pos]

        hybrid.add_user_group(group_id, users_in_group)

    for group_id in range(0, bins):
        if group_id < 2:
            hybrid.add_relation_recommender_group(recommender_object=top_pop, group_id=group_id)
        else:
            hybrid.add_relation_recommender_group(recommender_object=item_cf, group_id=group_id)

    hybrid.fit()

    path = "../../report/graphics/Nov24_20-35-00_item_cf_top_pop_user_profile_len/"
    # Plots
    basic_plots_recommender(hybrid, URM_train, URM_test, output_path_folder=path, save_on_file=True,
                            compare_top_pop_points=None,
                            is_compare_top_pop=True, demographic_list=None, demographic_list_name=None)
