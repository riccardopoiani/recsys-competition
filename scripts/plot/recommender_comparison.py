from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical, merge_UCM
from src.data_management.data_getter import get_warmer_UCM
from src.model.FallbackRecommender.AdvancedTopPopular import AdvancedTopPopular
from src.plots.recommender_plots import *
from src.data_management.dataframe_preprocesser import get_preprocessed_dataframe
from src.model import best_models
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data reading
    data_reader = RecSys2019Reader()
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']
    df = get_preprocessed_dataframe("../../data/", keep_warm_only=True)

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    top_pop = TopPop(URM_train)
    top_pop.fit()

    advanced_top_pop_keywargs =  {'clustering_method': 'kmodes', 'n_clusters': 7, 'init_method': 'Cao'}
    advanced_top_pop = AdvancedTopPopular(URM_train, df, mapper)
    advanced_top_pop.fit(**advanced_top_pop_keywargs)

    recommender_list = []
    recommender_list.append(top_pop)
    recommender_list.append(advanced_top_pop)
    recommender_list.append(best_models.UserCBF().get_model(URM_train, UCM_all))

    # Plotting the comparison based on user activity
    #plot_compare_recommender_user_group()
    plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True)