from datetime import datetime

from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_getter import get_warmer_UCM, get_user_demographic
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.plots.recommender_plots import basic_plots_recommender

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    best_parameters = {'topK': 3285, 'shrink': 1189, 'similarity': 'cosine',
                       'normalize': False, 'feature_weighting': 'BM25'}

    model = UserKNNCBFRecommender(URM_train=URM_train, UCM_train=UCM_all)
    model.fit(**best_parameters)

    version_path = "../../report/graphics/user_cbf_UCM_URM/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    # Plots
    demographic_age = get_user_demographic(UCM_age, URM_all, 3)
    demographic_region = get_user_demographic(UCM_region, URM_all, 3)
    demographic_list = [demographic_age, demographic_region]
    demographic_list_name = ['age', 'region']

    basic_plots_recommender(model, URM_train, URM_test, output_path_folder=version_path, save_on_file=True,
                            compare_top_pop_points=None,
                            is_compare_top_pop=True, demographic_list=demographic_list,
                            demographic_list_name=demographic_list_name)