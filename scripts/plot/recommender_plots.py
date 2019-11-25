from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.plots.recommender_plots import basic_plots_recommender
from course_lib.Base.NonPersonalizedRecommender import TopPop, GlobalEffects
from datetime import datetime

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    global_effect = GlobalEffects(URM_train)
    global_effect.fit()

    version_path = "../../report/graphics/global_effects/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    # Plots
    basic_plots_recommender(global_effect, URM_train, URM_test, output_path_folder=version_path, save_on_file=True,
                            compare_top_pop_points=None,
                            is_compare_top_pop=True, demographic_list=None, demographic_list_name=None)
