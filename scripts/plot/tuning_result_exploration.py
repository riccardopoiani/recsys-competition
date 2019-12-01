from src.feature.demographics_content import get_user_demographic
from src.plots.recommender_plots import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.New_DataSplitter_leave_k_out import *
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender

from numpy.random import seed

from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    URM_all = data_reader.dataReader_object.get_URM_all()

    path = "../../report/hp_tuning/p3alpha/Nov23_14-29-55_k_out_value_3/"

    UCM_region = data_reader.dataReader_object.get_UCM_from_name('UCM_region')
    region_demographic = get_user_demographic(UCM_region, URM_all, 3)

    UCM_age = data_reader.dataReader_object.get_UCM_from_name('UCM_age')
    age_demographic = get_user_demographic(UCM_age, URM_all, 3)

    demographics = [region_demographic, age_demographic]
    demographics_names = ["region", "age"]

    basic_plots_from_tuning_results(path, P3alphaRecommender, URM_train, URM_test, save_on_file=True,
                                    demographic_list=demographics, demographic_list_name=demographics_names,
                                    output_path_folder="../../report/graphics/p3alpha/Nov23_14-29-55_k_out_value_3/")
