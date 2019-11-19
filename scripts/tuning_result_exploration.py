from src.model_management.tuning_result_exploration import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.DataPreprocessing import *
from course_lib.KNN.UserKNNCFRecommender import *

if __name__ == '__main__':
    data_reader = RecSys2019Reader()
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_users=3)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    path = "../report/hp_tuning/user_cf/Nov19_14-58-37_k_out_value_3/"
    explore_tuning_result(path, UserKNNCFRecommender, URM_train, URM_test, save_on_file=True,
                          output_path_folder="../report/hp_tuning/user_cf/Nov19_14-58-37_k_out_value_3/graphics/")