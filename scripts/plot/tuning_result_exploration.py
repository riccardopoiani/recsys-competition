from src.plots.recommender_plots import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.New_DataSplitter_leave_k_out import *
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

if __name__ == '__main__':
    data_reader = RecSys2019Reader()
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    path = "../report/hp_tuning/slim_bpr/Nov23_23-24-57_k_out_value_3/"
    basic_plots_from_tuning_results(path, SLIM_BPR_Cython, URM_train, URM_test, save_on_file=True,
                                    output_path_folder="../report/graphics/slim_bpr/Nov23_23-24-57_k_out_value_3/")
