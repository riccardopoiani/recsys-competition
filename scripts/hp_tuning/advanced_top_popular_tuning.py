from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.dataframe_preprocesser import get_preprocessed_dataframe
from src.tuning.run_parameter_search_advanced_top_pop import run_parameter_search_advanced_top_pop
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_sub_class = data_reader.get_ICM_from_name("ICM_sub_class")

    df = get_preprocessed_dataframe("../../data/", keep_warm_only=True)

    # Setting evaluator
    #warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    #warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    #ignore_users = warm_users
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/advanced_top_pop/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    run_parameter_search_advanced_top_pop(URM_train=URM_train, data_frame_ucm=df, mapper=mapper,
                                          metric_to_optimize="MAP",
                                          evaluator_validation=evaluator,
                                          n_cases=35,
                                          output_folder_path=version_path,
                                          n_random_starts=10)
    print("...tuning ended")
