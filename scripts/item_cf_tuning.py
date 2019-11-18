from src.data_management.RecSys2019Reader import RecSys2019Reader
from course_lib.Base.Evaluation.Evaluator import *
from course_lib.ParameterTuning.run_parameter_search import *
from src.data_management.New_DataSplitter_leave_k_out import *
import os
from datetime import datetime
from src.data_management.DataPreprocessing import *

if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader()
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_users=1)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False, force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # HP tuning
    print("Start tuning...")
    version_path = "../report/hp_tuning/item_cf/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    version_path = os.path.join(version_path, now)

    runParameterSearch_Collaborative(URM_train=URM_train, recommender_class=ItemKNNCFRecommender,
                                     evaluator_validation=evaluator,
                                     metric_to_optimize="MAP",
                                     output_folder_path=version_path,
                                     n_cases=35)
    print("...tuning ended")