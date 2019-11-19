from src.data_management.RecSys2019Reader import RecSys2019Reader
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.ParameterTuning.run_parameter_search import runParameterSearch_Content
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from datetime import datetime
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from numpy.random import seed

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    SEED = 69420
    seed(SEED)

    # Data loading

    data_reader = RecSys2019Reader("../../data/")
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_users=3)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False, force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all = data_reader.get_ICM_from_name("ICM_all")

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/item_cbf/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + now

    runParameterSearch_Content(URM_train=URM_train, ICM_object=ICM_all, ICM_name="ICM_all",
                               recommender_class=ItemKNNCBFRecommender,
                               evaluator_validation=evaluator,
                               metric_to_optimize="MAP",
                               output_folder_path=version_path,
                               parallelizeKNN=True,
                               n_cases=35)
    print("...tuning ended")
