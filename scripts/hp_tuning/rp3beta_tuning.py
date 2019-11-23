from src.data_management.RecSys2019Reader import RecSys2019Reader
from course_lib.Base.Evaluation.Evaluator import *
from course_lib.ParameterTuning.run_parameter_search import *
from src.data_management.New_DataSplitter_leave_k_out import *
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from datetime import datetime
from numpy.random import seed

SEED = 69420

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed(SEED)

    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False, force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Reset seed for hyper-parameter tuning
    seed()

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/rp3beta/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    runParameterSearch_Collaborative(URM_train=URM_train, recommender_class=RP3betaRecommender,
                                     evaluator_validation=evaluator,
                                     metric_to_optimize="MAP",
                                     output_folder_path=version_path,
                                     n_cases=35)
    print("...tuning ended")