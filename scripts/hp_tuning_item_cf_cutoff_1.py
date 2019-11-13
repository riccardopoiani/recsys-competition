from src.data_management.RecSys2018Reader import RecSys2018Reader
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender

if __name__ == '__main__':
    dataset = RecSys2018Reader("../data/train.csv", "../data/tracks.csv")
    dataset = DataSplitter_Warm_k_fold(dataset, n_folds = 10)
    dataset.load_data(save_folder_path="../notebooks/Data_manager_split_datasets/data/warm_10_fold/original/")
    URM_train, URM_test = dataset.get_URM_train_for_test_fold(n_test_fold=9)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    runParameterSearch_Collaborative(RP3betaRecommender, URM_train, evaluator_validation=evaluator, metric_to_optimize="MAP")


