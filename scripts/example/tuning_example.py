from src.data_management.RecSys2019Reader import RecSys2019Reader
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.Base.Evaluation.Evaluator import *
from course_lib.ParameterTuning.run_parameter_search import *

if __name__ == '__main__':
    # Data loading
    dataset = RecSys2019Reader("../data/train.csv", "../data/tracks.csv")
    dataset = DataSplitter_Warm_k_fold(dataset, n_folds = 10)
    dataset.load_data()
    URM_train, URM_test = dataset.get_URM_train_for_test_fold(n_test_fold=8)

    # Hyperparameter tuning
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # Sarebbe meglio un subset della matrice per fare early stopping, in modo che vada molto più velcoe...

    print("Start tuning...")
    runParameterSearch_Collaborative(URM_train=URM_train, recommender_class=MatrixFactorization_BPR_Cython, evaluator_validation=evaluator,
                                     metric_to_optimize="MAP",
                                     output_folder_path="../report/hp_tuning/hp_tuning_MF_BPR_cutoff_10_MAP/",
                                     n_cases=5)
    print("...tuning ended")