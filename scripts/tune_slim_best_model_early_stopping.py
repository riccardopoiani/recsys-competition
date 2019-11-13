from src.data_management.RecSys2018Reader import RecSys2018Reader
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from Base.Evaluation.Evaluator import *
from course_lib.ParameterTuning.run_parameter_search import *
from course_lib.Notebooks_utils.data_splitter import train_test_holdout

if __name__ == '__main__':
    # Data loading
    dataset = RecSys2018Reader("../data/train.csv", "../data/tracks.csv")
    dataset = DataSplitter_Warm_k_fold(dataset, n_folds=10)
    dataset.load_data()
    URM_train, URM_test = dataset.get_URM_train_for_test_fold(n_test_fold=8)

    # Early stop evaluator
    fake_URM_train, subset_URM_test = train_test_holdout(URM_test, train_perc=0.2)
    cutoff_list = [10]
    earlystopping_evaluator = EvaluatorHoldout(subset_URM_test, cutoff_list=cutoff_list)
    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": earlystopping_evaluator,
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    # Model creation
    best_parameter_dict = {'topK': 808, 'epochs': 1499, 'symmetric': True, 'sgd_mode': 'adagrad', 'lambda_i': 0.01,
                           'lambda_j': 1e-05,
                           'learning_rate': 0.008004365459157264}

    slim_recommender = SLIM_BPR_Cython(URM_train)
    slim_recommender.fit(**earlystopping_keywargs, epochs=1500, topK=808, symmetric=True, sgd_mode="adagrad",
                         lambda_i=0.01, lambda_j=1e-05, learning_rate=0.008004365459157264)

    # Evaluate algorithm
    hold_out_validation = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results = hold_out_validation.evaluateRecommender(slim_recommender)
    slim_recommender.saveModel(folder_path="../report/models/slim_bpr/", file_name="best_slim_early_stopped_model")

    print(str(results))
    print(results)

    # Print Results
    path = "../report/models/slim_bpr/best_slim_early_stopped_results"
    f = open(path, "w+")
    f.write(str(results))
    f.close()
