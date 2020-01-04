import os

from skopt.space import Real, Integer

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from src.model.Ensemble.Boosting.LightGBMRecommender import LightGBMRecommender


def run_parameter_search_lightgbm(URM_train, X_train, y_train, X_test, y_test, cutoff_test,
                                  categorical_features=None,
                                  num_iteration=10000, early_stopping_iteration=150,
                                  objective="lambdarank", verbose=True,
                                  output_folder_path="result_experiments/",
                                  evaluator_validation=None, n_cases=35,
                                  n_random_starts=5, metric_to_optimize="MAP"):
    recommender_class = LightGBMRecommender

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-1, prior="log-uniform")
    hyperparameters_range_dictionary["min_gain_to_split"] = Real(low=1e-4, high=1e-1, prior="log-uniform")
    hyperparameters_range_dictionary["reg_l1"] = Real(low=1e-7, high=1e1, prior="log-uniform")
    hyperparameters_range_dictionary["reg_l2"] = Real(low=1e-7, high=1e1, prior="log-uniform")
    hyperparameters_range_dictionary["max_depth"] = Integer(low=4, high=100)
    hyperparameters_range_dictionary["min_data_in_leaf"] = Integer(low=5, high=100)
    hyperparameters_range_dictionary["bagging_freq"] = Integer(low=2, high=100)
    hyperparameters_range_dictionary["num_leaves"] = Integer(low=16, high=400)
    hyperparameters_range_dictionary["bagging_fraction"] = Real(low=0.1, high=0.9, prior="log-uniform")
    hyperparameters_range_dictionary["feature_fraction"] = Real(low=0.1, high=0.9, prior="log-uniform")

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, X_train, y_train, X_test, y_test, cutoff_test, categorical_features],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={"num_iteration": num_iteration, "early_stopping_round": early_stopping_iteration,
                          "verbose": verbose, "objective": objective}
    )

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation)

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="best")
