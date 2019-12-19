import os

from skopt.space import Real

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from src.model.FeatureWeighting.SearchFieldWeightICMRecommender import SearchFieldWeightICMRecommender
from src.model.FeatureWeighting.SearchFieldWeightUCMRecommender import SearchFieldWeightUCMRecommender


def run_parameter_search_field_UCM_weight(URM_train, UCM_train, base_recommender_class,
                                          base_recommender_parameter, user_feature_to_range_mapper,
                                          output_folder_path="result_experiments/",
                                          evaluator_validation=None, evaluator_test=None, n_cases=35,
                                          n_random_starts=5, metric_to_optimize="MAP"):
    recommender_class = SearchFieldWeightUCMRecommender

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME

    hyperparameters_range_dictionary = {}
    for user_feature_name in user_feature_to_range_mapper.keys():
        hyperparameters_range_dictionary[user_feature_name] = Real(low=0, high=2, prior="uniform")

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UCM_train, base_recommender_class, base_recommender_parameter,
                                     user_feature_to_range_mapper],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation)

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="no")


def run_parameter_search_field_ICM_weight(URM_train, ICM_train, base_recommender_class,
                                          base_recommender_parameter, item_feature_to_range_mapper,
                                          output_folder_path="result_experiments/",
                                          evaluator_validation=None, evaluator_test=None, n_cases=35,
                                          n_random_starts=5, metric_to_optimize="MAP"):
    recommender_class = SearchFieldWeightICMRecommender

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME

    hyperparameters_range_dictionary = {}
    for user_feature_name in item_feature_to_range_mapper.keys():
        hyperparameters_range_dictionary[user_feature_name] = Real(low=0, high=2, prior="uniform")

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_train, base_recommender_class, base_recommender_parameter,
                                     item_feature_to_range_mapper],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation)

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="no")