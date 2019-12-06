import multiprocessing
from functools import partial

from skopt.space import Real, Integer, Categorical, Space
import os

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.model.HybridRecommender.HybridMixedSimilarityRecommender import HybridMixedSimilarityRecommender
from src.tuning.SearchBayesianSkoptObject import SearchBayesianSkoptObject


def run_parameter_search_mixed_similarity_item(recommender_object: HybridMixedSimilarityRecommender, URM_train,
                                               output_folder_path="result_experiments/",
                                               evaluator_validation=None, evaluator_test=None, n_cases=35,
                                               n_random_starts=5, metric_to_optimize="MAP"):
    print("Start search")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_object.RECOMMENDER_NAME

    hyperparameters_range_dictionary = {"topK": Integer(1, 2000), "alpha1": Real(0, 1),
                                        "alpha2": Real(0, 1), "alpha3": Real(0, 1)}

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameterSearch = SearchBayesianSkoptObject(recommender_object, evaluator_validation=evaluator_validation)

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="no")


def run_parameter_search_mixed_similarity_user(recommender_object: HybridMixedSimilarityRecommender, URM_train,
                                               output_folder_path="result_experiments/",
                                               evaluator_validation=None, evaluator_test=None, n_cases=35,
                                               n_random_starts=5, metric_to_optimize="MAP"):
    print("Start search")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_object.RECOMMENDER_NAME

    hyperparameters_range_dictionary = {"topK": Integer(1, 2000), "alpha1": Real(0, 1),
                                        "alpha2": Real(0, 1)}

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameterSearch = SearchBayesianSkoptObject(recommender_object, evaluator_validation=evaluator_validation)

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="no")
