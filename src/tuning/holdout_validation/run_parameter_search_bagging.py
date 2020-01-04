import multiprocessing
from functools import partial

import os

from skopt.space import Categorical, Real, Integer

from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender, \
    BaggingMergeUserSimilarityRecommender


def run_parameter_search_bagging(recommender_class, URM_train, constructor_kwargs, fit_kwargs,
                                 URM_train_last_test=None,
                                 n_cases=30, n_random_starts=5, resume_from_saved=False, save_model="no",
                                 evaluator_validation=None, evaluator_test=None,
                                 metric_to_optimize="PRECISION",
                                 output_folder_path="result_experiments/", parallelizeKNN=False):
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_" + \
                            constructor_kwargs['recommender_class'].RECOMMENDER_NAME

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        CONSTRUCTOR_KEYWORD_ARGS=constructor_kwargs,
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS=fit_kwargs
    )

    hyperparameters_range = {}
    hyperparameters_range['num_models'] = Integer(10, 100)

    if recommender_class in [BaggingMergeItemSimilarityRecommender, BaggingMergeUserSimilarityRecommender]:
        hyperparameters_range['topK'] = Integer(low=1, high=3000)

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize,
                           save_model=save_model,
                           resume_from_saved=resume_from_saved)
