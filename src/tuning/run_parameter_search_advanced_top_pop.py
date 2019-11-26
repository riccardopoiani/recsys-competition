from skopt.space import Integer, Categorical, Real
import os

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from src.model.FallbackRecommender import AdvancedTopPopular

from functools import partial


def run_parameter_search_advanced_top_pop(URM_train, data_frame_ucm, mapper, verbose=0, seed=69420, n_jobs=1,
                                          output_folder_path="result_experiments/",
                                          evaluator_validation=None, evaluator_test=None, n_cases=35,
                                          n_random_starts=5, metric_to_optimize="MAP"):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()

    hyperparameters_range_dictionary = {"clustering_method": Categorical(['kmodes', 'kproto']),
                                        'n_clusters': Integer(1, 20),
                                        'init_method': Categorical(["Huang", "random", "Cao"]), 'verbose': 0,
                                        'seed': seed,
                                        'n_jobs': n_jobs}

    output_file_name_root = AdvancedTopPopular.RECOMMENDER_NAME

    parameterSearch = SearchBayesianSkopt(AdvancedTopPopular, evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, data_frame_ucm, mapper],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="no")
