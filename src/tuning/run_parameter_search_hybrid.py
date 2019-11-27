import multiprocessing
import os
from functools import partial

from skopt.space import Integer, Categorical, Real

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender
from src.model.HybridRecommender.AbstractHybridRecommender import AbstractHybridRecommender
from src.tuning.SearchBayesianSkoptObject import SearchBayesianSkoptObject


def run_hybrid_rank_based_rs_on_strategy(strategy_type, parameterSearch,
                                         parameter_search_space,
                                         recommender_input_args,
                                         n_cases,
                                         n_random_starts,
                                         output_folder_path,
                                         output_file_name_root,
                                         metric_to_optimize):
    original_parameter_search_space = parameter_search_space

    hyperparameters_range_dictionary = {"strategy": Categorical([strategy_type]), "multiplier_cutoff": Integer(1, 10)}

    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=local_parameter_search_space,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root + "_" + strategy_type,
                           metric_to_optimize=metric_to_optimize, save_model="no")


def run_parameter_search_hybrid(recommender_object: AbstractHybridRecommender, metric_to_optimize="PRECISION",
                                evaluator_validation=None, output_folder_path="result_experiments/",
                                parallelizeKNN=True, n_cases=35, n_random_starts=5):
    # Create folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_object.RECOMMENDER_NAME

    parameterSearch = SearchBayesianSkoptObject(recommender_object, evaluator_validation=evaluator_validation)

    # Set hyperparameters
    hyperparameters_range_dictionary = {}
    for model_name in recommender_object.get_recommender_names():
        hyperparameters_range_dictionary[model_name] = Real(0, 1)

    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    if recommender_object.RECOMMENDER_NAME == "HybridRankBasedRecommender":

        strategies = HybridRankBasedRecommender.get_possible_strategies()

        run_hybrid_rank_based_rs_on_strategy_partial = partial(run_hybrid_rank_based_rs_on_strategy,
                                                               recommender_input_args=recommender_input_args,
                                                               parameter_search_space=hyperparameters_range_dictionary,
                                                               parameterSearch=parameterSearch,
                                                               n_cases=n_cases,
                                                               n_random_starts=n_random_starts,
                                                               output_folder_path=output_folder_path,
                                                               output_file_name_root=output_file_name_root,
                                                               metric_to_optimize=metric_to_optimize)

        if parallelizeKNN:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
            pool.map(run_hybrid_rank_based_rs_on_strategy_partial, strategies)

            pool.close()
            pool.join()

        else:
            for similarity_type in strategies:
                run_hybrid_rank_based_rs_on_strategy_partial(similarity_type)
        return

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root,
                           metric_to_optimize=metric_to_optimize, save_model="no")
