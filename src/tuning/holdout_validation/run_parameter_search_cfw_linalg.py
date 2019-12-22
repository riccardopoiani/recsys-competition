from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Real, Integer, Categorical
from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
import os

from src.model.FeatureWeighting.User_CFW_D_Similarity_Linalg import User_CFW_D_Similarity_Linalg


def run_parameter_search(URM_train, ICM_all, W_sparse_CF, evaluator_test,
                         metric_to_optimize="MAP", n_cases=10, n_random_starts=3,
                         output_folder_path="result_experiments/"):
    recommender_class = CFW_D_Similarity_Linalg

    parameterSearch = SearchBayesianSkopt(recommender_class,
                                          evaluator_validation=evaluator_test)

    hyperparameters_range_dictionary = {"topK": Integer(1000, 2000),
                                        "add_zeros_quota": Real(low=0, high=0.1, prior='uniform'),
                                        "normalize_similarity": Categorical([True, False])}

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_all, W_sparse_CF],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Clone data structure to perform the fitting with the best hyper parameters on train + validation data
    recommender_input_args_last_test = recommender_input_args.copy()
    recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train

    parameterSearch.search(recommender_input_args,
                           recommender_input_args_last_test=recommender_input_args_last_test,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           save_model="no",
                           output_folder_path=output_folder_path,
                           output_file_name_root=recommender_class.RECOMMENDER_NAME,
                           metric_to_optimize=metric_to_optimize
                           )
