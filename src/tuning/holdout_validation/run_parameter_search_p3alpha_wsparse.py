import os

from skopt.space import Real, Integer, Categorical

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt


def run_parameter_search_p3alpha_wsparse(recommender_class, URM_train, item_W_sparse, user_W_sparse,
                                         evaluator_validation, metric_to_optimize="MAP", n_cases=60, n_random_starts=20,
                                         output_folder_path="result_experiments/"):
    parameterSearch = SearchBayesianSkopt(recommender_class,
                                          evaluator_validation=evaluator_validation)

    hyperparameters_range_dictionary = {"topK": Integer(5, 1000),
                                        "alpha": Real(low=0, high=2, prior='uniform'),
                                        "normalize_similarity": Categorical([True, False])}

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, user_W_sparse, item_W_sparse],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    parameterSearch.search(recommender_input_args,
                           recommender_input_args_last_test=None,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           save_model="no",
                           output_folder_path=output_folder_path,
                           output_file_name_root=recommender_class.RECOMMENDER_NAME,
                           metric_to_optimize=metric_to_optimize
                           )
