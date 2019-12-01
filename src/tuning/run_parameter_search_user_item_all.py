import itertools
import multiprocessing
import os
from functools import partial

from skopt.space import Integer, Categorical, Real

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender


def _get_feature_weighting_for_similarity_type(similarity_type, allow_weighting):
    is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]
    if not is_set_similarity:
        if allow_weighting:
            return Categorical(["none", "BM25", "TF-IDF"])
    return Categorical(["none"])


def run_user_item_all_on_combination_similarity_type(similarity_types, parameterSearch,
                                                     parameter_search_space,
                                                     recommender_input_args,
                                                     n_cases,
                                                     n_random_starts,
                                                     output_folder_path,
                                                     output_file_name_root,
                                                     metric_to_optimize,
                                                     allow_user_weighting=False,
                                                     allow_item_weighting=False):
    user_similarity_type = similarity_types[0]
    item_similarity_type = similarity_types[1]
    original_parameter_search_space = parameter_search_space

    hyperparameters_range_dictionary = {"user_similarity_type": Categorical([user_similarity_type]),
                                        "item_similarity_type": Categorical([item_similarity_type]),
                                        "user_feature_weighting": _get_feature_weighting_for_similarity_type(
                                            user_similarity_type, allow_user_weighting),
                                        "item_feature_weighting": _get_feature_weighting_for_similarity_type(
                                            item_similarity_type, allow_item_weighting),
                                        "user_normalize": Categorical([True, False]),
                                        "item_normalize": Categorical([True, False])
                                        }
    if user_similarity_type in ["asymmetric"]:
        hyperparameters_range_dictionary["user_asymmetric_alpha"] = Real(low=0, high=4, prior='uniform')
        hyperparameters_range_dictionary["user_normalize"] = Categorical([True])

    if item_similarity_type in ["asymmetric"]:
        hyperparameters_range_dictionary["item_asymmetric_alpha"] = Real(low=0, high=4, prior='uniform')
        hyperparameters_range_dictionary["item_normalize"] = Categorical([True])

    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=local_parameter_search_space,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root + "_item_" +
                                                 item_similarity_type + "_user_" + user_similarity_type,
                           metric_to_optimize=metric_to_optimize)


def run_parameter_search_user_item_all(recommender_class, URM_train, UCM_train, ICM_train, UCM_name,
                                       ICM_name, metric_to_optimize="PRECISION", evaluator_validation=None,
                                       output_folder_path="result_experiments/", parallelizeKNN=True, n_cases=60,
                                       n_random_starts=10, similarity_type_list=None):
    # Create folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(UCM_name) + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation)

    if similarity_type_list is None:
        similarity_type_list = ['jaccard', 'asymmetric', "cosine"]

    # Set hyperparameters
    hyperparameters_range_dictionary = {"user_topK": Integer(5, 2000), "user_shrink": Integer(0, 2000),
                                        "item_topK": Integer(5, 2000), "item_shrink": Integer(0, 2000)}


    # Set args for recommender
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, UCM_train, ICM_train],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    run_user_item_all_on_combination_similarity_type_partial = partial(run_user_item_all_on_combination_similarity_type,
                                                                       recommender_input_args=recommender_input_args,
                                                                       parameter_search_space=hyperparameters_range_dictionary,
                                                                       parameterSearch=parameterSearch,
                                                                       n_cases=n_cases,
                                                                       n_random_starts=n_random_starts,
                                                                       output_folder_path=output_folder_path,
                                                                       output_file_name_root=output_file_name_root,
                                                                       metric_to_optimize=metric_to_optimize,
                                                                       allow_user_weighting=True,
                                                                       allow_item_weighting=True)

    if parallelizeKNN:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        pool.map(run_user_item_all_on_combination_similarity_type_partial,
                 list(itertools.product(*[similarity_type_list, similarity_type_list])))
        pool.close()
        pool.join()
    else:
        for user_similarity_type in similarity_type_list:
            for item_similarity_type in similarity_type_list:
                run_user_item_all_on_combination_similarity_type_partial(user_similarity_type, item_similarity_type)
