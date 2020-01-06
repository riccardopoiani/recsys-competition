import multiprocessing
from functools import partial

from skopt.space import Real, Integer, Categorical
import os

from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.model.SLIM.SSLIM_BPR import SSLIM_BPR
from src.tuning.cross_validation.hyper_parameters_ranges import get_hyper_parameters_dictionary


def run_KNNRecommender_on_similarity_type(similarity_type, parameterSearch,
                                          parameter_search_space,
                                          recommender_input_args,
                                          n_cases,
                                          n_random_starts,
                                          resume_from_saved,
                                          save_model,
                                          output_folder_path,
                                          output_file_name_root,
                                          metric_to_optimize,
                                          allow_weighting=False,
                                          recommender_input_args_last_test=None):
    original_parameter_search_space = parameter_search_space

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(1, 1000)
    hyperparameters_range_dictionary["shrink"] = Integer(0, 1500)
    hyperparameters_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparameters_range_dictionary["normalize"] = Categorical([True, False])

    is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]

    if similarity_type == "asymmetric":
        hyperparameters_range_dictionary["asymmetric_alpha"] = Real(low=0, high=2, prior='uniform')
        hyperparameters_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparameters_range_dictionary["tversky_alpha"] = Real(low=0, high=2, prior='uniform')
        hyperparameters_range_dictionary["tversky_beta"] = Real(low=0, high=2, prior='uniform')
        hyperparameters_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "euclidean":
        hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
        hyperparameters_range_dictionary["normalize_avg_row"] = Categorical([True, False])
        hyperparameters_range_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])

    if not is_set_similarity:

        if allow_weighting:
            hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])
            # hyperparameters_range_dictionary["interactions_feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=local_parameter_search_space,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           resume_from_saved=resume_from_saved,
                           save_model=save_model,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root + "_" + similarity_type,
                           metric_to_optimize=metric_to_optimize,
                           recommender_input_args_last_test=recommender_input_args_last_test)


def run_parameter_search_item_content(recommender_class, URM_train, ICM_object, ICM_name, URM_train_last_test=None,
                                      n_cases=30, n_random_starts=5, resume_from_saved=False, save_model="best",
                                      evaluator_validation=None, evaluator_test=None, metric_to_optimize="PRECISION",
                                      output_folder_path="result_experiments/", parallelizeKNN=False,
                                      allow_weighting=True,
                                      similarity_type_list=None):
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()
    ICM_object = ICM_object.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)

    if similarity_type_list is None:
        similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_object],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    if URM_train_last_test is not None:
        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
    else:
        recommender_input_args_last_test = None

    if recommender_class.RECOMMENDER_NAME == SSLIM_BPR.RECOMMENDER_NAME:
        hyper_parameters_range = get_hyper_parameters_dictionary(recommender_class)
        parameterSearch.search(recommender_input_args, hyper_parameters_range,
                               metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts,
                               output_folder_path=output_folder_path, output_file_name_root=output_file_name_root,
                               save_model=save_model, resume_from_saved=resume_from_saved,
                               recommender_input_args_last_test=recommender_input_args_last_test)
        return

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                               recommender_input_args=recommender_input_args,
                                                               parameter_search_space={},
                                                               parameterSearch=parameterSearch,
                                                               n_cases=n_cases,
                                                               n_random_starts=n_random_starts,
                                                               resume_from_saved=resume_from_saved,
                                                               save_model=save_model,
                                                               output_folder_path=output_folder_path,
                                                               output_file_name_root=output_file_name_root,
                                                               metric_to_optimize=metric_to_optimize,
                                                               allow_weighting=allow_weighting,
                                                               recommender_input_args_last_test=recommender_input_args_last_test)

    if parallelizeKNN:
        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

        pool.close()
        pool.join()

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)
