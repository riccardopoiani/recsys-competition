import multiprocessing
import os
import traceback
from functools import partial

from course_lib.KNN import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model.KNN.NewUserKNNCFRecommender import NewUserKNNCFRecommender
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.tuning.cross_validation.CrossSearchBayesianSkopt import CrossSearchBayesianSkopt
from src.tuning.cross_validation.hyper_parameters_ranges import get_hyper_parameters_dictionary, \
    add_knn_similarity_type_hyper_parameters

KNN_RECOMMENDERS = [ItemKNNCFRecommender, UserKNNCFRecommender, NewUserKNNCFRecommender, ItemKNNCBFRecommender,
                    ItemKNNCBFCFRecommender, UserKNNCBFRecommender, UserKNNCBFCFRecommender]


def run_tuning_on_similarity_type(similarity_type,
                                  parameter_search: CrossSearchBayesianSkopt,
                                  hyper_parameters_dictionary: dict,
                                  recommender_class,
                                  recommender_input_args_list: list,
                                  n_cases,
                                  n_random_starts,
                                  resume_from_saved,
                                  output_folder_path,
                                  output_file_name_root,
                                  metric_to_optimize,
                                  allow_weighting=False):
    original_parameter_search_space = hyper_parameters_dictionary

    hyperparameters_range_dictionary = get_hyper_parameters_dictionary(recommender_class)
    hyperparameters_range_dictionary = add_knn_similarity_type_hyper_parameters(hyperparameters_range_dictionary,
                                                                                similarity_type, allow_weighting)
    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameter_search.search(recommender_input_args_list,
                            parameter_search_space=local_parameter_search_space,
                            n_cases=n_cases,
                            n_random_starts=n_random_starts,
                            resume_from_saved=resume_from_saved,
                            output_folder_path=output_folder_path,
                            output_file_name_root=output_file_name_root + "_" + similarity_type,
                            metric_to_optimize=metric_to_optimize)


def run_cv_parameter_search(recommender_class, URM_train_list, ICM_train_list=None, UCM_train_list=None,
                            ICM_name=None, UCM_name=None, metric_to_optimize="MAP",
                            evaluator_validation_list=None, output_folder_path="result_experiments/",
                            parallelize_search=True, n_cases=60, n_random_starts=20,
                            resume_from_saved=False, n_jobs=multiprocessing.cpu_count()):
    if len(evaluator_validation_list) != len(URM_train_list):
        raise ValueError("Number of evaluators does not coincide with the number of URM_train")

    # If directory does not exist, create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_class.RECOMMENDER_NAME
    if ICM_name is not None:
        output_file_name_root = output_file_name_root + "_{}".format(ICM_name)
    if UCM_name is not None:
        output_file_name_root = output_file_name_root + "_{}".format(UCM_name)

    try:

        parameter_search = CrossSearchBayesianSkopt(recommender_class,
                                                    evaluator_validation_list=evaluator_validation_list)

        # Set recommender_input_args for each fold
        recommender_input_args_list = []
        for i in range(len(URM_train_list)):
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_list[i]],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )
            if ICM_train_list is not None:
                recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS["ICM_train"] = ICM_train_list[i]
            if UCM_train_list is not None:
                recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS["UCM_train"] = UCM_train_list[i]
            recommender_input_args_list.append(recommender_input_args)

        # Get hyper parameters range dictionary by recommender_class
        hyperparameters_range_dictionary = get_hyper_parameters_dictionary(recommender_class)

        # -------------------- KNN RECOMMENDERS -------------------- #
        if recommender_class in KNN_RECOMMENDERS:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]
            run_tuning_on_similarity_type_partial = partial(run_tuning_on_similarity_type,
                                                            recommender_input_args_list=recommender_input_args_list,
                                                            hyper_parameters_dictionary=hyperparameters_range_dictionary,
                                                            recommender_class=recommender_class,
                                                            parameter_search=parameter_search,
                                                            n_cases=n_cases,
                                                            n_random_starts=n_random_starts,
                                                            resume_from_saved=resume_from_saved,
                                                            output_folder_path=output_folder_path,
                                                            output_file_name_root=output_file_name_root,
                                                            metric_to_optimize=metric_to_optimize,
                                                            allow_weighting=True)
            if parallelize_search:
                pool = multiprocessing.Pool(processes=n_jobs, maxtasksperchild=1)
                pool.map(run_tuning_on_similarity_type_partial, similarity_type_list)
                pool.close()
                pool.join()
            else:
                for similarity_type in similarity_type_list:
                    run_tuning_on_similarity_type_partial(similarity_type)
            return

        parameter_search.search(recommender_input_args_list,
                                parameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                resume_from_saved=resume_from_saved,
                                output_folder_path=output_folder_path,
                                output_file_name_root=output_file_name_root,
                                metric_to_optimize=metric_to_optimize)
    except Exception as e:
        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()
        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()
