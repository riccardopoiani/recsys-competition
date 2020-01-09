import multiprocessing
import os
import traceback

from skopt.space import Real, Integer

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from src.tuning.cross_validation.CrossSearchBayesianSkoptObject import CrossSearchBayesianSkoptObject


def run_cv_parameter_search_hybrid_avg(recommender_object_list, URM_train_list, ICM_train_list=None,
                                       UCM_train_list=None,
                                       ICM_name=None, UCM_name=None, metric_to_optimize="MAP",
                                       evaluator_validation_list=None, output_folder_path="result_experiments/",
                                       parallelize_search=True, n_cases=60, n_random_starts=20,
                                       resume_from_saved=False, n_jobs=multiprocessing.cpu_count(),
                                       map_max=0):
    if len(evaluator_validation_list) != len(URM_train_list):
        raise ValueError("Number of evaluators does not coincide with the number of URM_train")

    # If directory does not exist, create it
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_file_name_root = recommender_object_list[0].RECOMMENDER_NAME
    if ICM_name is not None:
        output_file_name_root = output_file_name_root + "_{}".format(ICM_name)
    if UCM_name is not None:
        output_file_name_root = output_file_name_root + "_{}".format(UCM_name)

    try:

        parameter_search = CrossSearchBayesianSkoptObject(recommender_object_list,
                                                          evaluator_validation_list=evaluator_validation_list)

        # Set recommender_input_args for each fold
        recommender_input_args_list = []
        for i in range(len(URM_train_list)):
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )
            recommender_input_args_list.append(recommender_input_args)

        # Get hyper parameters range dictionary by recommender_class
        hyperparameters_range_dictionary = {}
        for model_name in recommender_object_list[0].get_recommender_names():
            if map_max == 0:
                hyperparameters_range_dictionary[model_name] = Real(0, 1)
            else:
                if model_name == "ItemAvg":
                    hyperparameters_range_dictionary[model_name] = Integer((map_max//2)-5, map_max)
                else:
                    hyperparameters_range_dictionary[model_name] = Integer(0, map_max-3)

        parameter_search.search(recommender_input_args_list,
                                parameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                resume_from_saved=resume_from_saved,
                                output_folder_path=output_folder_path,
                                output_file_name_root=output_file_name_root,
                                metric_to_optimize=metric_to_optimize)
    except Exception as e:
        print("On recommender {} Exception {}".format(recommender_object_list, str(e)))
        traceback.print_exc()
        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_object_list, str(e)))
        error_file.close()
