import traceback

from skopt.space import Integer, Categorical, Real

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
import os


def run_parameter_search_collaborative(recommender_class, URM_train, URM_train_last_test=None,
                                     metric_to_optimize="PRECISION",
                                     evaluator_validation=None, evaluator_test=None,
                                     evaluator_validation_earlystopping=None,
                                     output_folder_path="result_experiments/", parallelizeKNN=True,
                                     n_cases=35, n_random_starts=5, resume_from_saved=False, save_model="best",
                                     allow_weighting=True,
                                     similarity_type_list=None):
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

    URM_train = URM_train.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation,
                                              evaluator_test=evaluator_test)

        if recommender_class is ImplicitALSRecommender:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(10, 400)
            hyperparameters_range_dictionary["regularization"] = Real(low=1e-3, high=1e2, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={"epochs": 50}
            )

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_input_args,
                               parameter_search_space=hyperparameters_range_dictionary,
                               n_cases=n_cases,
                               n_random_starts=n_random_starts,
                               resume_from_saved=resume_from_saved,
                               save_model=save_model,
                               output_folder_path=output_folder_path,
                               output_file_name_root=output_file_name_root,
                               metric_to_optimize=metric_to_optimize,
                               recommender_input_args_last_test=recommender_input_args_last_test)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()
