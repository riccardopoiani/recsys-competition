import multiprocessing
import traceback
from functools import partial

from skopt.space import Integer, Categorical, Real

from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from src.model.FactorizationMachine.FieldAwareFMRecommender import FieldAwareFMRecommender
from src.model.MatrixFactorization.FunkSVDRecommender import FunkSVDRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
import os

from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.utils.general_utility_functions import get_project_root_path


def run_parameter_search_mf_collaborative(recommender_class, URM_train, UCM_train=None, UCM_name="NO_UCM",
                                          ICM_train=None, ICM_name="NO_ICM",
                                          URM_train_last_test=None,
                                          metric_to_optimize="PRECISION",
                                          evaluator_validation=None, evaluator_test=None,
                                          evaluator_validation_earlystopping=None,
                                          output_folder_path="result_experiments/", parallelize_search=True,
                                          n_cases=35, n_random_starts=5, resume_from_saved=False, save_model="best",
                                          approximate_recommender=None):
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

        output_file_name_root = recommender_class.RECOMMENDER_NAME + "_" + ICM_name + "_" + UCM_name

        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation,
                                              evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={}
        )
        hyperparameters_range_dictionary = {}

        if recommender_class is ImplicitALSRecommender:
            hyperparameters_range_dictionary["num_factors"] = Integer(300, 550)
            hyperparameters_range_dictionary["regularization"] = Real(low=1e-2, high=200, prior='log-uniform')
            hyperparameters_range_dictionary["epochs"] = Categorical([50])
            hyperparameters_range_dictionary["confidence_scaling"] = Categorical(["linear"])
            hyperparameters_range_dictionary["alpha"] = Real(low=1e-2, high=1e2, prior='log-uniform')

        if recommender_class is MF_BPR_Recommender:
            hyperparameters_range_dictionary["num_factors"] = Categorical([600])
            hyperparameters_range_dictionary["regularization"] = Real(low=1e-4, high=1e-1, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-2, high=1e-1, prior='log-uniform')
            hyperparameters_range_dictionary["epochs"] = Categorical([300])

        if recommender_class is FunkSVDRecommender:
            hyperparameters_range_dictionary["num_factors"] = Integer(50, 400)
            hyperparameters_range_dictionary["regularization"] = Real(low=1e-5, high=1e-0, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-2, high=1e-1, prior='log-uniform')
            hyperparameters_range_dictionary["epochs"] = Categorical([500])

        if recommender_class is LogisticMFRecommender:
            hyperparameters_range_dictionary["num_factors"] = Integer(20, 400)
            hyperparameters_range_dictionary["regularization"] = Real(low=1e-5, high=1e1, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-2, high=1e-1, prior='log-uniform')
            hyperparameters_range_dictionary["epochs"] = Categorical([300])

        if recommender_class is LightFMRecommender:
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['UCM_train'] = UCM_train
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['ICM_train'] = ICM_train

            hyperparameters_range_dictionary['no_components'] = Categorical([100])
            hyperparameters_range_dictionary['epochs'] = Categorical([100])

            run_light_fm_search(parameterSearch, recommender_input_args, hyperparameters_range_dictionary,
                                URM_train_last_test=URM_train_last_test, parallelize_search=parallelize_search,
                                n_cases=n_cases, n_random_starts=n_random_starts, output_folder_path=output_folder_path,
                                output_file_name_root=output_file_name_root, metric_to_optimize=metric_to_optimize,
                                save_model=save_model)

        if recommender_class is FieldAwareFMRecommender:
            if approximate_recommender is None:
                raise ValueError("approximate_recommender has to be set")
            root_path = get_project_root_path()
            train_svm_file_path = os.path.join(root_path, "resources", "fm_data", "URM_ICM_UCM_uncompressed.txt")
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['train_svm_file_path'] = train_svm_file_path
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['approximate_recommender'] = approximate_recommender
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['UCM_train'] = UCM_train
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['ICM_train'] = ICM_train

            hyperparameters_range_dictionary['epochs'] = Categorical([200])
            hyperparameters_range_dictionary['latent_factors'] = Integer(low=20, high=500)
            hyperparameters_range_dictionary['regularization'] = Real(low=10e-7, high=10e-1, prior="log-uniform")
            hyperparameters_range_dictionary['learning_rate'] = Real(low=10e-3, high=10e-1, prior="log-uniform")


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


def run_light_fm_search(parameter_search, recommender_input_args, hyperparameters_range_dictionary, URM_train_last_test,
                        parallelize_search=True, n_cases=35, n_random_starts=5, output_folder_path="./",
                        output_file_name_root=LightFMRecommender.RECOMMENDER_NAME, metric_to_optimize="MAP",
                        save_model="no"):
    optimizer_list = ['adagrad']

    if URM_train_last_test is not None:
        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
    else:
        recommender_input_args_last_test = None

    run_light_fm_on_optimizer_type_partial = partial(_run_light_fm_on_optimizer_type,
                                                     recommender_input_args=recommender_input_args,
                                                     parameter_search_space=hyperparameters_range_dictionary,
                                                     parameterSearch=parameter_search,
                                                     n_cases=n_cases,
                                                     n_random_starts=n_random_starts,
                                                     output_folder_path=output_folder_path,
                                                     output_file_name_root=output_file_name_root,
                                                     metric_to_optimize=metric_to_optimize,
                                                     save_model=save_model)

    if parallelize_search:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        pool.map(run_light_fm_on_optimizer_type_partial, optimizer_list)
        pool.close()
        pool.join()
    else:
        for optimizer in optimizer_list:
            run_light_fm_on_optimizer_type_partial(optimizer)


def _run_light_fm_on_optimizer_type(optimizer_type, recommender_input_args, parameter_search_space, parameterSearch,
                                    n_cases, n_random_starts, output_folder_path, output_file_name_root,
                                    metric_to_optimize, save_model):
    original_parameter_search_space = parameter_search_space

    hyperparameters_range_dictionary = {'learning_schedule': Categorical([optimizer_type]),
                                        'learning_rate': Real(low=1e-3, high=1e-1, prior="log-uniform"),
                                        'user_alpha': Real(low=1e-5, high=1e1, prior="log-uniform"),
                                        'item_alpha': Real(low=1e-5, high=1e1, prior="log-uniform")}
    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=local_parameter_search_space,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root + "_" + optimizer_type,
                           metric_to_optimize=metric_to_optimize,
                           save_model=save_model)
