#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/03/2018

@author: Maurizio Ferrari Dacrema
"""

import os
import time
import traceback
from typing import List

import numpy as np

from course_lib.Base.DataIO import DataIO
from course_lib.Base.Evaluation.Evaluator import Evaluator
from course_lib.ParameterTuning.SearchAbstractClass import _compute_avg_time_non_none_values


def compute_mean_std_result_dict(result_dicts):
    metric_list = list(result_dicts[0].keys())

    # ------- Calculate mean of each metrics -------
    mean_result_dict: dict = {}
    # Initialization
    for metric in metric_list:
        mean_result_dict[metric] = 0

    # Sum operation
    for result_dict in result_dicts:
        for metric in metric_list:
            mean_result_dict[metric] = mean_result_dict[metric] + result_dict[metric]

    # Average
    for metric in metric_list:
        mean_result_dict[metric] = mean_result_dict[metric] / len(result_dicts)

    # ------- Calculate std of each metrics -------
    # Initialization
    std_result_dict = {}
    for metric in metric_list:
        std_result_dict[metric] = 0

    # Sum operation
    for idx, result_dict in enumerate(result_dicts):
        for metric in metric_list:
            mean_value = mean_result_dict[metric]
            squared_difference = (result_dict[metric] - mean_value) ** 2
            std_result_dict[metric] = std_result_dict[metric] + squared_difference

    # Average and root square
    for metric in metric_list:
        std_result_dict[metric] = (std_result_dict[metric] / len(result_dicts)) ** (1 / 2)

    return mean_result_dict, std_result_dict


def get_result_string(mean_result_dict, std_result_dict, n_mean_decimals=7, n_std_decimals=4):
    output_str = ""

    for metric in mean_result_dict.keys():
        output_str += "{}: {:.{n_mean_decimals}f}\u00B1{:.{n_std_decimals}f}, ".format(metric, mean_result_dict[metric],
                                                                                       std_result_dict[metric],
                                                                                       n_mean_decimals=n_mean_decimals,
                                                                                       n_std_decimals=n_std_decimals)

    return output_str


class CrossSearchAbstractClass(object):
    ALGORITHM_NAME = "CrossSearchAbstractClass"

    # Value to be assigned to invalid configuration or if an Exception is raised
    INVALID_CONFIG_VALUE = np.finfo(np.float16).max

    def __init__(self, recommender_class,
                 evaluator_validation_list: List[Evaluator] = None,
                 verbose=True):

        super(CrossSearchAbstractClass, self).__init__()

        self.recommender_class = recommender_class
        self.verbose = verbose
        self.log_file = None

        self.results_test_best = {}
        self.parameter_dictionary_best = {}

        self.evaluator_validation_list: List[Evaluator] = evaluator_validation_list

    def search(self, recommender_input_args_list,
               parameter_search_space,
               metric_to_optimize="MAP",
               n_cases=None,
               output_folder_path=None,
               output_file_name_root=None,
               parallelize=False,
               save_metadata=True,
               ):

        raise NotImplementedError("Function search not implemented for this class")

    def _set_search_attributes(self, recommender_input_args,
                               metric_to_optimize,
                               output_folder_path,
                               output_file_name_root,
                               resume_from_saved,
                               save_metadata,
                               n_cases):

        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        self.log_file = open(
            self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")

        self.recommender_input_args_list = recommender_input_args
        self.metric_to_optimize = metric_to_optimize
        self.resume_from_saved = resume_from_saved
        self.save_metadata = save_metadata

        self.model_counter = 0
        self._init_metadata_dict(n_cases=n_cases)

        if self.save_metadata:
            self.dataIO = DataIO(folder_path=self.output_folder_path)

    def _init_metadata_dict(self, n_cases):

        self.metadata_dict = {"algorithm_name_search": self.ALGORITHM_NAME,
                              "algorithm_name_recommender": self.recommender_class.RECOMMENDER_NAME,
                              "exception_list": [None] * n_cases,

                              "hyperparameters_list": [None] * n_cases,
                              "hyperparameters_best": None,
                              "hyperparameters_best_index": None,

                              "result_on_validation_list": [None] * n_cases,
                              "result_on_validation_best": None,
                              "std_result_on_validation_list": [None] * n_cases,
                              "std_result_on_validation_best": None,

                              "time_on_train_list": [None] * n_cases,
                              "time_on_train_total": 0.0,
                              "time_on_train_avg": 0.0,

                              "time_on_validation_list": [None] * n_cases,
                              "time_on_validation_total": 0.0,
                              "time_on_validation_avg": 0.0,

                              "time_on_test_list": [None] * n_cases,
                              "time_on_test_total": 0.0,
                              "time_on_test_avg": 0.0,

                              "result_on_last": None,
                              "time_on_last_train": None,
                              "time_on_last_test": None,
                              }

    def _print(self, string):

        if self.verbose:
            print(string)

    def _write_log(self, string):

        self._print(string)

        if self.log_file is not None:
            self.log_file.write(string)
            self.log_file.flush()

    def _fit_model(self, current_fit_parameters, recommender_input_args):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

        recommender_instance.fit(*recommender_input_args.FIT_POSITIONAL_ARGS,
                                 **recommender_input_args.FIT_KEYWORD_ARGS,
                                 **current_fit_parameters)

        train_time = time.time() - start_time

        return recommender_instance, train_time

    def _evaluate_on_validation(self, current_fit_parameters):

        result_dicts = []
        evaluation_times = np.zeros(len(self.evaluator_validation_list))
        train_times = np.zeros(len(self.evaluator_validation_list))

        self._print("{}: Testing config: {}".format(self.ALGORITHM_NAME, current_fit_parameters))

        for i, evaluator in enumerate(self.evaluator_validation_list):
            recommender_instance, train_times[i] = self._fit_model(current_fit_parameters,
                                                                   self.recommender_input_args_list[i])

            start_time = time.time()

            # Evaluate recommender and get results for the first cutoff
            mean_result_dict, _ = self.evaluator_validation_list[i].evaluateRecommender(recommender_instance)
            mean_result_dict = mean_result_dict[list(mean_result_dict.keys())[0]]
            result_dicts.append(mean_result_dict)

            evaluation_times[i] = time.time() - start_time

        mean_result_dict, std_result_dict = compute_mean_std_result_dict(result_dicts)
        evaluation_time = evaluation_times.sum() / evaluation_times.size
        train_time = train_times.sum() / train_times.size
        result_string = get_result_string(mean_result_dict, std_result_dict, n_mean_decimals=6, n_std_decimals=4)

        return mean_result_dict, std_result_dict, result_string, train_time, evaluation_time

    def _objective_function(self, current_fit_parameters_dict):

        try:

            self.metadata_dict["hyperparameters_list"][self.model_counter] = current_fit_parameters_dict.copy()

            mean_result_dict, std_result_dict, result_string, train_time, evaluation_time = self._evaluate_on_validation(
                current_fit_parameters_dict)

            current_result = - mean_result_dict[self.metric_to_optimize]

            if self.metadata_dict["result_on_validation_best"] is None:
                new_best_config_found = True
            else:
                best_solution_val = self.metadata_dict["result_on_validation_best"][self.metric_to_optimize]
                new_best_config_found = best_solution_val < mean_result_dict[self.metric_to_optimize]

            if new_best_config_found:

                self._write_log("{}: New best config found. Config {}: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                                  self.model_counter,
                                                                                                  current_fit_parameters_dict,
                                                                                                  result_string))


            else:
                self._write_log("{}: Config {} is suboptimal. Config: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                                 self.model_counter,
                                                                                                 current_fit_parameters_dict,
                                                                                                 result_string))

            if current_result >= self.INVALID_CONFIG_VALUE:
                self._write_log(
                    "{}: WARNING! Config {} returned a value equal or worse than the default value to be assigned to invalid configurations."
                    " If no better valid configuration is found, this parameter search may produce an invalid result.\n")

            self.metadata_dict["result_on_validation_list"][self.model_counter] = mean_result_dict.copy()
            self.metadata_dict["std_result_on_validation_list"][self.model_counter] = std_result_dict.copy()

            self.metadata_dict["time_on_train_list"][self.model_counter] = train_time
            self.metadata_dict["time_on_validation_list"][self.model_counter] = evaluation_time

            self.metadata_dict["time_on_train_total"], self.metadata_dict["time_on_train_avg"] = \
                _compute_avg_time_non_none_values(self.metadata_dict["time_on_train_list"])
            self.metadata_dict["time_on_validation_total"], self.metadata_dict["time_on_validation_avg"] = \
                _compute_avg_time_non_none_values(self.metadata_dict["time_on_validation_list"])

            if new_best_config_found:
                self.metadata_dict["hyperparameters_best"] = current_fit_parameters_dict.copy()
                self.metadata_dict["hyperparameters_best_index"] = self.model_counter
                self.metadata_dict["result_on_validation_best"] = mean_result_dict.copy()
                self.metadata_dict["std_result_on_validation_best"] = std_result_dict.copy()

        except (KeyboardInterrupt, SystemExit) as e:
            # If getting a interrupt, terminate without saving the exception
            raise e

        except:
            # Catch any error: Exception, Tensorflow errors etc...

            traceback_string = traceback.format_exc()

            self._write_log("{}: Config {} Exception. Config: {} - Exception: {}\n".format(self.ALGORITHM_NAME,
                                                                                           self.model_counter,
                                                                                           current_fit_parameters_dict,
                                                                                           traceback_string))

            self.metadata_dict["exception_list"][self.model_counter] = traceback_string

            # Assign to this configuration the worst possible score
            # Being a minimization problem, set it to the max value of a float
            current_result = + self.INVALID_CONFIG_VALUE

            traceback.print_exc()

        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save=self.metadata_dict.copy(),
                                  file_name=self.output_file_name_root + "_metadata")

        self.model_counter += 1

        return current_result
