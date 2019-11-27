from numpy.random import seed

from course_lib.Base.Evaluation.Evaluator import Evaluator, EvaluatorHoldout
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_getter import get_warmer_UCM


class EvaluatorCrossValidationKeepKOut(Evaluator):
    EVALUATOR_NAME = "EvaluatorCrossValidationKeepKOut"

    def __init__(self, cutoff, seed_list, data_path, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object=None, ignore_items=None, ignore_users=None, n_folds=10):
        if type(cutoff) != int:
            raise TypeError()
        temp_list = [cutoff]

        if n_folds < 2:
            raise RuntimeError("The number of folds should be at least 2")

        # Just save the parameter, that will be passed to the new hold out validator that will be created
        self.cutoff_list = temp_list
        self.seed_list = seed_list
        self.data_path = data_path
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen
        self.diversity_object = diversity_object
        self.ignore_items = ignore_items
        self.ignore_users = ignore_users
        self.n_folds = n_folds

    def evaluateRecommender(self, recommender_object):
        '''
        Not implemented for this method.
        In this case you should use crossvaluateRecommender

        :param recommender_object: none
        :return: none
        '''
        raise NotImplementedError("The method evaluateRecommender not implemented for this evaluator class")

    def crossevaluateHybridRecommender(self, recommender_class, recommender_fit_parameters, models_classes_dict,
                                       model_constructor_parameters_dict, model_fit_parameters_dict):
        results_list = []

        # For all the folds...
        for i in range(0, self.n_folds):
            current_seed = self.seed_list[i]
            seed(current_seed)
            data_reader = RecSys2019Reader(self.data_path)
            data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                                       force_new_split=True)
            data_reader.load_data()
            URM_train, URM_test = data_reader.get_holdout_split()

            print("Holdout number {}".format(i + 1))

            # Creating recommender instance
            print("Fitting the recommender...")
            recommender_instance = recommender_class(URM_train)

            for model_name, model_class in models_classes_dict.items():
                model_object = model_class(URM_train, **model_constructor_parameters_dict[model_name])
                model_object.fit(**model_fit_parameters_dict[model_name])
                recommender_instance.add_fitted_model(model_name, model_object)

            recommender_instance.fit(**recommender_fit_parameters)

            hold_out_validator = EvaluatorHoldout(URM_test, self.cutoff_list, exclude_seen=self.exclude_seen,
                                                  diversity_object=self.diversity_object,
                                                  ignore_items=self.ignore_items
                                                  , ignore_users=self.ignore_users,
                                                  minRatingsPerUser=self.minRatingsPerUser)

            print("Recommender holdout...", end="")
            fold_result = hold_out_validator.evaluateRecommender(recommender_instance)[0][self.cutoff_list[0]]
            print("..Fold done")

            print("FOLD RESULT IS: " + str(fold_result))

            results_list.append(fold_result)
        return self._get_average_result(results_list, self.n_folds)

    def crossevaluateRecommender(self, recommender_class, **recommender_keywargs):
        '''
        Cross-evaluate the recommender class passed.

        :param recommender_class: recommender class to recommender
        :param recommender_keywargs: non-positional argument of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the X-validation
        '''
        results_list = []

        # For all the folds...
        for i in range(0, self.n_folds):
            current_seed = self.seed_list[i]
            seed(current_seed)
            data_reader = RecSys2019Reader(self.data_path)
            data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                                       force_new_split=True)
            data_reader.load_data()
            URM_train, URM_test = data_reader.get_holdout_split()

            print("Holdout number {}".format(i + 1))

            # Creating recommender instance
            print("Fitting the recommender...")
            recommender_instance = recommender_class(URM_train)
            recommender_instance.fit(**recommender_keywargs)

            hold_out_validator = EvaluatorHoldout(URM_test, self.cutoff_list, exclude_seen=self.exclude_seen,
                                                  diversity_object=self.diversity_object,
                                                  ignore_items=self.ignore_items
                                                  , ignore_users=self.ignore_users,
                                                  minRatingsPerUser=self.minRatingsPerUser)

            print("Recommender holdout...", end="")
            fold_result = hold_out_validator.evaluateRecommender(recommender_instance)[0][self.cutoff_list[0]]
            print("..Fold done")

            print("FOLD RESULT IS: " + str(fold_result))

            results_list.append(fold_result)
        return self._get_average_result(results_list, self.n_folds)

    def crossevaluateDemographicRecommender(self, recommender_class, on_cold_users=False, **recommender_keywargs):
        '''
        Cross-evaluate the recommender class passed that uses UCMs.

        :param recommender_class: recommender class to recommender
        :param on_cold_users: evaluate only on cold users if True, otherwise "self.ignore_users" does not change
        :param recommender_keywargs: non-positional argument of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the X-validation
        '''
        results_list = []

        data_reader = RecSys2019Reader(self.data_path)
        data_reader.load_data()
        URM_all = data_reader.get_URM_all()
        UCM_age = data_reader.get_UCM_from_name("UCM_age")
        UCM_region = data_reader.get_UCM_from_name("UCM_region")
        UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})
        UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)

        # For all the folds...
        for i in range(0, self.n_folds):
            current_seed = self.seed_list[i]
            seed(current_seed)
            data_reader = RecSys2019Reader(self.data_path)
            data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                                       force_new_split=True)
            data_reader.load_data()
            URM_train, URM_test = data_reader.get_holdout_split()
            UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

            ignore_users = self.ignore_users
            if on_cold_users:
                warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
                warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
                ignore_users = warm_users

            print("Holdout number {}".format(i + 1))

            # Creating recommender instance
            print("Fitting the recommender...")
            recommender_instance = recommender_class(URM_train, UCM_all)
            recommender_instance.fit(**recommender_keywargs)

            hold_out_validator = EvaluatorHoldout(URM_test, self.cutoff_list, exclude_seen=self.exclude_seen,
                                                  diversity_object=self.diversity_object,
                                                  ignore_items=self.ignore_items, ignore_users=ignore_users,
                                                  minRatingsPerUser=self.minRatingsPerUser)

            print("Recommender holdout...", end="")
            fold_result = hold_out_validator.evaluateRecommender(recommender_instance)[0][self.cutoff_list[0]]
            print("..Fold done")

            print("FOLD RESULT IS: " + str(fold_result))

            results_list.append(fold_result)
        return self._get_average_result(results_list, self.n_folds)

    def crossevaluateCBFRecommender(self, recommender_class, **recommender_keywargs):
        '''
        Cross-evaluate the recommender class passed that uses ICMs.

        :param recommender_class: recommender class to recommender
        :param on_cold_users: evaluate only on cold users if True, otherwise "self.ignore_users" does not change
        :param recommender_keywargs: non-positional argument of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the X-validation
        '''
        results_list = []

        data_reader = RecSys2019Reader(self.data_path)
        data_reader.load_data()

        # For all the folds...
        for i in range(0, self.n_folds):
            current_seed = self.seed_list[i]
            seed(current_seed)
            data_reader = RecSys2019Reader(self.data_path)
            data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                                       force_new_split=True)
            data_reader.load_data()
            URM_train, URM_test = data_reader.get_holdout_split()
            ICM_sub_class = data_reader.get_ICM_from_name("ICM_sub_class")
            ICM_all, _ = merge_ICM(ICM_sub_class, URM_train.transpose(), {}, {})

            print("Holdout number {}".format(i + 1))

            # Creating recommender instance
            print("Fitting the recommender...")
            recommender_instance = recommender_class(URM_train, ICM_all)
            recommender_instance.fit(**recommender_keywargs)

            hold_out_validator = EvaluatorHoldout(URM_test, self.cutoff_list, exclude_seen=self.exclude_seen,
                                                  diversity_object=self.diversity_object,
                                                  ignore_items=self.ignore_items, ignore_users=self.ignore_users,
                                                  minRatingsPerUser=self.minRatingsPerUser)

            print("Recommender holdout...", end="")
            fold_result = hold_out_validator.evaluateRecommender(recommender_instance)[0][self.cutoff_list[0]]
            print("..Fold done")

            print("FOLD RESULT IS: " + str(fold_result))

            results_list.append(fold_result)
        return self._get_average_result(results_list, self.n_folds)

    def _get_average_result(self, results_list, n_folds):
        average_result = results_list[0]

        # Averaging results
        for key in average_result:
            key_sum = 0
            for i in range(1, n_folds):
                key_sum += results_list[i][key]
            average_result[key] += key_sum
            average_result[key] /= n_folds

        return average_result




class EvaluatorCrossValidation(Evaluator):
    """EvaluatorCrossValidation"""

    EVALUATOR_NAME = "EvaluatorCrossValidation"

    def __init__(self, data_set, cutoff, minRatingsPerUser=1, exclude_seen=True, diversity_object=None,
                 ignore_items=None,
                 ignore_users=None, n_folds=10):
        '''
        Cross validation Recommender

        :param data_set: data set
        :param cutoff_list: the one used in the holdout phase, but here it has to be a number
        :param minRatingsPerUser: the one used in the holdout phase
        :param exclude_seen: the one used in the holdout phase
        :param diversity_object: the one used in the holdout phase
        :param ignore_items: the one used in the holdout phase
        :param ignore_users: the one used in the holdout phase
        :param n_folds: number of folds to cross validate
        '''
        if type(cutoff) != int:
            raise TypeError()
        temp_list = [cutoff]

        if n_folds < 2:
            raise RuntimeError("The number of folds should be at least 2")

        data_set = DataSplitter_Warm_k_fold(data_set, n_folds=n_folds)
        data_set.load_data()

        # Just save the parameter, that will be passed to the new hold out validator that will be created
        self.data_set = data_set
        self.cutoff_list = temp_list
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen
        self.diversity_object = diversity_object
        self.ignore_items = ignore_items
        self.ignore_users = ignore_users
        self.n_folds = n_folds

    def evaluateRecommender(self, recommender_object):
        '''
        Not implemented for this method.
        In this case you should use crossvaluateRecommender

        :param recommender_object: none
        :return: none
        '''
        raise NotImplementedError("The method evaluateRecommender not implemented for this evaluator class")

    def crossevaluateRecommender(self, recommender_class, **recommender_keywargs):
        '''
        Cross-evaluate the recommender class passed.

        :param recommender_class: recommender class to recommender
        :param recommender_keywargs: non-positional argument of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the X-validation
        '''
        results_list = []

        # For all the folds...
        for i in range(0, self.n_folds):
            print("Holdout validation on fold number {}".format(i + 1))
            # Getting data for the validation
            URM_train, URM_test = self.data_set.get_URM_train_for_test_fold(n_test_fold=i)

            # Creating recommender instance
            print("Fitting the recommender...")
            recommender_instance = recommender_class(URM_train)
            recommender_instance.fit(**recommender_keywargs)

            hold_out_validator = EvaluatorHoldout(URM_test, self.cutoff_list, exclude_seen=self.exclude_seen,
                                                  diversity_object=self.diversity_object,
                                                  ignore_items=self.ignore_items
                                                  , ignore_users=self.ignore_users,
                                                  minRatingsPerUser=self.minRatingsPerUser)

            print("Recommender holdout...", end="")
            fold_result = hold_out_validator.evaluateRecommender(recommender_instance)[0][self.cutoff_list[0]]
            print("..Fold done")

            results_list.append(fold_result)

        average_result = results_list[0]

        # Averaging results
        for key in average_result:
            key_sum = 0
            for i in range(1, self.n_folds):
                key_sum += results_list[i][key]
            average_result[key] += key_sum
            average_result[key] /= self.n_folds

        return average_result
