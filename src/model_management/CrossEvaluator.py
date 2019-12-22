from tqdm import tqdm

from course_lib.Base.Evaluation.Evaluator import Evaluator, EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from src.tuning.cross_validation.CrossSearchAbstractClass import compute_mean_std_result_dict, get_result_string


class EvaluatorCrossValidationKeepKOut(Evaluator):
    EVALUATOR_NAME = "EvaluatorCrossValidationKeepKOut"

    def __init__(self, URM_train_list, evaluator_list, cutoff, verbose=True):
        if len(URM_train_list) < 2:
            raise RuntimeError("The number of folds should be at least 2")

        if len(URM_train_list) != len(evaluator_list):
            raise ValueError("The number of URM_train should be the same of the number of evaluators")

        self.URM_train_list = URM_train_list
        self.evaluator_list = evaluator_list
        self.n_folds = len(URM_train_list)
        self.cutoff = cutoff
        self.verbose = verbose

    def evaluateRecommender(self, recommender_object):
        raise NotImplementedError("Method not implemented for this class")

    def crossevaluateRecommender(self, recommender_class, recommender_fit_kwargs):
        """
        Cross-evaluate the recommender class passed.

        :param recommender_class: recommender class to recommender
        :param recommender_kwargs: non-positional argument of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the X-validation
        """
        results_list = []
        for i in range(self.n_folds):
            URM_train = self.URM_train_list[i]
            self._print("FOLD-{} VALIDATING...".format(i + 1))
            recommender_instance = recommender_class(URM_train)
            recommender_instance.fit(**recommender_fit_kwargs)
            fold_result = self.evaluator_list[i].evaluateRecommender(recommender_instance)[0][self.cutoff]
            self._print("FOLD-{} RESULTS: {}".format(i + 1, fold_result))
            results_list.append(fold_result)
        mean_result_dict, std_result_dict = compute_mean_std_result_dict(results_list)
        return get_result_string(mean_result_dict, std_result_dict)

    def crossevaluateDemographicRecommender(self, recommender_class, recommender_fit_kwargs, UCM_train_list):
        """
        Cross-evaluate the recommender class passed that uses UCMs.

        :param recommender_class: recommender class to recommender
        :param UCM_train_list: list of UCM_train respectively of the i-URM_train
        :param recommender_fit_kwargs: dictionary containing the parameters of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the cross-validation
        """
        if len(self.URM_train_list) != len(UCM_train_list):
            raise ValueError("The number of URM_train should be the same of the number ICM_train")

        results_list = []
        for i in range(self.n_folds):
            URM_train = self.URM_train_list[i]
            self._print("FOLD-{} VALIDATING...".format(i + 1))
            recommender_instance = recommender_class(URM_train, UCM_train=UCM_train_list[i])
            recommender_instance.fit(**recommender_fit_kwargs)
            fold_result = self.evaluator_list[i].evaluateRecommender(recommender_instance)[0][self.cutoff]
            self._print("FOLD-{} RESULTS: {}".format(i + 1, fold_result))
            results_list.append(fold_result)
        mean_result_dict, std_result_dict = compute_mean_std_result_dict(results_list)
        return get_result_string(mean_result_dict, std_result_dict)

    def crossevaluateContentRecommender(self, recommender_class, recommender_fit_kwargs, ICM_train_list):
        """
        Cross-evaluate the recommender class passed that uses ICMs.

        :param recommender_class: recommender class to recommender
        :param ICM_train_list: list of ICM_train respectively of the i-URM_train
        :param recommender_fit_kwargs: dictionary containing the parameters of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the cross-validation
        """
        if len(self.URM_train_list) != len(ICM_train_list):
            raise ValueError("The number of URM_train should be the same of the number ICM_train")

        results_list = []
        for i in range(self.n_folds):
            URM_train = self.URM_train_list[i]
            self._print("FOLD-{} VALIDATING...".format(i + 1))
            recommender_instance = recommender_class(URM_train, ICM_train=ICM_train_list[i])
            recommender_instance.fit(**recommender_fit_kwargs)
            fold_result = self.evaluator_list[i].evaluateRecommender(recommender_instance)[0][self.cutoff]
            self._print("FOLD-{} RESULTS: {}".format(i + 1, fold_result))
            results_list.append(fold_result)
        mean_result_dict, std_result_dict = compute_mean_std_result_dict(results_list)
        return get_result_string(mean_result_dict, std_result_dict)


@DeprecationWarning
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
        """
        Not implemented for this method.
        In this case you should use crossvaluateRecommender

        :param recommender_object: none
        :return: none
        """
        raise NotImplementedError("The method evaluateRecommender not implemented for this evaluator class")

    def crossevaluateRecommender(self, recommender_class, **recommender_kwargs):
        """
        Cross-evaluate the recommender class passed.

        :param recommender_class: recommender class to recommender
        :param recommender_kwargs: non-positional argument of the fit method for the recommender class
        :return: dictionary containing the various metric calculated in the X-validation
        """
        results_list = []

        # For all the folds...
        for i in range(0, self.n_folds):
            self._print("Holdout validation on fold number {}".format(i + 1))
            # Getting data for the validation
            URM_train, URM_test = self.data_set.get_URM_train_for_test_fold(n_test_fold=i)

            # Creating recommender instance
            self._print("Fitting the recommender...")
            recommender_instance = recommender_class(URM_train)
            recommender_instance.fit(**recommender_kwargs)

            hold_out_validator = EvaluatorHoldout(URM_test, self.cutoff_list, exclude_seen=self.exclude_seen,
                                                  diversity_object=self.diversity_object,
                                                  ignore_items=self.ignore_items
                                                  , ignore_users=self.ignore_users,
                                                  minRatingsPerUser=self.minRatingsPerUser)

            self._print("Recommender holdout...")
            fold_result = hold_out_validator.evaluateRecommender(recommender_instance)[0][self.cutoff_list[0]]
            self._print("..Fold done")

            results_list.append(fold_result)
        mean_result_dict, std_result_dict = compute_mean_std_result_dict(results_list)
        return get_result_string(mean_result_dict, std_result_dict)
