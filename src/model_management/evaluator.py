import numpy as np
from course_lib.Base.BaseRecommender import BaseRecommender
from scipy import sparse as sps
import pandas as pd


def evaluate_recommender_by_user_demographic(recommender_object: BaseRecommender, URM_train: sps.csr_matrix,
                                             URM_test: sps.csr_matrix, cutoff_list: list, user_demographic: np.ndarray, n_folds: int = 5):
    """
    Split the URM_test into "n_folds" folds based on the list "user_demographic" (e.g. if n_folds=5, then 20% for each fold), then
    evaluate each folds of URM_test and return the metric results.
     - Folds are sorted user_demographic an increasing user activity.

    :param recommender_object: Base.BaseRecommender object to be evaluated
    :param URM_train: csr_matrix
    :param URM_test: csr_matrix
    :param cutoff_list: list of cutoff to evaluate on
    :param user_demographic: the list (np.ndarray) containing a single user demographic for all users in URM_train (in the same order!!)
    :param n_folds: the number of folds you want to divide the URM_test
    :return: a dict of metric results for each cutoff
    """
    if len(user_demographic) != URM_train.shape[0]:
        raise AttributeError("The list user_demographic containing user demographic does not contain the value for each user in "
                             "URM_train")

    from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout

    users_to_evaluate = np.array(list(set(URM_test.tocsc().indices)))
    user_activity_test = user_demographic[users_to_evaluate]
    user_activity_test_indices = np.argsort(user_activity_test)
    users_to_evaluate_ordered = users_to_evaluate[user_activity_test_indices]

    users_array_to_evaluate = np.array_split(users_to_evaluate_ordered, n_folds)

    fold_splits = []

    for fold_index in range(n_folds):
        fold_split = sps.coo_matrix(URM_test.shape)
        fold_split.row = []
        fold_split.col = []
        fold_split.data = []

        curr_fold_users = users_array_to_evaluate[fold_index]

        for user_id in curr_fold_users:
            current_fold_user_profile = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]
            current_fold_user_interactions = URM_test.data[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

            fold_split.row.extend([user_id] * len(current_fold_user_profile))
            fold_split.col.extend(current_fold_user_profile)
            fold_split.data.extend(current_fold_user_interactions)

        fold_split.row = np.array(fold_split.row, dtype=np.int)
        fold_split.col = np.array(fold_split.col, dtype=np.int)
        fold_split.data = np.array(fold_split.data, dtype=np.float)

        fold_splits.append(sps.csr_matrix(fold_split))

    evaluators = [EvaluatorHoldout(fold_split, cutoff_list=cutoff_list) for fold_split in fold_splits]
    return [evaluator.evaluateRecommender(recommender_object)[0] for evaluator in evaluators]


def get_singular_user_metrics(URM_test, recommender_object: BaseRecommender, cutoff=10):
    """
    Return a pandas.DataFrame containing the precision, recall and average precision of all the users

    :param URM_test: URM to be tested on
    :param recommender_object: recommender system to be tested
    :param cutoff: the cutoff to be evaluated on
    :return: pandas.DataFrame containing the precision, recall and average precision of all the users
    """

    from course_lib.Base.Evaluation.metrics import average_precision, precision, recall

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    average_precision_list = []
    precision_list = []
    recall_list = []
    user_list = []

    for user_id in range(n_users):

        if user_id % 10000 == 0:
            print("Evaluated user {} of {}".format(user_id, n_users))

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos : end_pos]

            recommended_items = recommender_object.recommend(user_id, cutoff=cutoff)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            user_list.append(user_id)
            average_precision_list.append(average_precision(is_relevant, relevant_items))
            precision_list.append(precision(is_relevant))
            recall_list.append(recall(is_relevant, relevant_items))

    return pd.DataFrame(data={'user_id': user_list, 'precision': precision_list, 'recall': recall_list,
                              'AP': average_precision_list})
