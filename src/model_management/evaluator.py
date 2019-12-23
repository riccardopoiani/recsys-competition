import numpy as np
from tqdm import tqdm

from course_lib.Base.BaseRecommender import BaseRecommender
from scipy import sparse as sps
import pandas as pd
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.utils.general_utility_functions import block_print


def evaluate_recommender_by_item_content(recommender_object: BaseRecommender, URM_train, URM_test,
                                         cutoff: int, content: list, metric="MAP", exclude_cold_items=False):
    """
    Evaluate the recommender object, computing a metric score of the given recommender in terms of some item
    content

    :param recommender_object: object to be analyzed
    :param URM_train:
    :param URM_test:
    :param cutoff:
    :param content: list of list. In each list, you have the users of the group.
    The MAP will be evaluated on these groups
    :param metric: metric to be considered
    :param exclude_cold_items: if cold items should be excluded from the plot
    :return: desired metric for each group, number of items present in the URM_test for each group
    """
    # Evaluate each group
    total_items = np.arange(URM_train.shape[1])

    result_per_group = []
    number_of_items_in_URM_test_per_group = []
    total_num_ratings_in_test = URM_test.getnnz()

    for group in content:
        mask_items_of_others_groups = np.logical_not(np.in1d(total_items, group))
        items_of_others_groups = total_items[mask_items_of_others_groups]

        if exclude_cold_items:
            cold_items_mask = np.ediff1d(URM_train.tocsc().indptr) == 0
            cold_items = total_items[cold_items_mask]

            # Mixing the two array
            items_to_keep_out = np.concatenate((cold_items, items_of_others_groups))
            items_to_keep_out = np.unique(items_to_keep_out)
        else:
            items_to_keep_out = items_of_others_groups

        num_ratings_removed_in_test = URM_test[:, items_to_keep_out].getnnz()
        number_of_items_in_URM_test_per_group.append(total_num_ratings_in_test - num_ratings_removed_in_test)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_items=items_to_keep_out)
        results = evaluator.evaluateRecommender(recommender_object)
        result_per_group.append(results[0][cutoff][metric])

    return result_per_group, number_of_items_in_URM_test_per_group


def evaluate_recommender_by_demographic(recommender_object: BaseRecommender, URM_train: sps.csr_matrix,
                                        URM_test: sps.csr_matrix, cutoff: int, demographic: list, metric="MAP",
                                        exclude_cold_users=False, foh=-1):
    """
    Evaluate the recommender object, computing the MAP@10 score of the given recommender in terms of some given
    demographic.

    :param recommender_object: object to be analyzed
    :param URM_train: csr_matrix
    :param URM_test: csr_matrix
    :param cutoff: cutoff to evaluate. Only an integer can be given
    :param demographic: list of list. In each list, you have the users of the group. MAP will be evaluate on these
    groups
    :param metric: metric to be considered
    :param exclude_cold_users: if cold users should be excluded from the plot
    :param foh: exclude users with ratings < foh
    :return: desired metric for each group, and support for the metric
    """
    # Evaluate each group
    total_users = np.arange(URM_train.shape[0])

    result_per_group = []
    number_of_items_in_URM_test_per_group = []
    total_num_ratings_in_test = URM_test.getnnz()

    for group in demographic:
        mask_users_of_others_groups = np.logical_not(np.in1d(total_users, group))
        users_of_others_groups = total_users[mask_users_of_others_groups]

        if foh != -1:
            foh_mask = np.ediff1d(URM_train.tocsr().indptr) < foh
            foh_users = total_users[foh_mask]

            users_to_keep_out = np.concatenate((foh_users, users_of_others_groups))
            users_to_keep_out = np.unique(users_to_keep_out)
        elif exclude_cold_users:
            cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
            cold_users = total_users[cold_users_mask]

            # Mixing the two arrays
            users_to_keep_out = np.concatenate((cold_users, users_of_others_groups))
            users_to_keep_out = np.unique(users_to_keep_out)
        else:
            users_to_keep_out = users_of_others_groups

        num_ratings_removed_in_test = URM_test[users_to_keep_out].getnnz()
        number_of_items_in_URM_test_per_group.append(total_num_ratings_in_test - num_ratings_removed_in_test)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_to_keep_out)
        results = evaluator.evaluateRecommender(recommender_object)
        result_per_group.append(results[0][cutoff][metric])

    return result_per_group, number_of_items_in_URM_test_per_group


def evaluate_recommender_by_user_demographic(recommender_object: BaseRecommender, URM_train: sps.csr_matrix,
                                             URM_test: sps.csr_matrix, cutoff_list: list, user_demographic: np.ndarray,
                                             n_folds: int = 5):
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
        raise AttributeError(
            "The list user_demographic containing user demographic does not contain the value for each user in "
            "URM_train")

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
            relevant_items = URM_test.indices[start_pos: end_pos]

            recommended_items = recommender_object.recommend(user_id, cutoff=cutoff)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            user_list.append(user_id)
            average_precision_list.append(average_precision(is_relevant, relevant_items))
            precision_list.append(precision(is_relevant))
            recall_list.append(recall(is_relevant, relevant_items))

    return pd.DataFrame(data={'user_id': user_list, 'precision': precision_list, 'recall': recall_list,
                              'AP': average_precision_list})
