from course_lib.Base.BaseRecommender import BaseRecommender
import numpy as np

def compare_with_top_popular_recommender(top_pop_recommender: BaseRecommender, recommender_to_compare: BaseRecommender,
                                         URM, cutoff=10, num_most_pop=100):
    '''
    Calculate the average number of popular items (cutoff num_most_pop) are recommender within the first (cutoff cutoff)
    recommendation of the recommender to compare

    :param top_pop_recommender: top popular recommender
    :param recommender_to_compare: recommender to be compared
    :param URM: URM on which to do the recommendations
    :param cutoff: number of recommendation that recommender_to_compare will do
    :param num_most_pop: number of recommendatio that the top popular recommender will do
    :return: average number of popular items (cutoff num_most_pop) are recommender within the first (cutoff cutoff)
    recommendation of the recommender to compare
    '''
    import numpy as np
    count = 0
    for user_id in range(URM.shape[0]):
        # Getting the recommendation
        recs_to_compare = recommender_to_compare.recommend(user_id, cutoff=cutoff)
        recs_top_pop = top_pop_recommender.recommend(user_id, cutoff=num_most_pop)

        mask = np.in1d(recs_to_compare, recs_top_pop, assume_unique=True, invert=False)
        recs_to_compare = np.array(recs_to_compare)
        common_items = recs_to_compare[mask]

        count += len(common_items)

        if user_id % 10000 == 0:
            print("Recommended to user {}/{}".format(user_id, URM.shape[0]))

    average_count = count / URM.shape[0]
    return average_count

def compare_recommendation(df_first_recommender, df_second_recommender, threshold, metric):
    '''
    Compare the recommendations made by two recommender systems.

    :param df_first_recommender: dataframe containing user_id and metrics (on a validation set) for each user
    and considering the first recommender
    :param df_second_recommender: dataframe containing user_id and metrics (on a validation set) for each user
    and considering the second recommender
    :param threshold: threshold to specify the goodness of a recommendation
    :param metric: metric to be considered
    :return: users in which both recommender performs well, users in which only the first performs well,
    users in which only the second performs well, users in which both performs bad
    '''
    # Getting where you have a true in both the lists
    good_users_first = df_first_recommender[df_first_recommender[metric] > threshold][
        'user_id'].values  # nparray of good users for the first
    good_users_second = df_second_recommender[df_second_recommender[metric] > threshold][
        'user_id'].values  # nparray of good users for the second

    good_mask = np.in1d(good_users_first, good_users_second, assume_unique=True,
                        invert=False)  # True in the mask when you have an item of the second in the first
    good_users_in_both = good_users_first[good_mask]

    # Getting where users are bad in both the lists
    bad_users_first = df_first_recommender[df_first_recommender[metric] < threshold][
        'user_id'].values  # nparray of good users for the first
    bad_users_second = df_second_recommender[df_second_recommender[metric] < threshold][
        'user_id'].values  # nparray of good users for the second

    bad_mask = np.in1d(bad_users_first, bad_users_second, assume_unique=True,
                       invert=False)  # True in the mask when you have an item of the second in the first
    bad_users_in_both = bad_users_first[bad_mask]

    # Getting where you have good recommendations only for the first one
    good_first_only_mask = np.in1d(good_users_first, bad_users_second, assume_unique=True, invert=False)
    good_users_first_only = good_users_first[good_first_only_mask]

    # Getting where you have good recommendations only for the second one
    good_second_only_mask = np.in1d(good_users_second, bad_users_first, assume_unique=True, invert=False)
    good_users_second_only = good_users_second[good_second_only_mask]

    return good_users_in_both, good_users_first_only, good_users_second_only, bad_users_in_both