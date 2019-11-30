from course_lib.Base.BaseRecommender import BaseRecommender
import implicit
import numpy as np


class ImplicitALSRecommender(BaseRecommender):
    """ Implicit ALS Recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

    def fit(self, epochs = 300, num_factors=50, regularization=0.01):
        self.model = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=regularization,
                                                          iterations=epochs, calculate_training_loss=True)
        self.model.fit(self.URM_train.T.tocsr())

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        ranking_list = []
        for user in user_id_array:
            items_scores = self.model.recommend(user, self.URM_train, N=cutoff,
                                                filter_already_liked_items=remove_seen_flag)
            ranking_list.append( [item_score[0] for item_score in items_scores])

        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, np.full(shape=(len(user_id_array), self.n_items), fill_value=0)

        else:
            return ranking_list

