import numpy as np
from course_lib.Base.BaseRecommender import BaseRecommender
from typing import Dict

class HybridWeightedAverageRecommender(BaseRecommender):
    """
    Pure weighted average hybrid recommender: the weighted average of scores is calculated over all scores
    """

    RECOMMENDER_NAME = "HybridWeightedAverageRecommender"

    def __init__(self, URM_train):
        """

        :param URM_train: The URM train, but it is useless to add

        """
        super().__init__(URM_train)
        self.models: Dict[str, BaseRecommender] = {}
        self.weights: Dict[str, float] = {}


    def add_fitted_model(self, recommender_name: str, recommender_object: BaseRecommender):
        """
        Add an already fitted model to the hybrid

        :param recommender_name: The unique identifier name of the recommender
        :param recommender_object: the recommender model to be added
        :return:
        """
        self.models[recommender_name] = recommender_object

    def get_number_of_models(self):
        return len(self.models)

    def get_recommender_names(self):
        return list(self.models.keys())

    def fit(self, normalize=True, **weights):
        """
        Fit the hybrid model by setting the weight of each recommender

        :param normalize: if True, then each recommender score is normalized
        :param weights: the list of weight of each recommender
        :return: None
        """
        if np.all(np.in1d(weights.keys(), list(self.models.keys()), assume_unique=True)):
            raise ValueError("The weights key name passed does not correspond to the name of the model inside the "
                             "hybrid recommender: {}".format(self.get_recommender_names()))
        self.normalize = normalize
        self.weights = weights

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        Compute weighted average scores from all the recommenders added. If normalize is true, then each recommender
        score will be normalized before weighted.

        :param user_id_array:
        :param items_to_compute:
        :return: weighted average scores of the hybrid model
        """
        cum_scores_batch = np.zeros(shape=(len(user_id_array), self.URM_train.shape[1]))

        for recommender_name, recommender_model in self.models.items():
            scores_batch = recommender_model._compute_item_score(user_id_array, items_to_compute = items_to_compute)

            if self.normalize:
                maximum = np.max(scores_batch, axis=1).reshape((len(user_id_array), 1))

                scores_batch_for_minimum = scores_batch.copy()
                scores_batch_for_minimum[scores_batch_for_minimum == float('-inf')] = np.inf
                minimum = np.min(scores_batch_for_minimum, axis=1).reshape((len(user_id_array), 1))

                denominator = maximum - minimum
                denominator[denominator == 0] = 1.0

                scores_batch = (scores_batch - minimum) / denominator

            scores_batch = scores_batch * self.weights[recommender_name]
            cum_scores_batch = np.add(cum_scores_batch, scores_batch)
        return cum_scores_batch
