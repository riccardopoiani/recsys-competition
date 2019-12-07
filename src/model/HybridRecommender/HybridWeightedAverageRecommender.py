import numpy as np
from src.model.HybridRecommender.AbstractHybridRecommender import AbstractHybridRecommender


class HybridWeightedAverageRecommender(AbstractHybridRecommender):
    """
    Pure weighted average hybrid recommender: the weighted average of scores is calculated over all scores
    """

    RECOMMENDER_NAME = "HybridWeightedAverageRecommender"

    def __init__(self, URM_train, normalize=True):
        super().__init__(URM_train)

        self.normalize = normalize

    def fit(self, **weights):
        """
        Fit the hybrid model by setting the weight of each recommender

        :param normalize: if True, then each recommender score is normalized
        :param weights: the list of weight of each recommender
        :return: None
        """
        if np.all(np.in1d(weights.keys(), list(self.models.keys()), assume_unique=True)):
            raise ValueError("The weights key name passed does not correspond to the name of the models inside the "
                             "hybrid recommender: {}".format(self.get_recommender_names()))
        self.weights = weights

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Compute weighted average scores from all the recommenders added. If normalize is true, then each recommender
        score will be normalized before weighted.

        :param user_id_array:
        :param items_to_compute:
        :return: weighted average scores of the hybrid model
        """
        cum_scores_batch = np.zeros(shape=(len(user_id_array), self.URM_train.shape[1]))

        for recommender_name, recommender_model in self.models.items():
            scores_batch = recommender_model._compute_item_score(user_id_array, items_to_compute=items_to_compute)

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
        cum_scores_batch = cum_scores_batch / len(self.models.keys())
        return cum_scores_batch

    def copy(self):
        copy = HybridWeightedAverageRecommender(URM_train=self.URM_train)
        copy.models = self.models
        copy.weights = self.weights
        copy.normalize = self.normalize
        return copy
