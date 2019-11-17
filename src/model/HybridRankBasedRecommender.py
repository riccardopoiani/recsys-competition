import numpy as np
from src.model.AbstractHybridRecommender import AbstractHybridRecommender
from typing import Dict, List


class CumulativeScoreBuilder(object):

    def __init__(self):
        self.scores: Dict[int, float] = {}

    def add_score(self, item, score):
        if self.scores.get(item) is None:
            self.scores[item] = score
        else:
            self.scores[item] += score

    def add_scores(self, ranking, scores):
        for item, score in zip(ranking, scores):
            self.add_score(item, score)

    def get_ranking(self, cutoff=None):
        scores_ordered_indices = np.argsort(list(self.scores.values()))[::-1]
        ranking = np.array(list(self.scores.keys()))[scores_ordered_indices]

        if cutoff is None:
            return ranking.tolist()
        else:
            return ranking[:cutoff].tolist()

######## HYBRID STRATEGIES ########

class RecommendationStrategyInterface():

    def get_hybrid_rankings_and_scores(self, rankings: list, scores: np.ndarray, weight: float):
        """
        Get new rankings and new scores based on the recommendation strategy implemented

        :param rankings: list of each user's ranking of a recommender system
        :param scores: np.ndarray of each user's scores of all items
        :param weight: the weight of a recommender system
        :return:
        """
        raise NotImplementedError("This method has to be implemented by another class")


class WeightedAverageStrategy(RecommendationStrategyInterface):

    def __init__(self, normalize = True):
        self.normalize = normalize

    def get_hybrid_rankings_and_scores(self, rankings: list, scores: np.ndarray, weight: float):

        # indexing scores by ranking list
        ranking_scores = scores[np.arange(len(rankings)), np.array(rankings).T].T

        # min-max normalization
        if self.normalize:
            maximum = np.max(ranking_scores, axis=1).reshape((len(ranking_scores), 1))

            scores_batch_for_minimum = ranking_scores.copy()
            scores_batch_for_minimum[scores_batch_for_minimum == float('-inf')] = np.inf
            minimum = np.min(scores_batch_for_minimum, axis=1).reshape((len(ranking_scores), 1))

            denominator = maximum - minimum
            denominator[denominator == 0] = 1.0

            ranking_scores = (ranking_scores - minimum) / denominator

        weighted_scores = weight * ranking_scores
        return rankings, weighted_scores


class WeightedCountStrategy(RecommendationStrategyInterface):

    def get_hybrid_rankings_and_scores(self, rankings: list, scores: np.ndarray, weight: float):
        ranking_scores = np.ones(shape=(np.array(rankings).shape))
        weighted_scores = weight * ranking_scores
        return rankings, weighted_scores


class WeightedRankingStrategy(RecommendationStrategyInterface):

    def get_hybrid_rankings_and_scores(self, rankings: list, scores: np.ndarray, weight: float):
        ranking_scores = np.tile(np.arange(1, len(rankings[0])+1)[::-1], reps=(len(rankings), 1))
        weighted_scores = weight * ranking_scores
        return rankings, weighted_scores



########## Hybrid Recommender ##########

class HybridRankBasedRecommender(AbstractHybridRecommender):
    """
    Hybrid recommender based on rankings of different recommender models: the weighted merge of rankings is done only
    on a specific size of ranking (i.e. cutoff * cutoff_multiplier)
    """

    RECOMMENDER_NAME = "HybridRankBasedRecommender"

    def __init__(self, URM_train, hybrid_strategy = WeightedAverageStrategy(), multiplier_cutoff=5):
        """

        :param URM_train:
        :param hybrid_strategy: an object of strategy pattern tha'''t handles the hybrid core functioning of the recommender
        system
        :param multiplier_cutoff: the multiplier used for multiplying the cutoff for handling more recommended items
        to average
        """
        super().__init__(URM_train)

        self.hybrid_strategy = hybrid_strategy
        self.multiplier_cutoff = multiplier_cutoff
        self.weights = None

    def fit(self, **weights):
        """
        Fit the hybrid model by setting the weight of each recommender

        :param weights: the list of weight of each recommender
        :return: None
        """
        if np.all(np.in1d(weights.keys(), list(self.models.keys()), assume_unique=True)):
            raise ValueError("The weights key name passed does not correspond to the name of the models inside the "
                             "hybrid recommender: {}".format(self.get_recommender_names()))
        self.weights = weights

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        """
        Recommend "number of cutoff" items to a given user_id_array

        REMARK: The RMSE measure got from this hybrid model is incorrect because the score returned is the one from
                last model in the for
        :param user_id_array:
        :param cutoff:
        :param remove_seen_flag:
        :param items_to_compute:
        :param remove_top_pop_flag:
        :param remove_CustomItems_flag:
        :param return_scores:
        :return: for each user in user_id_array, a list of recommended items
        """

        # handle user_id_array in case it is scalar
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        all_cum_score_builder: List[CumulativeScoreBuilder] = []
        for i in range(len(user_id_array)):
            all_cum_score_builder.append(CumulativeScoreBuilder())

        cutoff_model = cutoff * self.multiplier_cutoff
        scores = []

        for recommender_name, recommender_model in self.models.items():
            rankings, scores = recommender_model.recommend(user_id_array, cutoff=cutoff_model, remove_seen_flag=remove_seen_flag,
                                                           items_to_compute=items_to_compute, remove_top_pop_flag=remove_top_pop_flag,
                                                           remove_CustomItems_flag=remove_CustomItems_flag, return_scores=True)
            rankings, weighted_scores = self.hybrid_strategy.get_hybrid_rankings_and_scores(rankings, scores,
                                                                                            self.weights[recommender_name])
            for i in range(len(all_cum_score_builder)):
                all_cum_score_builder[i].add_scores(rankings[i], weighted_scores[i])

        weighted_rankings = [all_cum_score_builder[i].get_ranking(cutoff=cutoff) for i in
                             range(len(all_cum_score_builder))]

        if single_user:
            weighted_rankings = weighted_rankings[0]

        if return_scores:
            return weighted_rankings, scores
        else:
            return weighted_rankings

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Useless for this recommender
        pass
