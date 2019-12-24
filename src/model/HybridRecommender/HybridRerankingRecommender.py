from abc import ABC

import numpy as np
import scipy.sparse as sps

from src.model.HybridRecommender.AbstractHybridRecommender import AbstractHybridRecommender


# --------- UTILITY ---------
def get_ui_matrix_from_ranking(ranking, scores, n_users, n_items):
    ranking_ui_matrix = sps.coo_matrix((n_users, n_items))
    item_per_user = ranking.shape[1]
    ranking_ui_matrix.row = np.repeat(np.arange(0, n_users), item_per_user)
    ranking_ui_matrix.col = np.reshape(ranking, newshape=ranking_ui_matrix.row.shape)
    ranking_ui_matrix.data = np.reshape(scores, newshape=ranking_ui_matrix.row.shape)
    return ranking_ui_matrix.tocsr()


class RerankingStrategyInterface(ABC):

    def get_sub_scores_related_to_main_ranking(self, main_ranking, sub_ranking, sub_scores, weight):
        """
        Get new rankings based on the recommendation strategy implemented

        :param main_ranking: main recommendations of the main recommender
        :param sub_ranking: sub recommendations of the sub recommender
        :param sub_scores: sub scores of the sub recommender
        :param weight: weight of the sub recommender
        :return: the new sub scores to be added on the main scores
        """
        raise NotImplementedError("This method has to be implemented by another class")


class VotingStrategy(RerankingStrategyInterface):

    def get_sub_scores_related_to_main_ranking(self, main_ranking: np.ndarray, sub_ranking, sub_scores, weight):
        # Set sub scores all to 1 since it is a voting and each recommender votes the items in the same way
        sub_scores[:, :] = 1

        main_max_item_id = np.max(np.max(main_ranking, axis=-1))
        sub_max_item_id = np.max(np.max(sub_ranking, axis=-1))
        max_item_id = np.maximum(main_max_item_id, sub_max_item_id)
        n_items = max_item_id + 1

        main_ranking_ui_matrix = get_ui_matrix_from_ranking(main_ranking,
                                                            np.ones(shape=main_ranking.shape),
                                                            main_ranking.shape[0], n_items)
        sub_ranking_ui_matrix = get_ui_matrix_from_ranking(sub_ranking, sub_scores * weight,
                                                           main_ranking.shape[0],
                                                           n_items)
        new_sub_scores_ui_matrix = main_ranking_ui_matrix.multiply(sub_ranking_ui_matrix)

        # Reshape the new sub scores matrix into a main_ranking shape
        new_sub_scores = new_sub_scores_ui_matrix[np.arange(main_ranking.shape[0]), main_ranking.T].T.todense()
        return np.array(new_sub_scores)


class WeightedAverageStrategy(RerankingStrategyInterface):

    def get_sub_scores_related_to_main_ranking(self, main_ranking, sub_ranking, sub_scores, weight):
        main_max_item_id = np.max(np.max(main_ranking, axis=-1))
        sub_max_item_id = np.max(np.max(sub_ranking, axis=-1))
        max_item_id = np.maximum(main_max_item_id, sub_max_item_id)
        n_items = max_item_id + 1

        main_ranking_ui_matrix = get_ui_matrix_from_ranking(main_ranking,
                                                            np.ones(shape=main_ranking.shape),
                                                            main_ranking.shape[0], n_items)
        sub_ranking_ui_matrix = get_ui_matrix_from_ranking(sub_ranking, sub_scores * weight,
                                                           main_ranking.shape[0],
                                                           n_items)
        new_sub_scores_ui_matrix = main_ranking_ui_matrix.multiply(sub_ranking_ui_matrix)

        # Reshape the new sub scores matrix into a main_ranking shape
        new_sub_scores = new_sub_scores_ui_matrix[np.arange(main_ranking.shape[0]), main_ranking.T].T.todense()
        return np.array(new_sub_scores)


class HybridRerankingRecommender(AbstractHybridRecommender):
    """
    Hybrid reranking recommender
    """

    RECOMMENDER_NAME = "HybridRerankingRecommender"

    def __init__(self, URM_train, main_recommender):
        super().__init__(URM_train)
        self.hybrid_strategy = None
        self.main_recommender = main_recommender
        self.weights = None
        self.STRATEGY_MAPPER = {"weighted_avg": WeightedAverageStrategy(),
                                "norm_weighted_avg": WeightedAverageStrategy(),
                                "voting": VotingStrategy()}

    def fit(self, main_cutoff=20, sub_cutoff=5, bias=True, strategy="voting", **weights):
        """
        Fit the hybrid model by setting the weight of each recommender

        :param main_cutoff: cutoff of the main recommendations that has to be reranked
        :param bias: True if the main_recommender contributes to the reranking, False otherwise
        :param strategy: an object of strategy pattern that handles the hybrid core functioning of the recommender
        system
        :param sub_cutoff: the multiplier used for multiplying the cutoff for handling more recommended items
        to average
        :return: None
        """
        if strategy not in self.STRATEGY_MAPPER:
            raise ValueError("The strategy name passed does not correspond to the implemented strategy: "
                             "{}".format(self.STRATEGY_MAPPER))
        self.weights = weights
        self.main_cutoff = main_cutoff
        self.sub_cutoff = sub_cutoff
        self.bias = bias
        self.hybrid_strategy = self.STRATEGY_MAPPER[strategy]
        self.normalize = False
        if strategy == "norm_weighted_avg":
            self.normalize = True

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        """
        Recommend "number of cutoff" items to a given user_id_array

        REMARK: The RMSE measure got from this hybrid model is incorrect because the score returned is the one from
                last model in the for
        :param user_id_array:
        :param cutoff:
        :param remove_seen_flag:
        :param items_to_compute:
        :param remove_top_pop_flag:
        :param remove_custom_items_flag:
        :param return_scores:
        :return: for each user in user_id_array, a list of recommended items
        """
        # handle user_id_array in case it is scalar
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        main_ranking, main_scores_all = self.main_recommender.recommend(user_id_array, cutoff=self.main_cutoff,
                                                                        remove_seen_flag=remove_seen_flag,
                                                                        items_to_compute=items_to_compute,
                                                                        remove_top_pop_flag=remove_top_pop_flag,
                                                                        remove_custom_items_flag=remove_custom_items_flag,
                                                                        return_scores=True)
        main_ranking = np.array(main_ranking, dtype=np.int)
        main_scores = np.zeros(shape=main_ranking.shape)
        if self.bias:
            main_scores = main_scores_all[np.arange(len(main_ranking)), main_ranking.T].T

        for idx, recommender_key_value in enumerate(self.models.items()):
            recommender_name, recommender_model = recommender_key_value
            ranking, scores_all = recommender_model.recommend(user_id_array, cutoff=self.sub_cutoff,
                                                              remove_seen_flag=remove_seen_flag,
                                                              items_to_compute=items_to_compute,
                                                              remove_top_pop_flag=remove_top_pop_flag,
                                                              remove_custom_items_flag=remove_custom_items_flag,
                                                              return_scores=True)
            if self.normalize:
                maximum = np.max(scores_all, axis=1).reshape((len(user_id_array), 1))

                scores_batch_for_minimum = scores_all.copy()
                scores_batch_for_minimum[scores_batch_for_minimum == float('-inf')] = np.inf
                minimum = np.min(scores_batch_for_minimum, axis=1).reshape((len(user_id_array), 1))

                denominator = maximum - minimum
                denominator[denominator == 0] = 1.0

                scores_all = (scores_all - minimum) / denominator

            ranking = np.array(ranking, dtype=np.int)
            ranking_scores = scores_all[np.arange(len(ranking)), ranking.T].T
            sub_scores = self.hybrid_strategy.get_sub_scores_related_to_main_ranking(main_ranking=main_ranking,
                                                                                     sub_ranking=ranking,
                                                                                     sub_scores=ranking_scores,
                                                                                     weight=self.weights[
                                                                                         recommender_name])
            main_scores += sub_scores
        main_scores_argsort_indices = np.argsort(-main_scores, axis=1)
        rankings = main_ranking[np.arange(len(main_ranking)), main_scores_argsort_indices.T].T
        rankings = rankings[:, :cutoff]
        if single_user:
            rankings = rankings[0]

        if return_scores:
            return rankings, np.full(shape=(len(user_id_array), self.n_items), fill_value=1)
        else:
            return rankings

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Useless for this recommender
        pass

    def copy(self):
        copy = HybridRerankingRecommender(URM_train=self.URM_train, main_recommender=self.main_recommender)
        copy.models = self.models
        return copy

    def save_model(self, folder_path, file_name=None):
        pass

    @classmethod
    def get_possible_strategies(cls):
        return ["voting", "weighted_avg", "norm_weighted_avg"]
