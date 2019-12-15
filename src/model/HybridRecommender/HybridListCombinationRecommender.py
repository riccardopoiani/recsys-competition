from abc import ABC

import numpy as np

from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.model.HybridRecommender.AbstractHybridRecommender import AbstractHybridRecommender


class CombinationStrategyInterface(ABC):

    def get_combined_rankings(self, rankings: np.ndarray, weights: np.ndarray, cutoff: int):
        """
        Get new rankings based on the recommendation strategy implemented

        :param rankings: np.ndarray(N_user, cutoff, N_recommender) list of each user's ranking of all recommender system
        :param weights: np.ndarray(N_recommender) list of the weights of all recommender systems
        :param cutoff:
        :return: combined rankings of np.ndarray(N_user, cutoff)
        """
        raise NotImplementedError("This method has to be implemented by another class")


class CountStrategy(CombinationStrategyInterface):

    def get_combined_rankings(self, rankings: np.ndarray, weights: np.ndarray, cutoff: int):
        n_users, cutoff_model, n_recommenders = rankings.shape

        unrolled_rankings = np.reshape(rankings, newshape=(n_users, cutoff_model*n_recommenders))
        combined_rankings = np.empty(shape=(n_users, cutoff), dtype=np.int)
        for i in range(n_users):
            unique_rankings, counts = np.unique(unrolled_rankings[i], return_counts=True)
            top_cutoff_index = np.argsort(-counts)[0:cutoff]
            combined_rankings[i] = unique_rankings[top_cutoff_index]
        return combined_rankings


class RoundRobinStrategy(CombinationStrategyInterface):

    def get_combined_rankings(self, rankings: np.ndarray, weights: np.ndarray, cutoff: int):
        n_users, cutoff_model, n_recommenders = rankings.shape

        round_robin_rankings = np.empty(shape=(n_users, cutoff*n_recommenders))
        for i in range(cutoff):
            ith_rank = rankings[:, i, :]
            round_robin_rankings[:, i*n_recommenders:i*n_recommenders + n_recommenders] = ith_rank

        combined_rankings = np.empty(shape=(n_users, cutoff), dtype=np.int)
        for i in range(n_users):
            unique_rankings, indices = np.unique(round_robin_rankings[i], return_index=True)
            indices = np.sort(indices)
            combined_rankings[i] = round_robin_rankings[i, indices[:cutoff]]
        return combined_rankings


class HybridListCombinationRecommender(AbstractHybridRecommender):
    """
    Hybrid recommender based on list combination of different recommender models: the merge of rankings
    is done only on a specific size of ranking (i.e. cutoff * cutoff_multiplier)
    """

    RECOMMENDER_NAME = "HybridListCombinationRecommender"

    def __init__(self, URM_train):
        super().__init__(URM_train)
        self.hybrid_strategy = None
        self.multiplier_cutoff = None
        self.weights = None
        self.STRATEGY_MAPPER = {"round_robin": RoundRobinStrategy(), "count": CountStrategy()}

    def fit(self, multiplier_cutoff=5, strategy="round_robin", **weights):
        """
        Fit the hybrid model by setting the weight of each recommender

        :param strategy: an object of strategy pattern that handles the hybrid core functioning of the recommender
        system
        :param multiplier_cutoff: the multiplier used for multiplying the cutoff for handling more recommended items
        to average
        :return: None
        """
        if strategy not in self.STRATEGY_MAPPER:
            raise ValueError("The strategy name passed does not correspond to the implemented strategy: "
                             "{}".format(self.STRATEGY_MAPPER))
        self.weights = weights
        self.multiplier_cutoff = multiplier_cutoff
        self.hybrid_strategy = self.STRATEGY_MAPPER[strategy]
        self.sub_recommender = TopPop(URM_train=self.URM_train)
        self.sub_recommender.fit()

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

        cutoff_model = cutoff * self.multiplier_cutoff
        sub_rankings = self.sub_recommender.recommend(user_id_array, cutoff=cutoff_model,
                                                 remove_seen_flag=remove_seen_flag,
                                                 items_to_compute=items_to_compute,
                                                 remove_top_pop_flag=remove_top_pop_flag,
                                                 remove_custom_items_flag=remove_custom_items_flag)

        rankings = np.zeros(shape=(len(user_id_array), cutoff_model, len(self.models.keys())))
        for idx, recommender_key_value in enumerate(self.models.items()):
            recommender_name, recommender_model = recommender_key_value
            ranking = recommender_model.recommend(user_id_array, cutoff=cutoff_model,
                                                  remove_seen_flag=remove_seen_flag,
                                                  items_to_compute=items_to_compute,
                                                  remove_top_pop_flag=remove_top_pop_flag,
                                                  remove_custom_items_flag=remove_custom_items_flag)
            # Fill empty rankings due to cold users
            for i in range(len(ranking)):
                if len(ranking[i]) == 0:
                    ranking[i] = sub_rankings[i]
            rankings[:, :, idx] = ranking

        rankings = self.hybrid_strategy.get_combined_rankings(rankings=rankings,
                                                              weights=self.weights, cutoff=cutoff)

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
        copy = HybridListCombinationRecommender(URM_train=self.URM_train)
        copy.models = self.models
        copy.weights = self.weights
        copy.multiplier_cutoff = self.multiplier_cutoff
        return copy

    @classmethod
    def get_possible_strategies(cls):
        return ["count", "round_robin"]
