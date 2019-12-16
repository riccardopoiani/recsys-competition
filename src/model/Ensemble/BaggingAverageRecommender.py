import numpy as np
from tqdm import tqdm

from course_lib.Base.BaseRecommender import BaseRecommender
from src.model.Ensemble.BaggingUtils import get_bootstrap_URM
from src.utils.general_utility_functions import block_print, enable_print, get_split_seed


class BaggingAverageRecommender(BaseRecommender):
    """
    Bagging Average Recommender: samples with replacement only the positive interactions and groups the models
    by averaging the scores
    """

    RECOMMENDER_NAME = "BaggingAverageRecommender"

    def __init__(self, URM_train, recommender_class, do_bootstrap=True, weight_replacement=True,
                 **recommender_constr_kwargs):
        super().__init__(URM_train)

        self.weight_replacement = weight_replacement
        self.do_bootstrap = do_bootstrap
        self.recommender_class = recommender_class
        self.recommender_constr_kwargs = recommender_constr_kwargs
        self.models = []

    def fit(self, num_models=5, hyper_parameters_range=None):
        if hyper_parameters_range is None:
            hyper_parameters_range = {}

        np.random.seed(get_split_seed())
        seeds = np.random.randint(low=0, high=2 ** 32 - 1, size=num_models)
        np.random.seed()

        for i in tqdm(range(num_models), desc="Fitting bagging models"):
            URM_bootstrap = self.URM_train
            if self.do_bootstrap:
                URM_bootstrap = get_bootstrap_URM(self.URM_train, weight_replacement=self.weight_replacement)
            parameters = {}
            for parameter_name, parameter_range in hyper_parameters_range.items():
                parameters[parameter_name] = parameter_range.rvs(random_state=seeds[i])

            block_print()
            recommender_object = self.recommender_class(URM_bootstrap, **self.recommender_constr_kwargs)
            recommender_object.fit(**parameters)
            enable_print()

            self.models.append(recommender_object)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        cum_scores_batch = np.zeros(shape=(len(user_id_array), self.URM_train.shape[1]))

        for recommender_model in self.models:
            scores_batch = recommender_model._compute_item_score(user_id_array, items_to_compute=items_to_compute)
            cum_scores_batch = np.add(cum_scores_batch, scores_batch)
        cum_scores_batch = cum_scores_batch / len(self.models)
        return cum_scores_batch
