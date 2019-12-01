import multiprocessing

from lightfm import lightfm

from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import numpy as np


class LightFMRecommender(BaseMatrixFactorizationRecommender):
    """ LightFM Recommender """

    RECOMMENDER_NAME = "LightFMRecommender"

    def __init__(self, URM_train, UCM_train=None, ICM_train=None, verbose=True):
        super().__init__(URM_train, verbose)
        self.UCM_train = UCM_train
        self.ICM_train = ICM_train

    def fit(self, epochs=300, no_components=50, user_alpha=0.01, item_alpha=0.01, learning_schedule="adadelta",
            learning_rate=0.05, loss="warp", max_sampled=10, **kwargs):
        self.model = lightfm.LightFM(no_components=no_components, item_alpha=item_alpha, user_alpha=user_alpha,
                                     loss=loss, learning_schedule=learning_schedule, learning_rate=learning_rate,
                                     max_sampled=max_sampled, **kwargs)
        self.model.fit(self.URM_train, user_features=self.UCM_train, item_features=self.ICM_train, epochs=epochs,
                       verbose=True)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items)

        user_ids = np.reshape(np.repeat(np.int32(user_id_array), self.n_items),
                              newshape=(len(user_id_array) * len(items_to_compute)))
        item_ids = np.reshape(np.tile(items_to_compute, reps=(len(user_id_array), 1)), newshape=len(user_ids))
        scores_batch = self.model.predict(user_ids, item_ids, user_features=self.UCM_train,
                                          item_features=self.ICM_train)
        return np.reshape(scores_batch, newshape=(len(user_id_array), len(items_to_compute)))
