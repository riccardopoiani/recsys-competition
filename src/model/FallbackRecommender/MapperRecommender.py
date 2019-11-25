from course_lib.Base.BaseRecommender import BaseRecommender
from typing import Dict
import numpy as np

class MapperRecommender(BaseRecommender):

    def __init__(self, URM_train):
        self.main_model: BaseRecommender = None
        self.sub_model: BaseRecommender = None
        self.mapper: Dict = {}
        super().__init__(URM_train)

    def fit(self, main_recommender: BaseRecommender = None, sub_recommender: BaseRecommender = None,
            mapper: Dict = None):
        self.main_model = main_recommender
        self.sub_model = sub_recommender
        self.mapper = mapper

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        arr = np.array(user_id_array)

        # Building masks
        mapper_keys = np.array(list(self.mapper.keys()), dtype=np.int32)
        main_mask = np.in1d(arr, mapper_keys, assume_unique=True, invert=False)  # user to recommend with main
        sub_mask = np.logical_not(main_mask)  # user to recommend with sub
        # User distinction

        main_users = [self.mapper[original] for original in self.mapper.keys() if int(original) in arr[main_mask]]
        sub_users = arr[sub_mask]

        # Computing scores
        main_scores = self.main_model._compute_item_score(user_id_array=main_users, items_to_compute=items_to_compute)
        sub_scores = self.sub_model._compute_item_score(user_id_array=sub_users, items_to_compute=items_to_compute)

        # So far we have to (c, 20k) and (user-id-array-c, 20k) np arrays containing the arrays. Now we
        # have to fix them with the mapper

        # Mixing arrays
        scores = np.empty(shape=(len(user_id_array), max(main_scores.shape[1], sub_scores.shape[1])), dtype=np.float64)

        scores[main_mask, :] = main_scores
        scores[sub_mask, :] = sub_scores

        return scores

    def saveModel(self, folder_path, file_name=None):
        pass
