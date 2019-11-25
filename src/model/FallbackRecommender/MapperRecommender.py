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
        print(arr)

        # Building masks
        mapper_keys = np.array(list(self.mapper.keys()), dtype=np.int32)
        main_mask = np.in1d(arr, mapper_keys, assume_unique=True, invert=False)  # user to recommend with main
        sub_mask = np.logical_not(main_mask)  # user to recommend with sub
        print(main_mask)
        print(sub_mask)
        # User distinction

        main_users = [user_id for original, user_id in self.mapper.items() if original in arr[main_mask]]
        print(main_users)
        sub_users = [user_id for original, user_id in self.mapper.items() if original in arr[sub_mask]]
        print(sub_users)

        # Computing scores
        main_scores = self.main_model._compute_item_score(user_id_array=main_users, items_to_compute=items_to_compute)
        sub_scores = self.sub_model._compute_item_score(user_id_array=sub_users, items_to_compute=items_to_compute)

        # So far we have to (c, 20k) and (user-id-array-c, 20k) np arrays containing the arrays. Now we
        # have to fix them with the mapper

        # Mixing arrays
        scores_shape = (main_scores.shape[0] + sub_scores.shape[0], main_scores.shape[1])
        scores = np.zeros(shape=scores_shape, dtype=np.float32)

        scores[main_mask] = main_scores
        scores[sub_mask] = sub_scores

        return scores

    def saveModel(self, folder_path, file_name=None):
        pass
