from course_lib.Base.BaseRecommender import BaseRecommender
from typing import List, Dict
import numpy as np


class HybridDemographicRecommender(BaseRecommender):
    user_group_dict: Dict[int, List] = {}
    group_id_list: List[int] = []
    recommender_group_relation: Dict[int, BaseRecommender] = {}

    def __init__(self, URM_train):
        self.max_user_id = 0
        super().__init__(URM_train)

    def reset_groups(self):
        self.user_group_dict = {}
        self.group_id_list = []
        self.recommender_group_relation = {}

    def _verify_user_group_list_(self, new_user_group):
        for id in self.group_id_list:
            group = self.user_group_dict[id]
            for user in group:
                if user in new_user_group:
                    return False

        return True

    def _verify_group_consistency_(self, group_id):
        return False if group_id in self.group_id_list else True

    def _verify_relation_consistency(self, group_id):
        if group_id not in self.group_id_list:
            return False

        if group_id in self.recommender_group_relation.keys():
            return False

        return True

    def add_relation_recommender_group(self, recommender_object: BaseRecommender, group_id: int):
        """
        Add a relation between a recommender object and a group.

        :param recommender_object: recommender object to predicts user in the given group id
        :param group_id: id of the group of users to be predicted with the given recommender object
        :return: None
        """
        if self._verify_relation_consistency(group_id):
            self.recommender_group_relation[group_id] = recommender_object
        else:
            raise RuntimeError("Relation already added for this recommender")

    def add_user_group(self, group_id: int, user_group: List):
        """
        Add a new group id to the group of the users to be predicted with this recommender.
        Each group somehow encodes different characteristics.
        An example of a possible group is user profile length.

        We assume the groups to cover all the users id from [0, max_user_id_to_be_recommended]

        :param group_id: id of the group
        :param user_group: groups of user in this group
        :return: None
        """
        if self._verify_group_consistency_(group_id) and self._verify_user_group_list_(user_group):
            self.group_id_list.append(group_id)
            self.user_group_dict[group_id] = user_group
        else:
            raise RuntimeError("Users are already predicted with another recommender, or a group with "
                               "this ID already exists")

    def fit(self):
        """
        Computes what models should be used for each user

        :return: None
        """
        """
        # Compute max user id
        for user_group in self.user_group_list:
            temp = np.array(user_group).max()
            if temp > self.max_user_id:
                self.max_user_id = temp

        # Build the models_to_be_used array
        self.models_to_be_used = np.zeros(self.max_user_id)
        for i, user_group in enumerate(self.user_group_list):
            group = self.group_id_list[i]
            for user in user_group:
                self.models_to_be_used[user] = self.recommender_group_relation[group]
        """
        self.group_id_list.sort()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Compute for each user, its group, then, do the computations with that recommender
        arr = np.array(user_id_array)

        # Building masks
        mask_list = []
        for group_id in self.group_id_list:
            mask = np.in1d(arr, self.user_group_dict[group_id])
            mask_list.append(mask)

        # User distinctions
        users = []
        for mask in mask_list:
            users.append(arr[mask])

        # Building the predictions
        scores_list = []
        for i, group_id in enumerate(self.group_id_list):
            scores_list.append(
                self.recommender_group_relation[group_id]._compute_item_score(user_id_array=users[i],
                                                                              items_to_compute=items_to_compute))

        # Rebuilding the final scores
        score_first_dim = 0
        for i in range(0, len(users)):
            score_first_dim += scores_list[i].shape[0]

        scores_shape = (score_first_dim, scores_list[0].shape[1])
        scores = np.zeros(shape=scores_shape, dtype=np.float32)

        for i, group_id in enumerate(self.group_id_list):
            scores[mask_list[i]] = scores_list[i]

        return np.array(scores)

    def save_model(self, folder_path, file_name=None):
        pass
