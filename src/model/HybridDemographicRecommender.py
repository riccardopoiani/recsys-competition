from course_lib.Base.BaseRecommender import BaseRecommender
from typing import List, Dict
import numpy as np


class HybridDemographicRecommender(BaseRecommender):

    models_object: List[BaseRecommender] = []
    models_name: List[str] = []
    user_group_list: List[List] = [[]]
    group_id_list: List[int] = []
    recommender_group_relation: Dict[int, BaseRecommender]

    def __init__(self, URM_train):
        self.max_user_id = 0
        self.model_ap_size = None
        self.models_to_be_used = None
        super().__init__(URM_train)

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
            self.user_group_list.append(user_group)
        else:
            raise RuntimeError("Users are already predicted with another recommender, or a group with "
                               "this ID already exists")

    def _verify_user_group_list_(self, user_group):
        # TODO test this function
        for group in self.user_group_list:
            group_arr = np.array(group)  # converision to np array
            user_group_arr = np.array(user_group)  # conversion to np array
            mask = np.isin(group_arr, user_group_arr)
            mask = np.nonzero(mask)
            if np.max(mask) == 1:
                return False

        # Inefficient solution
        # for group in self.user_group_list:
        #    for user in user_group:
        #        if user in group:
        #            return False

        return True

    def _verify_name_consistency_(self, name):
        return False if name in self.models_name else True

    def _verify_group_consistency_(self, group_id):
        return False if group_id in self.group_id_list else True

    def add_fitted_model(self, recommender_name: str, recommender_object: BaseRecommender, group_id: np.array):
        """
        Add an already fitted model to the list of model that will be used to compute predictions.
        Models are assumed to be fitted on the same URM_train and validated on the same URM_validation.
        Also, recommendation average precision, are assumed to refer to the same map user-index.

        :param recommender_name: name of the recommender
        :param recommender_object: fitted recommended
        :param group_id: id of the group to be predicted with this recommender
        :return: None
        """
        if not (self._verify_name_consistency_(
                recommender_name)):
            raise AssertionError("The len of the aps of each recommender should be the same. Moreover, the name"
                                 "should not be in the ones already used")
        self.models_name.append(recommender_name)
        self.models_object.append(recommender_object)
        self.recommender_group_relation[group_id] = recommender_object

    def fit(self):
        """
        Computes what models should be used for each user

        :return: None
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


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Compute for each user, its group, then, do the computations with that recommender
        arr = np.array(user_id_array)

        # Building masks
        mask_list = []
        for i, group in enumerate(self.group_id_list):
            mask = np.in1d(self.user_group_list[i], arr)
            mask_list.append(mask)

        # User distinctions
        users = []
        for mask in mask_list:
            users.append(arr[mask])

        # Building the predictions
        scores_list = []
        for i, recommender in enumerate(self.models_object):
            scores_list.append(recommender._compute_item_score(user_id_array=users[i], items_to_compute=items_to_compute))

        # Rebuilding the final scores
        score_first_dim = 0
        for i in range(0, len(users)):
            score_first_dim += scores_list[i].shape[0]

        scores_shape = (score_first_dim, scores_list[0].shape[1])
        scores = np.zeros(shape=scores_shape, dtype=np.float32)

        for i in range(0, len(users)):
            scores[mask_list[i]] = scores_list[i]

        raise NotImplemented()
