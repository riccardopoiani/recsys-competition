from abc import ABC

from course_lib.Base.BaseRecommender import BaseRecommender
from typing import Dict


class AbstractHybridRecommender(BaseRecommender, ABC):
    RECOMMENDER_NAME = "AbstractHybridRecommender"

    def __init__(self, URM_train):
        """
        :param URM_train: The URM train, but it is useless to add
        """
        super().__init__(URM_train)
        self.models: Dict[str, BaseRecommender] = {}
        self.weights: Dict[str, float] = {}

    def add_fitted_model(self, recommender_name: str, recommender_object: BaseRecommender):
        """
        Add an already fitted model to the hybrid

        :param recommender_name: The unique identifier name of the recommender
        :param recommender_object: the recommender model to be added
        :return:
        """
        if not self._verify_name_consistency(recommender_name):
            raise AssertionError("The recommender name is already used. Choose another one")
        self.models[recommender_name] = recommender_object

    def get_number_of_models(self):
        return len(self.models)

    def get_recommender_names(self):
        return list(self.models.keys())

    def _verify_name_consistency(self, name):
        return False if name in list(self.models.keys()) else True

    def copy(self):
        raise NotImplementedError("Method not implemented")

    def saveModel(self, folder_path, file_name=None):
        # No need to save model or load it
        pass
