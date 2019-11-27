from abc import ABC

from course_lib.Base.BaseRecommender import BaseRecommender


class IBestModel(ABC):
    """
    Interface for best model classes
    """

    best_parameters = {}
    recommender_class: BaseRecommender.__class__ = None

    def get_best_parameters(self):
        return self.best_parameters


class ICollaborativeModel(IBestModel):

    def get_model(self, URM_train):
        model = self.recommender_class(URM_train)
        model.fit(**self.best_parameters)
        return model


class IContentModel(IBestModel):

    def get_model(self, URM_train, ICM_train):
        model = self.recommender_class(URM_train=URM_train, ICM_train=ICM_train)
        model.fit(**self.best_parameters)
        return model
