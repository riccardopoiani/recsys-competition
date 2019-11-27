from abc import ABC

from course_lib.Base.BaseRecommender import BaseRecommender


class IBestModel(ABC):
    """
    Interface for best model classes
    """

    best_parameters = {}
    recommender_class: BaseRecommender.__class__ = None

    @classmethod
    def get_best_parameters(cls):
        return cls.best_parameters


class ICollaborativeModel(IBestModel):

    @classmethod
    def get_model(cls, URM_train):
        model = cls.recommender_class(URM_train)
        model.fit(**cls.best_parameters)
        return model


class IContentModel(IBestModel):

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        model = cls.recommender_class(URM_train=URM_train, ICM_train=ICM_train)
        model.fit(**cls.best_parameters)
        return model
