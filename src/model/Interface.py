from abc import ABC

from course_lib.Base.BaseRecommender import BaseRecommender
from src.utils.general_utility_functions import get_project_root_path
import os

class IBestModel(ABC):
    """
    Interface for best model classes
    """

    best_parameters = {}
    recommender_class: BaseRecommender.__class__ = None
    recommender_name: str = None

    @classmethod
    def get_best_parameters(cls):
        return cls.best_parameters

    @classmethod
    def get_model(cls, *args, **kwargs):
        raise NotImplementedError("get_model is not implemented for this class")

    @classmethod
    def _save_model(cls, model: BaseRecommender):
        root_path = get_project_root_path()
        saved_models_path = os.path.join(root_path, "resources", "saved_models/")
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)

        if cls.recommender_name is None:
            raise ValueError("recommender_name of the model is not defined")

        model.save_model(folder_path=saved_models_path,
                         file_name=cls.recommender_name)

    @classmethod
    def _load_model(cls, model: BaseRecommender):
        root_path = get_project_root_path()
        saved_models_path = os.path.join(root_path, "resources", "saved_models/")
        if not os.path.exists(saved_models_path):
            raise FileNotFoundError("Cannot find model to be loaded")
        model.load_model(folder_path=saved_models_path, file_name=cls.recommender_name)
        return model


class ICollaborativeModel(IBestModel):

    @classmethod
    def get_model(cls, URM_train, load_saved_model=True):
        try:
            if load_saved_model:
                model = cls.recommender_class(URM_train)
                model = cls._load_model(model)
                return model
        except FileNotFoundError as e:
            print("WARNING: Cannot find model to be loaded")

        model = cls.recommender_class(URM_train)
        model.fit(**cls.best_parameters)
        cls._save_model(model)
        return model


class IContentModel(IBestModel):

    @classmethod
    def get_model(cls, URM_train, ICM_train, load_saved_model=True):
        try:
            if load_saved_model:
                model = cls.recommender_class(URM_train=URM_train, ICM_train=ICM_train)
                model = cls._load_model(model)
                return model
        except FileNotFoundError as e:
            print("WARNING: Cannot find model to be loaded")

        model = cls.recommender_class(URM_train=URM_train, ICM_train=ICM_train)
        model.fit(**cls.best_parameters)
        cls._save_model(model)
        return model

